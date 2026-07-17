<#
.SYNOPSIS
    Install build dependencies for arenai_desktop on Windows.
    Checks prerequisites, sets up vcpkg, installs packages (glfw3, glm, bullet3,
    Vulkan headers + loader) and LibTorch (CPU or CUDA). The Vulkan runtime
    driver comes from the GPU driver (system vulkan-1.dll + ICD); pass
    -SoftwareVulkan to also fetch Mesa3D's software Vulkan (lavapipe) as a
    fallback for GPU-less machines.

.USAGE
    # CPU (default)
    .\install_dependencies.ps1

    # CUDA
    .\install_dependencies.ps1 -Cuda

    # Custom libtorch destination
    .\install_dependencies.ps1 -LibtorchPath "C:\libtorch"

    # Also fetch the Mesa3D software Vulkan fallback (lavapipe)
    .\install_dependencies.ps1 -SoftwareVulkan
#>

param(
    [switch]$Cuda,
    [switch]$SoftwareVulkan,
    [string]$LibtorchPath = "",
    [string]$VcpkgRoot = "$PSScriptRoot\libs\vcpkg"
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Assert-Command($cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "$cmd is not installed or not in PATH."
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
Write-Step "Checking prerequisites"

Assert-Command "git"

$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) {
    Write-Error "Visual Studio not found. Install VS 2022 with the 'Desktop development with C++' workload."
    exit 1
}
$vsPath = & $vsWhere -latest -prerelease -property installationPath
$vsVersion = & $vsWhere -latest -prerelease -property installationVersion
Write-Host "  Visual Studio: $vsPath ($vsVersion)"

$vcvarsall = "$vsPath\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvarsall)) {
    Write-Error "vcvarsall.bat not found at $vcvarsall. Make sure the 'Desktop development with C++' workload is installed."
    exit 1
}
Write-Host "  Initializing MSVC environment (x64)..."
cmd /c "`"$vcvarsall`" x64 > nul 2>&1 && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}

# ---------------------------------------------------------------------------
# vcpkg
# ---------------------------------------------------------------------------
Write-Step "Setting up vcpkg"

if (-not (Test-Path "$VcpkgRoot\vcpkg.exe")) {
    if (-not (Test-Path $VcpkgRoot)) {
        git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
    }
    & "$VcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
}

Write-Host "  vcpkg: $VcpkgRoot\vcpkg.exe"

# ---------------------------------------------------------------------------
# Install vcpkg packages (glfw3, glm, bullet3)
# Mesa is NOT built from source anymore: GLES3/EGL headers + import libs come
# from the mesa-dist-win "devel" package, runtime DLLs from "release" (below).
# ---------------------------------------------------------------------------
Write-Step "Installing glfw3 glm bullet3 freetype vulkan via vcpkg"

# freetype: font engine of RmlUi (fetched by arenai_view through FetchContent)
# vulkan-headers + vulkan-loader: build-time Vulkan dependency of arenai_view
# (the runtime driver comes from the GPU driver's ICD)
& "$VcpkgRoot\vcpkg.exe" install glfw3:x64-windows glm:x64-windows bullet3:x64-windows gtest:x64-windows freetype:x64-windows vulkan-headers:x64-windows vulkan-loader:x64-windows
if ($LASTEXITCODE -ne 0) { Write-Error "vcpkg install failed"; exit 1 }

# ---------------------------------------------------------------------------
# Optional: Mesa3D software Vulkan fallback (lavapipe) from mesa-dist-win.
# The primary Vulkan runtime is the GPU driver's (system vulkan-1.dll + ICD);
# this bundle only serves GPU-less machines. CMake copies the DLLs next to
# the executable at build time when libs\mesa exists.
# ---------------------------------------------------------------------------
$mesaDir = "$PSScriptRoot\libs\mesa"

if (-not $SoftwareVulkan) {
    Write-Host "`n  (software Vulkan fallback skipped; pass -SoftwareVulkan to fetch Mesa3D lavapipe)"
} elseif (Test-Path "$mesaDir\x64\lvp_icd.x86_64.json") {
    Write-Step "Mesa3D software Vulkan already present at $mesaDir"
} else {
    Write-Step "Setting up Mesa3D software Vulkan (prebuilt from pal1000/mesa-dist-win)"

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    New-Item -ItemType Directory -Force -Path "$PSScriptRoot\libs" | Out-Null

    # mesa-dist-win ships .7z; fetch the standalone 7zr.exe to extract it
    $sevenZr = "$PSScriptRoot\libs\7zr.exe"
    if (-not (Test-Path $sevenZr)) {
        Invoke-WebRequest -Uri "https://www.7-zip.org/a/7zr.exe" -OutFile $sevenZr -UseBasicParsing
    }

    Write-Host "  Querying latest mesa-dist-win release..."
    $headers = @{ "User-Agent" = "arenai-installer" }
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/pal1000/mesa-dist-win/releases/latest" -Headers $headers

    $asset = $release.assets |
        Where-Object { $_.name -like "*release-msvc.7z" -and $_.name -notlike "*debug*" } |
        Select-Object -First 1
    if (-not $asset) { Write-Error "No 'release-msvc.7z' asset found in mesa-dist-win latest release"; exit 1 }

    $tmp = "$PSScriptRoot\libs\$($asset.name)"
    Write-Host "  Downloading $($asset.name)..."
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $tmp -UseBasicParsing
    if (Test-Path $mesaDir) { Remove-Item -Recurse -Force $mesaDir }
    New-Item -ItemType Directory -Force -Path $mesaDir | Out-Null
    & $sevenZr x $tmp "-o$mesaDir" -y | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Error "7zr extraction failed: $($asset.name)"; exit 1 }
    Remove-Item $tmp

    if (-not (Test-Path "$mesaDir\x64\lvp_icd.x86_64.json")) {
        Write-Error "Mesa extraction failed - x64\lvp_icd.x86_64.json not found in $mesaDir"; exit 1
    }
    Write-Host "  Mesa3D runtime: $mesaDir\x64"
    Write-Host "  To force lavapipe, set VK_DRIVER_FILES=$mesaDir\x64\lvp_icd.x86_64.json"
}

# ---------------------------------------------------------------------------
# LibTorch
# ---------------------------------------------------------------------------
Write-Step "Setting up LibTorch"

if ($LibtorchPath -eq "") {
    $LibtorchPath = "$PSScriptRoot\libs\libtorch"
}

New-Item -ItemType Directory -Force -Path "$PSScriptRoot\libs" | Out-Null

if (-not (Test-Path "$LibtorchPath\share\cmake\Torch\TorchConfig.cmake")) {
    Write-Host "  Downloading LibTorch..."

    if ($Cuda) {
        $torchUrl = "https://download.pytorch.org/libtorch/cu132/libtorch-win-shared-with-deps-2.12.1%2Bcu132.zip"
    } else {
        $torchUrl = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.12.1%2Bcpu.zip"
    }

    $zipPath = "$PSScriptRoot\libs\libtorch.zip"

    Write-Host "  URL: $torchUrl"
    Invoke-WebRequest -Uri $torchUrl -OutFile $zipPath -UseBasicParsing

    Write-Host "  Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath "$PSScriptRoot\libs" -Force
    Remove-Item $zipPath

    if (-not (Test-Path "$LibtorchPath\share\cmake\Torch\TorchConfig.cmake")) {
        Write-Error "LibTorch extraction failed - TorchConfig.cmake not found in $LibtorchPath"
        exit 1
    }
} else {
    Write-Host "  Using existing LibTorch at $LibtorchPath"
}

Write-Step "Dependencies installed successfully!"
Write-Host "  vcpkg:    $VcpkgRoot" -ForegroundColor Green
if (Test-Path "$mesaDir\x64") {
    Write-Host "  Mesa:     $mesaDir\x64 (software Vulkan fallback)" -ForegroundColor Green
}
Write-Host "  LibTorch: $LibtorchPath" -ForegroundColor Green
