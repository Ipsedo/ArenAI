<#
.SYNOPSIS
    Install build dependencies for arenai_desktop on Windows.
    Checks prerequisites, sets up vcpkg, installs packages (glfw3, glm, bullet3,
    Khronos GLES3/EGL headers), downloads prebuilt Mesa3D runtime DLLs
    (pal1000/mesa-dist-win, GPU-capable) and LibTorch (CPU or CUDA).

.USAGE
    # CPU (default)
    .\install_dependencies.ps1

    # CUDA
    .\install_dependencies.ps1 -Cuda

    # Custom libtorch destination
    .\install_dependencies.ps1 -LibtorchPath "C:\libtorch"

    # Skip Mesa download (use existing)
    .\install_dependencies.ps1 -SkipMesa
#>

param(
    [switch]$Cuda,
    [switch]$SkipMesa,
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
Write-Step "Installing glfw3 glm bullet3 via vcpkg"

& "$VcpkgRoot\vcpkg.exe" install glfw3:x64-windows glm:x64-windows bullet3:x64-windows gtest:x64-windows
if ($LASTEXITCODE -ne 0) { Write-Error "vcpkg install failed"; exit 1 }

# ---------------------------------------------------------------------------
# Mesa3D (prebuilt, GPU-capable: llvmpipe + d3d12 + zink) from mesa-dist-win
#   - release-msvc -> runtime DLLs       -> libs\mesa\x64   (MESA_PATH)
#   - devel-msvc   -> headers + .lib     -> libs\mesa-devel (MESA_SDK_DIR)
# Both packages must be the same Mesa version. CMake copies the runtime DLLs
# next to the executable at build time.
# ---------------------------------------------------------------------------
Write-Step "Setting up Mesa3D (prebuilt from pal1000/mesa-dist-win)"

$mesaDir    = "$PSScriptRoot\libs\mesa"
$mesaSdkDir = "$PSScriptRoot\libs\mesa-devel"

if ($SkipMesa) {
    Write-Host "  -SkipMesa specified, skipping Mesa download"
} elseif ((Test-Path "$mesaDir\x64\libgallium_wgl.dll") -and (Test-Path "$mesaSdkDir\lib\x64\libGLESv2.lib")) {
    Write-Host "  Using existing Mesa at $mesaDir (+ SDK at $mesaSdkDir)"
} else {
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

    function Get-MesaAsset($pattern) {
        $a = $release.assets |
            Where-Object { $_.name -like $pattern -and $_.name -notlike "*debug*" } |
            Select-Object -First 1
        if (-not $a) { Write-Error "No '$pattern' asset found in mesa-dist-win latest release"; exit 1 }
        return $a
    }

    function Expand-Mesa7z($asset, $dest) {
        $tmp = "$PSScriptRoot\libs\$($asset.name)"
        Write-Host "  Downloading $($asset.name)..."
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $tmp -UseBasicParsing
        if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        & $sevenZr x $tmp "-o$dest" -y | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Error "7zr extraction failed: $($asset.name)"; exit 1 }
        Remove-Item $tmp
    }

    Expand-Mesa7z (Get-MesaAsset "*release-msvc.7z") $mesaDir
    Expand-Mesa7z (Get-MesaAsset "*devel-msvc.7z")   $mesaSdkDir

    if (-not (Test-Path "$mesaDir\x64\libgallium_wgl.dll")) {
        Write-Error "Mesa runtime extraction failed - x64\libgallium_wgl.dll not found in $mesaDir"; exit 1
    }
    if (-not (Test-Path "$mesaSdkDir\lib\x64\libGLESv2.lib")) {
        Write-Error "Mesa SDK extraction failed - lib\x64\libGLESv2.lib not found in $mesaSdkDir"; exit 1
    }
    Write-Host "  Mesa3D runtime: $mesaDir\x64"
    Write-Host "  Mesa3D SDK:     $mesaSdkDir"
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
Write-Host "  vcpkg:        $VcpkgRoot" -ForegroundColor Green
Write-Host "  Mesa runtime: $mesaDir\x64" -ForegroundColor Green
Write-Host "  Mesa SDK:     $mesaSdkDir" -ForegroundColor Green
Write-Host "  LibTorch:     $LibtorchPath" -ForegroundColor Green
