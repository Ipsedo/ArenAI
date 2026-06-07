<#
.SYNOPSIS
    Build arenai_desktop on Windows.
    Installs vcpkg + dependencies (glfw3, boost, libtorch) then runs CMake.

.USAGE
    # CPU build (default)
    .\build_windows.ps1

    # CUDA build
    .\build_windows.ps1 -Cuda

    # Custom libtorch path (skip download)
    .\build_windows.ps1 -LibtorchPath "C:\libtorch"

    # Release build
    .\build_windows.ps1 -Config Release
#>

param(
    [switch]$Cuda,
    [string]$LibtorchPath = "",
    [string]$Config = "Release",
    [string]$VcpkgRoot = "$PSScriptRoot\vcpkg"
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
Assert-Command "cmake"

# Visual Studio / MSVC check
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) {
    Write-Error "Visual Studio not found. Install VS 2022 with the 'Desktop development with C++' workload."
    exit 1
}
$vsPath = & $vsWhere -latest -property installationPath
Write-Host "  Visual Studio: $vsPath"

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

$vcpkgExe = "$VcpkgRoot\vcpkg.exe"
$vcpkgToolchain = "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"
Write-Host "  vcpkg: $vcpkgExe"

# ---------------------------------------------------------------------------
# Install vcpkg packages (glfw3, boost)
# ---------------------------------------------------------------------------
Write-Step "Installing glfw3 and boost via vcpkg"

& $vcpkgExe install glfw3:x64-windows boost:x64-windows
if ($LASTEXITCODE -ne 0) { Write-Error "vcpkg install failed"; exit 1 }

# ---------------------------------------------------------------------------
# LibTorch
# ---------------------------------------------------------------------------
Write-Step "Setting up LibTorch"

if ($LibtorchPath -eq "") {
    $LibtorchPath = "$PSScriptRoot\libtorch"
}

if (-not (Test-Path "$LibtorchPath\share\cmake\Torch\TorchConfig.cmake")) {
    Write-Host "  Downloading LibTorch..."

    if ($Cuda) {
        $torchUrl = "https://download.pytorch.org/libtorch/cu124/libtorch-win-shared-with-deps-2.6.0%2Bcu124.zip"
    } else {
        $torchUrl = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.6.0%2Bcpu.zip"
    }

    $zipPath = "$PSScriptRoot\libtorch.zip"

    Write-Host "  URL: $torchUrl"
    Invoke-WebRequest -Uri $torchUrl -OutFile $zipPath -UseBasicParsing

    Write-Host "  Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath $PSScriptRoot -Force
    Remove-Item $zipPath

    if (-not (Test-Path "$LibtorchPath\share\cmake\Torch\TorchConfig.cmake")) {
        Write-Error "LibTorch extraction failed - TorchConfig.cmake not found in $LibtorchPath"
        exit 1
    }
} else {
    Write-Host "  Using existing LibTorch at $LibtorchPath"
}

# ---------------------------------------------------------------------------
# CMake configure
# ---------------------------------------------------------------------------
Write-Step "Configuring CMake ($Config)"

$buildDir = "$PSScriptRoot\build-windows"

$cmakeArgs = @(
    "-B", $buildDir
    "-S", $PSScriptRoot
    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
    "-DCMAKE_PREFIX_PATH=$LibtorchPath\share\cmake"
    "-DCMAKE_BUILD_TYPE=$Config"
    "-DLIBTORCH_PATH=$LibtorchPath"
)

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed"; exit 1 }

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
Write-Step "Building arenai_desktop ($Config)"

cmake --build $buildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
$exePath = "$buildDir\$Config\arenai_desktop.exe"
if (-not (Test-Path $exePath)) {
    $exePath = Get-ChildItem -Path $buildDir -Recurse -Filter "arenai_desktop.exe" | Select-Object -First 1 -ExpandProperty FullName
}

Write-Step "Build complete!"
Write-Host "  Executable: $exePath" -ForegroundColor Green
Write-Host ""
Write-Host "  To run:"
Write-Host "    cd $buildDir\$Config"
Write-Host '    .\arenai_desktop.exe'
