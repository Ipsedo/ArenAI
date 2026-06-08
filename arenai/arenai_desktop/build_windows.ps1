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
$vsPath = & $vsWhere -latest -prerelease -property installationPath
$vsVersion = & $vsWhere -latest -prerelease -property installationVersion
Write-Host "  Visual Studio: $vsPath ($vsVersion)"

# Initialize MSVC environment from vcvarsall.bat
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

$vcpkgExe = "$VcpkgRoot\vcpkg.exe"
$vcpkgToolchain = "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"
Write-Host "  vcpkg: $vcpkgExe"

# ---------------------------------------------------------------------------
# Install vcpkg packages (glfw3, boost)
# ---------------------------------------------------------------------------
Write-Step "Installing glfw3 via vcpkg"

& $vcpkgExe install glfw3:x64-windows glm:x64-windows bullet3:x64-windows angle:x64-windows
if ($LASTEXITCODE -ne 0) { Write-Error "vcpkg install failed"; exit 1 }

# ---------------------------------------------------------------------------
# Khronos GLES3 / EGL headers (headers only, no compilation)
# ---------------------------------------------------------------------------
Write-Step "Setting up GLES3/EGL headers"

$khronosInclude = "$PSScriptRoot\khronos-headers"
if (-not (Test-Path "$khronosInclude\GLES3\gl3.h")) {
    $null = New-Item -ItemType Directory -Force "$khronosInclude\GLES3"
    $null = New-Item -ItemType Directory -Force "$khronosInclude\EGL"
    $base = "https://raw.githubusercontent.com/KhronosGroup/OpenGL-Registry/main/api/GLES3"
    Invoke-WebRequest "$base/gl3.h"   -OutFile "$khronosInclude\GLES3\gl3.h"   -UseBasicParsing
    Invoke-WebRequest "$base/gl31.h"  -OutFile "$khronosInclude\GLES3\gl31.h"  -UseBasicParsing
    Invoke-WebRequest "$base/gl32.h"  -OutFile "$khronosInclude\GLES3\gl32.h"  -UseBasicParsing
    $eglBase = "https://raw.githubusercontent.com/KhronosGroup/EGL-Registry/main/api/EGL"
    Invoke-WebRequest "$eglBase/egl.h"      -OutFile "$khronosInclude\EGL\egl.h"      -UseBasicParsing
    Invoke-WebRequest "$eglBase/eglext.h"   -OutFile "$khronosInclude\EGL\eglext.h"   -UseBasicParsing
    Invoke-WebRequest "$eglBase/eglplatform.h" -OutFile "$khronosInclude\EGL\eglplatform.h" -UseBasicParsing
    Write-Host "  Headers downloaded to $khronosInclude"
} else {
    Write-Host "  Using existing GLES3/EGL headers at $khronosInclude"
}

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

$cacheFile = "$buildDir\CMakeCache.txt"
if (Test-Path $cacheFile) {
    $cachedGenerator = (Select-String -Path $cacheFile -Pattern "CMAKE_GENERATOR:INTERNAL=(.+)").Matches.Groups[1].Value
    if ($cachedGenerator -and $cachedGenerator -ne $vsGenerator) {
        Write-Host "  Generator changed ($cachedGenerator -> $vsGenerator), clearing CMake cache..."
        Remove-Item -Recurse -Force $buildDir
    }
}

$cmakeArgs = @(
    "-B", $buildDir
    "-S", $PSScriptRoot
    "-G", "Ninja"
"-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
    "-DCMAKE_PREFIX_PATH=$LibtorchPath\share\cmake"
    "-DCMAKE_BUILD_TYPE=$Config"
    "-DLIBTORCH_PATH=$LibtorchPath"
    "-DVCPKG_INSTALLED_DIR=$VcpkgRoot\installed"
    "-DGLES3_INCLUDE_DIR=$khronosInclude"
    "-DCMAKE_CXX_FLAGS=/EHsc -D_USE_MATH_DEFINES -DNOMINMAX"
)

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed"; exit 1 }

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
Write-Step "Building arenai_desktop ($Config)"

cmake --build $buildDir --config $Config --parallel 4
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
