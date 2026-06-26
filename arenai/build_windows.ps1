<#
.SYNOPSIS
    Build arenai_desktop on Windows.
    Assumes dependencies are already installed (run install_dependencies.ps1 first).

.USAGE
    # Release build (default)
    .\build_windows.ps1

    # Debug build
    .\build_windows.ps1 -Config Debug

    # Custom libtorch path
    .\build_windows.ps1 -LibtorchPath "C:\libtorch"
#>

param(
    [string]$LibtorchPath = "$PSScriptRoot\libs\libtorch",
    [string]$Config = "Release",
    [string]$VcpkgRoot = "$PSScriptRoot\libs\vcpkg"
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Assert-Command($cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "$cmd is not installed or not in PATH."; exit 1
    }
}

# Use CMake and Ninja bundled with Visual Studio
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) { Write-Error "Visual Studio not found"; exit 1 }
$vsPath = & $vsWhere -latest -prerelease -property installationPath
$cmake  = "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja  = "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path $cmake)) { Write-Error "cmake.exe not found at $cmake"; exit 1 }
if (-not (Test-Path $ninja)) { Write-Error "ninja.exe not found at $ninja"; exit 1 }

$vcpkgToolchain = "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"

# ---------------------------------------------------------------------------
# Initialize Visual Studio environment (equivalent to vcvarsall.bat x64)
# Without this, CMake+Ninja cannot find the MSVC compiler.
# ---------------------------------------------------------------------------
Write-Step "Initializing Visual Studio environment"
$vcvarsall = "$vsPath\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvarsall)) { Write-Error "vcvarsall.bat not found at $vcvarsall"; exit 1 }

$envOutput = cmd /c "`"$vcvarsall`" x64 && set"
foreach ($line in $envOutput) {
    if ($line -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

# ---------------------------------------------------------------------------
# CMake configure
# ---------------------------------------------------------------------------
Write-Step "Configuring CMake ($Config)"

$buildDir = "$PSScriptRoot\build"

$cacheFile = "$buildDir\CMakeCache.txt"
if (Test-Path $cacheFile) {
    $cachedGenerator = (Select-String -Path $cacheFile -Pattern "CMAKE_GENERATOR:INTERNAL=(.+)").Matches.Groups[1].Value
    if ($cachedGenerator -and $cachedGenerator -ne "Ninja") {
        Write-Host "  Generator changed ($cachedGenerator -> Ninja), clearing CMake cache..."
        Remove-Item -Recurse -Force $buildDir
    }
}

$cmakeArgs = @(
    "-B", $buildDir
    "-S", $PSScriptRoot
    "-G", "Ninja"
    "-DCMAKE_MAKE_PROGRAM=$ninja"
    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
    "-DCMAKE_PREFIX_PATH=$LibtorchPath\share\cmake"
    "-DCMAKE_BUILD_TYPE=$Config"
    "-DLIBTORCH_PATH=$LibtorchPath"
    "-DVCPKG_INSTALLED_DIR=$VcpkgRoot\installed"
    "-DCMAKE_CXX_FLAGS=/EHsc /D_USE_MATH_DEFINES /DNOMINMAX /O2"
)

& $cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed"; exit 1 }

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
Write-Step "Building arenai_desktop ($Config)"

& $cmake --build $buildDir --config $Config --parallel 4
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
$exePath = "$buildDir\arenai_desktop.exe"
if (-not (Test-Path $exePath)) {
    $exePath = Get-ChildItem -Path $buildDir -Recurse -Filter "arenai_desktop.exe" | Select-Object -First 1 -ExpandProperty FullName
}

Write-Step "Build complete!"
Write-Host "  Executable: $exePath" -ForegroundColor Green
