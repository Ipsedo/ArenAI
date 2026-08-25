<#
.SYNOPSIS
    Assembles the self-contained Windows release archive:

      arenai_windows-<arch>_<version>.zip
      |-- arenai_desktop.exe      (looks for .\resources next to itself)
      |-- arenai_agent_train.exe
      |-- *.dll                   (vcpkg applocal + libtorch, already staged
      |                            in build\bin by the build)
      \-- resources\

.USAGE
    .\package_windows.ps1 -BuildDir ..\..\build -Version 1.3.0 [-OutDir dist]
#>

param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"

$repoDir = Resolve-Path "$PSScriptRoot\..\.."
$binDir = Resolve-Path "$BuildDir\bin"

$arch = switch ($env:PROCESSOR_ARCHITECTURE) {
    "AMD64" { "x86_64" }
    "ARM64" { "arm64" }
    default { Write-Error "unsupported architecture: $env:PROCESSOR_ARCHITECTURE"; exit 1 }
}

$zipName = "arenai_windows-${arch}_${Version}.zip"

$stage = Join-Path ([System.IO.Path]::GetTempPath()) "arenai_package_$([System.IO.Path]::GetRandomFileName())"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

foreach ($exe in "arenai_desktop.exe", "arenai_agent_train.exe") {
    $path = Join-Path $binDir $exe
    if (-not (Test-Path $path)) { Write-Error "$exe not found in $binDir"; exit 1 }
    Copy-Item $path $stage
}

# build\bin holds the DLLs of everything the executables link (vcpkg applocal
# deployment + the libtorch copy from ArenaiRuntimeDlls.cmake). Test binaries
# and import .lib files stay out of the archive.
Copy-Item (Join-Path $binDir "*.dll") $stage
Copy-Item -Recurse (Join-Path $repoDir "resources") (Join-Path $stage "resources")

$zipPath = Join-Path (Resolve-Path $OutDir) $zipName
if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path "$stage\*" -DestinationPath $zipPath

Remove-Item -Recurse -Force $stage

$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "wrote $zipPath ($sizeMb MB)"
