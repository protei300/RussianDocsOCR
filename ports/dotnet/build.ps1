# Build and verify the .NET port.
#
# Why a script rather than "just run dotnet build": `dotnet test` builds ONLY the test project and
# its dependencies, so the conformance CLI keeps its own, older copy of
# RussianDocs.DocumentProcessing.dll in its output folder. That already produced one confusing
# result — a source fix was in place, the tests passed, and conformance still reported the OLD
# behaviour. Each .NET project's output has a private copy of every dependency; a single binary,
# as in Go, has no equivalent of this.
#
#   .\build.ps1            build + unit tests
#   .\build.ps1 -Conform   the above, then the conformance run
[CmdletBinding()]
param([switch]$Conform)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root '..' '..')

# Provenance for the sidecar. An env var, not a baked-in constant, so the binary stays
# reproducible and a stale value reads as empty rather than as the wrong commit.
if (-not $env:RDOCS_COMMIT) {
    $env:RDOCS_COMMIT = (& git -C $repo rev-parse --short HEAD)
}
Write-Host "commit : $env:RDOCS_COMMIT"

Write-Host "`n--- build (whole solution, so every output is current) ---"
& dotnet build (Join-Path $root 'RussianDocs.sln') -c Release --nologo
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "`n--- unit tests ---"
& dotnet test (Join-Path $root 'RussianDocs.sln') -c Release --nologo --no-build
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

if ($Conform) {
    Write-Host "`n--- conformance ---"
    Push-Location $repo
    try {
        & D:/miniconda3/envs/russiandocs/python.exe -m conformance.runner run --port dotnet
    } finally { Pop-Location }
}

Write-Host "`nall green"
