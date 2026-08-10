# Dot-source this before building or running the Go port on Windows:
#
#   . .\env.ps1
#
# It sets the three things a run needs and nothing else. Every one of them has cost
# real debugging time, so they live here rather than in a README the reader has to
# retype from.
#
#   PATH     MSYS2's bin directory, for the OpenCV DLLs the binary links against, and
#            D:\Go\bin for the toolchain (installed from the zip, not an MSI, so it is
#            not on PATH system-wide).
#   ORT_DLL  ONNX Runtime is loaded BY PATH at runtime, not linked. Without this the
#            CLI exits 1 on every case with "set ORT_DLL/ORT_SO to a matching build",
#            which reads like a broken build and is not one. The library deliberately
#            comes from the same conda environment the reference implementation uses
#            (1.21.1) -- comparing two ports across two ORT versions would confound
#            "Go differs from Python" with "1.21 differs from 1.28".
#
# Override any of them by setting it before dot-sourcing; nothing here clobbers an
# existing value.

if (-not $env:RDOCS_MSYS) { $env:RDOCS_MSYS = "C:\msys64\mingw64" }
if (-not $env:RDOCS_GOROOT_BIN) { $env:RDOCS_GOROOT_BIN = "D:\Go\bin" }
if (-not $env:ORT_DLL) {
    $env:ORT_DLL = "D:\miniconda3\envs\russiandocs\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
}

# Provenance for the sidecar. The spike's own conclusion was that over half of
# plausible-looking numeric failures are really a version or build mismatch wearing a
# disguise, so `info` reports what it was built from -- but it reads an env var rather
# than baking a value in, keeping the binary itself reproducible. Re-dot-source after
# committing, or the reported commit is the one you started the shell on.
if (-not $env:RDOCS_COMMIT) {
    $sha = & git -C "$PSScriptRoot\..\.." rev-parse --short HEAD 2>$null
    if ($LASTEXITCODE -eq 0) { $env:RDOCS_COMMIT = $sha.Trim() }
}

foreach ($dir in @($env:RDOCS_GOROOT_BIN, "$env:RDOCS_MSYS\bin")) {
    if (-not (";$env:PATH;".Contains(";$dir;"))) { $env:PATH = "$dir;$env:PATH" }
}

if (-not (Test-Path $env:ORT_DLL)) {
    Write-Warning "ORT_DLL does not exist: $env:ORT_DLL"
}

Write-Host "go      : $env:RDOCS_GOROOT_BIN"
Write-Host "opencv  : $env:RDOCS_MSYS\bin"
Write-Host "ORT_DLL : $env:ORT_DLL"
Write-Host "commit  : $env:RDOCS_COMMIT"
