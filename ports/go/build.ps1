# Build and test the Go port on Windows.
#
#   .\build.ps1            build binaries
#   .\build.ps1 -Test      build, then run the unit tests
#
# Two Windows-specific facts are handled here, both discovered the hard way.
#
# 1. `-tags customenv` (see also ../../conformance/README.md). gocv's built-in Windows
#    flags expect an OpenCV built by its own win_build_opencv.cmd into C:/opencv with
#    version-suffixed import libraries. MSYS2's prebuilt package installs unsuffixed
#    ones under C:/msys64/mingw64, so the defaults cannot find them. The flags below
#    come from pkgconf rather than being hand-listed, so adding an OpenCV module never
#    means editing this file.
#
# 2. DEVIATIONS D-09 — System32 shadows MSYS2's runtime DLLs. C:\Windows\System32 on
#    some machines contains libstdc++-6.dll, libgcc_s_seh-1.dll, libwinpthread-1.dll
#    and zlib1.dll from unrelated installers, older than MSYS2's. Windows searches the
#    application directory first and System32 second, BOTH BEFORE PATH, so a binary
#    anywhere else loads the stale libstdc++ and dies at load with
#    STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139) before main() runs, printing nothing.
#    Those four files are therefore copied next to the binaries.
#
#    The same trap bites `go test`, which builds its test binary into a TEMP directory
#    where the copies do not exist. Hence -Test compiles the test binaries with
#    `go test -c` into bin\ and runs them from there. Do not "simplify" this back to
#    plain `go test`; it fails with an exit code and no message.

param([switch]$Test, [switch]$Soak, [ValidateSet("cpu","gpu")][string]$SoakDevice = "cpu")

$ErrorActionPreference = "Stop"
$MSYS = "C:/msys64/mingw64"

if (-not (Test-Path "$MSYS/include/opencv4/opencv2/core.hpp")) {
    throw "OpenCV headers not found under $MSYS. Install with:`n" +
          "  C:\msys64\usr\bin\pacman.exe -S --needed mingw-w64-x86_64-opencv " +
          "mingw-w64-x86_64-gcc mingw-w64-x86_64-pkgconf mingw-w64-x86_64-qt6-base"
}

# One toolchain end to end: the compiler comes from the same place as the libraries.
$env:CC = "$MSYS/bin/gcc.exe"
$env:CXX = "$MSYS/bin/g++.exe"
$env:PKG_CONFIG_PATH = "$MSYS/lib/pkgconfig"
$env:PATH = "$MSYS/bin;" + $env:PATH
$env:CGO_ENABLED = "1"
$env:CGO_CXXFLAGS = "--std=c++11 -DNDEBUG"
$env:CGO_CPPFLAGS = (& "$MSYS/bin/pkgconf.exe" --cflags opencv4)
$env:CGO_LDFLAGS = (& "$MSYS/bin/pkgconf.exe" --libs opencv4)

$opencv = & "$MSYS/bin/pkgconf.exe" --modversion opencv4
Write-Output "opencv   : $opencv"
Write-Output "compiler : $((& $env:CXX --version | Select-Object -First 1))"
if ($opencv -notmatch '^4\.1[23]\.') {
    Write-Warning "OpenCV $opencv is neither 4.12 nor 4.13; pixel parity against the Python reference is not comparable."
}

New-Item -ItemType Directory -Force -Path "bin" | Out-Null

Write-Output "`n--- binaries ---"
& go build -tags customenv -o "bin/rdocs-conform.exe" "./cmd/rdocs-conform"
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Output "  bin/rdocs-conform.exe"

& go build -tags customenv -o "bin/rdocs-service.exe" "./cmd/rdocs-service"
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Output "  bin/rdocs-service.exe"

# D-09: give our binaries the same advantage OpenCV's own tools have.
$shadowed = @("libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll", "zlib1.dll")
Write-Output "`n--- System32-shadowed runtime DLLs (D-09) ---"
foreach ($d in $shadowed) {
    Copy-Item (Join-Path "$MSYS/bin" $d) (Join-Path "bin" $d) -Force
    Write-Output "  bin/$d"
}

if ($Test) {
    Write-Output "`n--- tests (compiled into bin\ so D-09 does not bite) ---"
    $failed = 0
    $packages = & go list ./... | Where-Object { $_ }
    foreach ($pkg in $packages) {
        $name = ($pkg -split '/')[-1]
        $exe = "bin\$name.test.exe"
        & go test -tags customenv -c -o $exe $pkg 2>&1 | Out-Null
        if (-not (Test-Path $exe)) { Write-Output "  $name : no test files"; continue }
        Push-Location "bin"
        # No -test.* flags: PowerShell parses `-test.v=$false` as `-test` and the
        # binary rejects it. Non-verbose is the default anyway.
        #
        # ErrorActionPreference is relaxed for this call ONLY. With it set to Stop,
        # PowerShell 5.1 turns any line a native process writes to stderr into a
        # terminating NativeCommandError -- so a test that deliberately exercises a
        # warning path (repo has one for an invalid environment value) fails the build
        # while reporting "ok". The exit code is the verdict; stderr is output.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & ".\$name.test.exe"
        $code = $LASTEXITCODE
        $ErrorActionPreference = $prev
        Pop-Location
        if ($code -eq 0) { Write-Output "  ok   $name" } else { Write-Output "  FAIL $name (exit $code)"; $failed++ }
        Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
    }
    if ($failed -gt 0) { throw "$failed package(s) failed" }
    Write-Output "`nall tests passed"
}

if ($Soak) {
    # The soak test is separate from -Test because it takes minutes, needs the model
    # artifacts, and on GPU monopolises the device. Same D-09 treatment: compiled into
    # bin\ and run from there.
    Write-Output "`n--- soak: one session, eight goroutines, 3000 calls ($SoakDevice) ---"
    $exe = "bin\modules.soak.exe"
    & go test -tags customenv -c -o $exe "./internal/docproc/modules"
    if ($LASTEXITCODE -ne 0) { throw "soak build failed" }
    Push-Location "bin"
    $env:RDOCS_SOAK = "1"
    $env:RDOCS_SOAK_DEVICE = $SoakDevice
    # Flags are QUOTED: PowerShell splits an unquoted -test.v at the dot and the binary
    # rejects the fragment.
    & ".\modules.soak.exe" "-test.run" "Soak" "-test.v" "-test.timeout" "30m"
    $code = $LASTEXITCODE
    Pop-Location
    Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
    if ($code -ne 0) { throw "soak failed (exit $code)" }
}

Write-Output "`nDone. Running the binaries also needs $MSYS/bin on PATH for the OpenCV DLLs themselves."
