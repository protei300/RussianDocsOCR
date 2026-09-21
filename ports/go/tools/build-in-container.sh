#!/bin/sh
# Build the Go port INSIDE the builder image, for a host without Go, MSYS2 or OpenCV.
#
# Usage (from the repository root on the host):
#
#   docker run -d --name rdocs-go-dev -v "$PWD":/host -w /host/ports/go \
#       -e ORT_SO=/usr/local/lib/libonnxruntime.so -e RDOCS_MODELS_ROOT=/host \
#       rdocs-go-soak:check sleep infinity
#   docker exec rdocs-go-dev sh ports/go/tools/build-in-container.sh          # binaries
#   docker exec rdocs-go-dev sh ports/go/tools/build-in-container.sh --test   # + unit tests
#   docker exec -w /host rdocs-go-dev python3 -m conformance.runner run --port go
#
# The image is the `go-builder` stage of build/Dockerfile plus libonnxruntime.so (the
# soak recipe in README.md). The cgo flags are derived from the installed .so files
# exactly as the Dockerfile does: OpenCV 4.13 ships no opencv4.pc unless asked, so
# pkg-config is silently empty here (measured, see build/Dockerfile stage 4).
#
# Output lands in ports/go/bin/ on the HOST mount, which is where ports.json points.
set -eu
cd "$(dirname "$0")/.."

libs=""
for so in /usr/local/lib/libopencv_*.so; do
    base="$(basename "$so" .so)"
    libs="$libs -l${base#lib}"
done
[ -n "$libs" ] || { echo "FAIL: no libopencv_*.so in /usr/local/lib"; exit 1; }
export CGO_ENABLED=1
export CGO_CXXFLAGS="--std=c++11 -DNDEBUG"
export CGO_CPPFLAGS="-I/usr/local/include/opencv4"
export CGO_LDFLAGS="-L/usr/local/lib$libs"

mkdir -p bin
echo "--- binaries ---"
go build -tags customenv -o bin/rdocs-conform ./cmd/rdocs-conform
echo "  bin/rdocs-conform"
go build -tags customenv -o bin/rdocs-service ./cmd/rdocs-service
echo "  bin/rdocs-service"

if [ "${1:-}" = "--test" ]; then
    echo "--- tests ---"
    go test -tags customenv ./...
fi
