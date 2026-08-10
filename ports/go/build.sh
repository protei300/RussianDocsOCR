#!/usr/bin/env bash
# Build and test the Go port on Linux (and in the Docker image).
#
#   ./build.sh          build binaries
#   ./build.sh --test   build, then run the unit tests
#
# Linux is markedly simpler than Windows here, and both reasons are worth knowing:
#
#  * gocv's default cgo directives already say `#cgo !windows pkg-config: opencv4`, so
#    there is no `customenv` tag and no hand-written CGO_* flags — just an OpenCV whose
#    .pc file pkg-config can find.
#  * DEVIATIONS D-09 (System32 shadowing MSYS2's runtime DLLs) does not exist here, so
#    plain `go test` works and no DLLs are copied anywhere.
#
# VERSION MATTERS, and not for pedantry. gocv 0.43.0 targets OpenCV 4.13.0 and the
# Python reference runs opencv-python-headless 4.12.0.88; the spike measured those two
# as bit-identical on every operation the pipeline performs. Debian bookworm's
# libopencv-dev is 4.6, far enough back that contour approximation and the resize
# fixed-point tables differ — which are exactly the things the conformance suite
# compares. Do not silently accept whatever the distro offers.
#
# Build OpenCV HEADLESS (-DWITH_QT=OFF -DWITH_GTK=OFF). gocv links highgui
# unconditionally, so a service that never draws a window otherwise drags Qt or GTK
# into the image for nothing. See DEVIATIONS D-07.
set -euo pipefail
cd "$(dirname "$0")"

if ! pkg-config --exists opencv4; then
  cat >&2 <<'EOF'
opencv4 not found by pkg-config. Options, best first:

  Docker -- prebuilt and version-pinned:
      FROM ghcr.io/hybridgroup/opencv:4.13.0
  Debian/Ubuntu (CHECK THE VERSION -- bookworm ships 4.6, which is too old):
      apt-get install -y libopencv-dev pkg-config
  A custom prefix:
      export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
EOF
  exit 1
fi

OPENCV_VERSION="$(pkg-config --modversion opencv4)"
echo "opencv   : ${OPENCV_VERSION}"
echo "compiler : $(${CXX:-g++} --version | head -1)"
case "${OPENCV_VERSION}" in
  4.12.*|4.13.*) ;;
  *) echo "WARNING: OpenCV ${OPENCV_VERSION} is neither 4.12 nor 4.13 -- pixel parity" \
          "against the Python reference is not comparable." >&2 ;;
esac

export CGO_ENABLED=1
mkdir -p bin

echo
echo "--- binaries ---"
go build -o bin/rdocs-conform ./cmd/rdocs-conform
echo "  bin/rdocs-conform"

if [ "${1:-}" = "--test" ]; then
  echo
  echo "--- tests ---"
  go test ./...
fi

echo
echo "Done. ONNX Runtime is loaded at RUNTIME by path; set ORT_DLL (for example"
echo "/usr/local/lib/libonnxruntime.so.1.21.1). It must be 1.21.x, because"
echo "onnxruntime_go v1.19.0 vendors ORT_API_VERSION 21 -- see ports/go/README.md."
