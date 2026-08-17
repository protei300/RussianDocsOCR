#!/usr/bin/env bash
# Builds the OpenCV this port needs, on a machine whose package manager does not offer it.
#
# **THIS CMAKE LINE IS STAGE 2 OF `build/Dockerfile`.** Keep them identical: two OpenCV builds that
# both claim 4.13.0 while differing in configuration is the hardest kind of divergence to find,
# because every version string agrees. The one deliberate addition is OPENCV_GENERATE_PKGCONFIG —
# see below.
#
# Why build at all, when `apt-get install libopencv-dev` exists: it does not get far enough to fail
# at conformance time, it fails at COMPILE time. Ubuntu 24.04 ships 4.6, and gocv 0.43.0's
# `aruco.cpp` needs `cv::aruco`, which moved out of contrib and into `objdetect` in OpenCV 4.7. The
# error is fifty lines of `'aruco' in namespace 'cv' does not name a type` and names no version.
# 4.13.0 specifically, rather than merely ">= 4.7", for the older reason: contour approximation
# changed in 4.8 and the conformance goldens encode 4.13 behaviour, so a port built against 4.9
# fails `borders.segments` for a reason nothing in the code explains.
#
# OPENCV_GENERATE_PKGCONFIG=ON is the difference from the Docker stage, and it is what lets
# `build.sh` work unchanged: gocv's default cgo directives say `#cgo !windows pkg-config: opencv4`,
# so on Linux the flags come from a .pc file and there is no `customenv` tag to pass. The image
# builds without it and derives the flags from the installed .so files instead — a workaround for a
# build that had already been configured, not a better way.
#
# Requirements: a C++ toolchain, cmake, ninja (or make), curl, unzip.
#
# Usage:
#   tools/build-opencv.sh [/path/to/work/root]
#
# Then point the build at the result:
#   export PKG_CONFIG_PATH=<work>/install/lib/pkgconfig
#   export LD_LIBRARY_PATH=<work>/install/lib
set -euo pipefail

VERSION=4.13.0
WORK=${1:-"${TMPDIR:-/tmp}/rdocs-opencv-go"}
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

mkdir -p "$WORK"
cd "$WORK"
PREFIX="$PWD/install"

if [ ! -d "opencv-$VERSION" ]; then
    curl -fsSL -o opencv.zip \
        "https://github.com/opencv/opencv/archive/refs/tags/$VERSION.zip"
    unzip -q opencv.zip
    rm opencv.zip
fi

mkdir -p build
cd build

# The module list is an ALLOWLIST rather than a set of exclusions: the pipeline uses roughly thirty
# functions across core, imgproc, imgcodecs and calib3d, and building the rest costs time for
# nothing. It is longer than the Java port's list because gocv is ONE cgo package — every .cpp in
# it compiles whether the port calls into it or not, so anything its headers reach must exist.
# `highgui` is built HEADLESS (D-07): gocv links it unconditionally, and WITH_QT=OFF/WITH_GTK=OFF
# make it a stub that drags in neither.
#
# Codecs come from OpenCV's own 3rdparty tree for the reason the Docker stage documents: Debian
# bookworm and Ubuntu jammy disagree on codec SONAMEs, so a library built against one cannot load
# on the other. Irrelevant on a CI runner that builds and runs in one place, kept because this is
# the same configuration as the image and divergence is the thing being avoided.
cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX="$PREFIX" \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,highgui,video,videoio,objdetect,photo,dnn \
    -D WITH_LAPACK=OFF -D WITH_OPENBLAS=OFF \
    -D WITH_QT=OFF -D WITH_GTK=OFF -D WITH_WIN32UI=OFF \
    -D WITH_FFMPEG=OFF -D WITH_GSTREAMER=OFF -D WITH_V4L=OFF \
    -D WITH_OPENEXR=OFF -D WITH_IPP=OFF -D WITH_TBB=OFF -D WITH_OPENCL=OFF \
    -D BUILD_JPEG=ON -D BUILD_PNG=ON -D BUILD_TIFF=ON -D BUILD_WEBP=ON \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF \
    -D BUILD_opencv_apps=OFF -D BUILD_opencv_python3=OFF \
    -D BUILD_SHARED_LIBS=ON \
    "../opencv-$VERSION"

ninja "-j$JOBS"
ninja install

# Asserted rather than assumed: a configure that quietly skipped the .pc file still builds and
# installs, and the failure then arrives from build.sh as "opencv4 not found by pkg-config", which
# reads as a missing PKG_CONFIG_PATH rather than a missing file.
PC="$PREFIX/lib/pkgconfig/opencv4.pc"
test -f "$PC" || {
    echo "FAIL: $PC was not produced — check OPENCV_GENERATE_PKGCONFIG in the cmake summary"
    exit 1
}

echo
echo "OpenCV $VERSION installed to $PREFIX. Point the build at it:"
echo "    export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig"
echo "    export LD_LIBRARY_PATH=$PREFIX/lib"
