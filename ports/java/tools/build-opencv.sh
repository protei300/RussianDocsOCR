#!/usr/bin/env bash
# Builds OpenCV with the Java bindings this port needs.
#
# **THIS CMAKE LINE IS THE SAME ONE THE DOCKER STAGE USES.** Keep them identical: two OpenCV builds
# that both claim 4.13.0 while differing in configuration is the hardest kind of divergence to find,
# because every version string agrees.
#
# Why build at all: there is no official `org.opencv` artefact on Maven Central — checked, not
# assumed, and all 24 `a:opencv` hits there are third-party republishers stopping at 4.9/4.10. 4.13.0
# is mandatory rather than preferred, because contour approximation changed in OpenCV 4.8 and the
# conformance goldens encode 4.13 behaviour. A port on 4.9 fails `borders.segments` for a reason
# nothing in the code explains.
#
# Requirements: a C++ toolchain, cmake, ninja (or make), and a JDK. NOT ant — OpenCV 4.13 assembles
# the jar itself, which was verified rather than assumed: cmake reports `ant: NO` and still produces
# `bin/opencv-4130.jar`.
#
# Usage:
#   tools/build-opencv.sh [/path/to/install/root]
#
# Then point the build and the runtime at the result:
#   export RDOCS_OPENCV_HOME=<build dir>
set -euo pipefail

VERSION=4.13.0
FLAT=${VERSION//./}
WORK=${1:-"${TMPDIR:-/tmp}/rdocs-opencv"}
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

mkdir -p "$WORK"
cd "$WORK"

if [ ! -d "opencv-$VERSION" ]; then
    curl -fsSL -o opencv.zip \
        "https://github.com/opencv/opencv/archive/refs/tags/$VERSION.zip"
    unzip -q opencv.zip
    rm opencv.zip
fi

mkdir -p build
cd build

# The module list is an ALLOWLIST rather than a set of exclusions: the pipeline uses roughly thirty
# functions across core, imgproc, imgcodecs and calib3d, and building the rest costs time and size for
# nothing.
#
# `dnn` is deliberately ABSENT, and that was worth checking rather than copying: neither the Go nor the
# .NET port calls OpenCV's NMSBoxes — both implement NMS by hand, because the reference's suppression
# has a specific stable-argsort order and a specific tie-break (on equal confidence, keep the LARGEST
# original index) that OpenCV reproduces in neither. `dnn` is one of the largest modules.
#
# `java` is in the list; `highgui` is there because the binding links it regardless, and is built
# HEADLESS (D-07) so it pulls in neither GTK nor Qt — verified: the configure summary reports
# `GUI: NONE`.
#
# WITH_LAPACK=OFF is not an optimisation. On a machine with MSYS2's OpenBLAS installed, LAPACK
# detection finds it and then demands a Fortran compiler: `No CMAKE_Fortran_COMPILER could be found`.
# Nothing in this module set needs LAPACK.
#
# Codecs come from OpenCV's own 3rdparty tree, for the reason the Go image documents: Debian bookworm
# and Ubuntu jammy disagree on codec SONAMEs (libjpeg.so.62 vs .8, libtiff.so.6 vs .5), so a library
# built against one cannot load on the other. Bundling removes the whole class of problem, and it is
# what the opencv-python wheels do.
cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D BUILD_JAVA=ON \
    -D BUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,highgui,java \
    -D WITH_LAPACK=OFF -D WITH_OPENBLAS=OFF \
    -D WITH_QT=OFF -D WITH_GTK=OFF -D WITH_WIN32UI=OFF \
    -D WITH_FFMPEG=OFF -D WITH_GSTREAMER=OFF -D WITH_V4L=OFF \
    -D WITH_OPENEXR=OFF -D WITH_IPP=OFF -D WITH_TBB=OFF -D WITH_OPENCL=OFF \
    -D BUILD_JPEG=ON -D BUILD_PNG=ON -D BUILD_TIFF=ON -D BUILD_WEBP=ON \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_SHARED_LIBS=ON \
    "../opencv-$VERSION"

ninja "-j$JOBS"

# Asserted rather than assumed: a configure that quietly disabled the Java wrappers still builds, and
# the failure then arrives as a missing jar at Gradle time with nothing pointing back to here.
test -f "bin/opencv-$FLAT.jar" || {
    echo "FAIL: bin/opencv-$FLAT.jar was not produced — check the 'Java' section of the cmake summary"
    exit 1
}
ls lib/libopencv_java"$FLAT"* >/dev/null 2>&1 || ls lib/opencv_java"$FLAT"* >/dev/null 2>&1 || {
    echo "FAIL: the JNI library was not produced"
    exit 1
}

echo
echo "OpenCV $VERSION built. Point the build at it:"
echo "    export RDOCS_OPENCV_HOME=$PWD"
