# syntax=docker/dockerfile:1.7
# =============================================================================
# RussianDocs OCR — the Kotlin/JVM service
#
# TWO TARGETS, ONE JAR. `onnxruntime_gpu` carries the CPU execution provider as
# well, so the SAME `rdocs-service.jar` runs on either host and the split exists
# only so a CPU-only host does not pull ~4 GB of CUDA kernels it will never
# execute.
#
#   runtime-cpu   eclipse-temurin:21-jre-jammy
#   runtime-gpu   nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 + the same JRE
#
# **BOTH TARGETS BUILT AND RUN, 2026-08-07.** Every number below is MEASURED.
#
#   russiandocs-java:cpu   2.01 GB   (the estimate here had said ~1.0 GB — wrong by 2x, see below)
#   russiandocs-java:gpu   7.21 GB   (the estimate had said ~7.3 GB — close, because CUDA dominates
#                                     and its size was known rather than guessed)
#
#   for comparison, all measured:
#   russiandocs-go:cpu       902 MB      russiandocs-go:gpu      6.73 GB
#   russiandocs-dotnet:cpu  2.31 GB      russiandocs-dotnet:gpu  7.56 GB
#   russiandocs:latest (Python)          7.72 GB
#
# **WHY THE CPU IMAGE IS A GIGABYTE BIGGER THAN IT SHOULD BE**, measured inside the running container:
# `rdocs-service.jar` is **520 MB**, because `bootJar` bundles `onnxruntime_gpu` — and that artefact
# carries ~450 MB of CUDA kernels which a CPU-only host will never execute. The models are 215 MB,
# OpenCV 23 MB, the SPA 624 KB; everything else is the JRE base. So the fix is known and not applied:
# a `cpu` build variant that swaps the dependency for plain `onnxruntime` would roughly halve this
# image. It is not done because the same jar currently serves BOTH targets, which is the property that
# keeps the two runtime stages honest — one artefact, two bases. Trading that away deserves a decision,
# not a silent commit.
#
# Measured in the containers: twelve ONNX sessions in **513 ms** on CPU and **3235 ms** on GPU (CUDA
# context creation is the difference), an INTPASSPORT_2011 in **553 ms** on CPU, a DL_2011 in **541 ms**
# with `device=gpu` — and `ocr_device=cpu` there, as designed. Privileges dropped in both, and
# `GIT_COMMIT` reaches the status page.
#
# The one structural difference from the .NET port: OpenCV is BUILT FROM SOURCE
# here, because there is no `org.opencv` artefact anywhere and 4.13.0 is mandatory
# (contour approximation changed in 4.8; the conformance goldens encode 4.13). That
# stage is 10-25 minutes of the build and is cached separately from everything
# else, which is why it comes first.
#
# Build (from the repository root):
#   docker build -f ports/base/Dockerfile.models -t russiandocs-models:latest .
#   docker build -f ports/java/docker/Dockerfile --target runtime-cpu -t russiandocs-java:cpu \
#       --build-arg GIT_COMMIT="$(git rev-parse --short HEAD)" .
#   docker build -f ports/java/docker/Dockerfile --target runtime-gpu -t russiandocs-java:gpu \
#       --build-arg GIT_COMMIT="$(git rev-parse --short HEAD)" .
#
# Run:
#   docker run --rm -p 8005:8005 -e JWT_SECRET=... -e DEFAULT_API_KEY=... russiandocs-java:cpu
#   docker run --rm --gpus all -p 8005:8005 -e JWT_SECRET=... russiandocs-java:gpu
#
# No volume for /app/data on purpose — see DATA_WIPE_ON_START.
#
# The DEVICE-SELECTION LOGIC is compiled identically into both images, and that
# matters more than the split: the `[gpu, cpu]` attempt loop, the "a listed
# provider is not a working GPU" rule, and the device-node probe are the same
# bytes. A CPU image asked for a GPU logs why it cannot have one rather than
# pretending — and in a container started WITHOUT `--gpus`, that probe is the only
# thing standing between the service and a segfault, because the CUDA provider
# does not fail cleanly there.
#
# Neither Windows native-loading trap (J-01, J-16) can occur in this image: there
# is no System32 to shadow a soname and no JDK-bundled C runtime to be too old.
# =============================================================================

# =============================================================================
# Stage 1: OpenCV 4.13.0 with the Java bindings
#
# **THE CMAKE LINE IS IDENTICAL TO tools/build-opencv.sh's, deliberately.** Two
# OpenCV builds that both claim 4.13.0 while differing in configuration is the
# hardest divergence to find, because every version string agrees. Change both or
# neither.
#
# HEADLESS (`WITH_QT=OFF WITH_GTK=OFF`): the Java binding links highgui
# unconditionally, so without this the image would carry a GUI stack for a service
# that never opens a window. This is D-07 in the Go port, and the .NET port pays
# the same cost in image size instead, because its prebuilt native is not headless.
#
# BUNDLED CODECS (`BUILD_JPEG/PNG/TIFF/WEBP=ON`): Debian bookworm and Ubuntu jammy
# disagree on codec SONAMEs (libjpeg.so.62 vs .8, libtiff.so.6 vs .5), so a library
# built against one cannot load on the other. Bundling removes the whole class of
# problem and is what the opencv-python wheels do.
#
# ~10-25 minutes, and it is why this stage is FIRST: nothing above it changes, so
# the layer is reused across every subsequent build.
# =============================================================================
FROM eclipse-temurin:21-jdk-jammy AS opencv-builder

ARG OPENCV_VERSION=4.13.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build curl unzip ca-certificates \
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opencv
RUN curl -fsSL -o opencv.zip \
        "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip" \
    && unzip -q opencv.zip \
    && rm opencv.zip

# FOUND ON THE FIRST REAL BUILD (2026-08-06): the `python3` above is REQUIRED, and it was missing.
# OpenCV generates its Java wrappers with a Python script, so without an interpreter cmake reports
# `Java: NO`, skips the bindings, and still produces a complete-looking build tree. The assertion after
# the build is what turned that into a failure here rather than a missing class four stages later — it
# earned its keep on the very first run. The Windows build had worked because MSYS2 ships Python.
#
# FIRST-BUILD SUSPECT #1. The module list is an ALLOWLIST: the pipeline uses about
# thirty functions across core, imgproc, imgcodecs and calib3d, and building the
# rest costs time and size for nothing. If a stage fails with an
# `UnsatisfiedLinkError` naming a method that exists in the desktop build, a module
# is missing here rather than in the Kotlin code. `dnn` is deliberately absent —
# every port implements NMS by hand, because the reference's suppression has a
# stable-argsort order and a tie-break OpenCV reproduces in neither.
RUN mkdir -p build && cd build && cmake \
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
        "/opencv/opencv-${OPENCV_VERSION}" \
    && ninja "-j$(nproc)"

# The jar and the JNI library are the two outputs that matter, so their presence is
# ASSERTED. A configure that silently skipped the Java bindings (no JDK found, for
# instance) still produces a complete-looking build tree, and the failure would
# then surface as a missing class four stages later.
RUN set -eu; \
    flat=$(echo "${OPENCV_VERSION}" | tr -d '.'); \
    test -f "/opencv/build/bin/opencv-${flat}.jar" \
        || { echo "FAIL: the Java bindings were not built"; exit 1; }; \
    test -f "/opencv/build/lib/libopencv_java${flat}.so" \
        || { echo "FAIL: the JNI library is missing"; exit 1; }; \
    echo "OpenCV ${OPENCV_VERSION} Java bindings built:"; \
    ls -la "/opencv/build/bin/opencv-${flat}.jar" "/opencv/build/lib/libopencv_java${flat}.so"

# Collected into ONE directory with the layout RDOCS_OPENCV_HOME expects, so the
# runtime stages copy a single path instead of reproducing the build tree.
RUN set -eu; \
    mkdir -p /opencv-dist/bin /opencv-dist/lib; \
    cp /opencv/build/bin/opencv-*.jar /opencv-dist/bin/; \
    cp -P /opencv/build/lib/lib*.so* /opencv-dist/lib/

# =============================================================================
# Stage 2: build the SPA
#
# COPIED BYTE-FOR-BYTE from the root Dockerfile's stage 1. Copied rather than
# rewritten on purpose: all four services serve the SAME frontend, and a second
# version of this stage would drift. If the root stage changes, change this one to
# match rather than improving it here.
# =============================================================================
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build
# Output: /frontend/dist/

# =============================================================================
# Stage 3: build the service jar
#
# Gradle with the OpenCV jar from stage 1 on the classpath. `--no-daemon` because
# a daemon in a build stage is a process nobody stops.
# =============================================================================
FROM eclipse-temurin:21-jdk-jammy AS builder
WORKDIR /src

# The build scripts and the wrapper first, so dependency resolution caches
# independently of the sources. The version catalogue is part of that input:
# libs.versions.toml IS the version set, so a change to it must invalidate the
# resolution layer.
COPY ports/java/gradlew ports/java/settings.gradle.kts ports/java/build.gradle.kts ./ports/java/
COPY ports/java/gradle/ ./ports/java/gradle/
COPY ports/java/docproc/build.gradle.kts ./ports/java/docproc/
COPY ports/java/conform/build.gradle.kts ./ports/java/conform/
COPY ports/java/service/build.gradle.kts ./ports/java/service/
COPY --from=opencv-builder /opencv-dist/ /opt/opencv/

WORKDIR /src/ports/java
ENV RDOCS_OPENCV_HOME=/opt/opencv
RUN chmod +x gradlew && ./gradlew --no-daemon dependencies --configuration runtimeClasspath \
        > /dev/null 2>&1 || true

COPY ports/java/docproc/src/ ./docproc/src/
COPY ports/java/conform/src/ ./conform/src/
COPY ports/java/service/src/ ./service/src/

# `-x test` because the tests need the native libraries on a library path the build
# stage does not set up, and because a container build is not where a test failure
# should first be discovered. CI runs `./gradlew build` on a real host.
RUN ./gradlew --no-daemon :service:bootJar -x test

# FIRST-BUILD SUSPECT #2. The name comes from `bootJar { archiveFileName }`; if
# that changes, this fails here with a clear message rather than three stages later
# with a missing ENTRYPOINT.
RUN set -eu; \
    test -f /src/ports/java/service/build/dist/rdocs-service.jar \
        || { echo "FAIL: rdocs-service.jar was not produced"; exit 1; }; \
    ls -la /src/ports/java/service/build/dist/rdocs-service.jar

# =============================================================================
# Stage 4a: the CPU runtime
# =============================================================================
FROM eclipse-temurin:21-jre-jammy AS runtime-cpu

# libgomp1 is OpenMP, which the OpenCV build links against directly. That is the
# WHOLE list — the headless build in stage 1 is what keeps GTK, Pango, Cairo, ATK
# and X11 off it, sixteen sonames the .NET port has to install because its
# prebuilt native is not headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/ports/java/service/build/dist/rdocs-service.jar ./rdocs-service.jar
COPY --from=opencv-builder /opencv-dist/ /opt/opencv/
COPY --from=frontend-builder /frontend/dist/ ./web/dist/
COPY --from=russiandocs-models /models/ ./document_processing/models/
COPY --from=russiandocs-models /config/ ./document_processing/config/
COPY samples/ ./samples/
COPY service/seed_data/ ./service/seed_data/

# **KEEP THIS.** It is the assertion that made the Go port's three distro problems
# findable rather than mysterious: an unresolved soname fails the BUILD, and without
# it the failure is every document at runtime with an `UnsatisfiedLinkError` that
# names nothing. The bundled codecs and the headless build are exactly what it
# verifies.
RUN set -eu; \
    jni=$(ls /opt/opencv/lib/libopencv_java*.so); \
    missing=$(LD_LIBRARY_PATH=/opt/opencv/lib ldd "$jni" 2>&1 | grep -c 'not found' || true); \
    [ "$missing" = "0" ] || { \
        echo "FAIL: $missing unresolved shared libraries in $jni:"; \
        LD_LIBRARY_PATH=/opt/opencv/lib ldd "$jni" | grep 'not found'; \
        exit 1; }; \
    echo "$(basename "$jni"): all shared libraries resolve"

# RDOCS_REPO_ROOT is the REPOSITORY ROOT, not the models directory. The Go port
# lost a build to that distinction: the loader appends `document_processing/models`
# itself, so pointing this at the models directory produces a path with the segment
# twice and a "model not found" that names a plausible-looking path.
#
# RDOCS_TOOLCHAIN_BIN is deliberately UNSET: it exists only to work around
# System32 shadowing a MinGW soname on Windows, and there is nothing to work around
# here.
ENV RDOCS_REPO_ROOT=/app \
    RDOCS_OPENCV_HOME=/opt/opencv \
    LD_LIBRARY_PATH=/opt/opencv/lib \
    DATA_DIR=/app/data \
    DATA_WIPE_ON_START=true \
    COMPUTE_DEVICE=auto \
    JAVA_TOOL_OPTIONS="-XX:MaxRAMPercentage=75 -XX:+UseContainerSupport"

# Declared AFTER everything cacheable, deliberately: an ARG invalidates every layer below it, and higher
# up a new commit hash would rebuild OpenCV from source. Same placement as the root Dockerfile's. The
# service reads GIT_COMMIT and shows it on the status page; without it the page reads `unknown`, which is
# indistinguishable from a deployment nobody can identify — seen on the first container run.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT

# FIRST-BUILD SUSPECT #3. `useradd --create-home` and `--shell /usr/sbin/nologin`
# are what the .NET images use on jammy; if the Temurin base lacks either path the
# fix is the argument, not the intent.
#
# The data directory is created and chowned HERE rather than at startup, because a
# non-root process cannot create it under /app itself — and the wipe on start
# empties the CONTENTS precisely so it never needs to unlink the directory, which
# would require write permission on /app. The Go image found that the hard way:
# `unlinkat /app/data: permission denied`, with the store then unusable.
RUN useradd --create-home --shell /usr/sbin/nologin rdocs \
    && mkdir -p /app/data \
    && chown -R rdocs:rdocs /app/data
USER rdocs

EXPOSE 8005

# The jar is its own probe, so the image needs neither curl nor a shell for this. A
# HEALTHCHECK that depends on tools installed purely to run it is a larger image and
# one more thing to keep patched.
#
# start-period is generous because model loading takes seconds and /health answers
# OK throughout — the service really is healthy while it loads, since it accepts
# uploads and queues them.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["java", "-jar", "/app/rdocs-service.jar", "--healthcheck", "--addr", ":8005"]

ENTRYPOINT ["java", "-jar", "/app/rdocs-service.jar"]
CMD ["--addr", ":8005"]

# =============================================================================
# Stage 4b: the GPU runtime
#
# **cudnn-runtime, not base.** onnxruntime 1.21 needs cuDNN 9, and with the `base`
# image the CUDA provider fails to load while `CUDAExecutionProvider` remains in the
# advertised list — a service that is silently on CPU and looks healthy. The Python
# and .NET images are pinned to this same tag for the same reason.
#
# The JRE is COPIED from the Temurin image rather than installed with apt, which is
# the supported way to get a specific JDK build onto an arbitrary base and avoids
# a second package repository in the image.
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runtime-gpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=eclipse-temurin:21-jre-jammy /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:$PATH

WORKDIR /app
COPY --from=builder /src/ports/java/service/build/dist/rdocs-service.jar ./rdocs-service.jar
COPY --from=opencv-builder /opencv-dist/ /opt/opencv/
COPY --from=frontend-builder /frontend/dist/ ./web/dist/
COPY --from=russiandocs-models /models/ ./document_processing/models/
COPY --from=russiandocs-models /config/ ./document_processing/config/
COPY samples/ ./samples/
COPY service/seed_data/ ./service/seed_data/

# Same assertion as the CPU stage. Repeated deliberately: the two runtime images
# have different bases, and the Go port's GPU image failed the equivalent check
# after the CPU one passed.
RUN set -eu; \
    jni=$(ls /opt/opencv/lib/libopencv_java*.so); \
    missing=$(LD_LIBRARY_PATH=/opt/opencv/lib ldd "$jni" 2>&1 | grep -c 'not found' || true); \
    [ "$missing" = "0" ] || { \
        echo "FAIL: $missing unresolved shared libraries in $jni:"; \
        LD_LIBRARY_PATH=/opt/opencv/lib ldd "$jni" | grep 'not found'; \
        exit 1; }; \
    echo "$(basename "$jni"): all shared libraries resolve"

ENV RDOCS_REPO_ROOT=/app \
    RDOCS_OPENCV_HOME=/opt/opencv \
    LD_LIBRARY_PATH=/opt/opencv/lib \
    DATA_DIR=/app/data \
    DATA_WIPE_ON_START=true \
    COMPUTE_DEVICE=auto \
    JAVA_TOOL_OPTIONS="-XX:MaxRAMPercentage=75 -XX:+UseContainerSupport"

# Declared AFTER everything cacheable, deliberately: an ARG invalidates every layer below it, and higher
# up a new commit hash would rebuild OpenCV from source. Same placement as the root Dockerfile's. The
# service reads GIT_COMMIT and shows it on the status page; without it the page reads `unknown`, which is
# indistinguishable from a deployment nobody can identify — seen on the first container run.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT

RUN useradd --create-home --shell /usr/sbin/nologin rdocs \
    && mkdir -p /app/data \
    && chown -R rdocs:rdocs /app/data
USER rdocs

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["java", "-jar", "/app/rdocs-service.jar", "--healthcheck", "--addr", ":8005"]

ENTRYPOINT ["java", "-jar", "/app/rdocs-service.jar"]
CMD ["--addr", ":8005"]
