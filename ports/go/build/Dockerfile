# syntax=docker/dockerfile:1.7
# =============================================================================
# RussianDocs OCR — the Go service
#
# TWO TARGETS, one file. This is a deliberate departure from the Python image,
# which manages with one: `onnxruntime-gpu` is a single wheel that imports fine
# without a GPU, so one Python image serves both hosts. Go links a specific
# ONNX Runtime shared library, and the GPU build is ~450 MB of CUDA kernels that
# a CPU host would download and never execute.
#
#   runtime-cpu   902 MB   debian:bookworm-slim + CPU libonnxruntime
#   runtime-gpu  6.73 GB   nvidia/cuda:12.6.3-cudnn-runtime + GPU libonnxruntime
#
# MEASURED, by `docker images`, against the Python image's 7.72 GB on the same host.
#
# **Go does not make the GPU image small, and the measurement is blunter than the
# estimate was: 6.73 GB against 7.72 GB is 13% off, not a different league.** Almost
# all of it is CUDA and cuDNN — the same bytes the Python image carries.
#
# The CPU image is the real difference: 902 MB. Python ships no CPU-only image at
# all (one `onnxruntime-gpu` wheel serves both), so a CPU-only host running the
# reference pulls the whole 7.72 GB. That is a packaging difference, not a language
# one, and it should be quoted that way.
#
# The 902 MB is also nearly double the 390 MB estimated here before anything was
# built, because the estimate omitted OpenCV (93 MB) and samples/ (53 MB). The floor
# is the 225 MB of model weights, which every port pays identically.
#
# The DEVICE-SELECTION LOGIC is identical in both, and that matters more than the
# split: the `[gpu, cpu]` attempt loop, the "a listed provider is not a working
# GPU" rule, and the device-node probe are compiled into each image. A CPU image
# asked for a GPU logs why it cannot have one rather than pretending.
#
# Build (from the repository root):
#   docker build -f ports/base/Dockerfile.models -t russiandocs-models:latest .
#   docker build -f ports/go/build/Dockerfile --target runtime-cpu -t russiandocs-go:cpu .
#   docker build -f ports/go/build/Dockerfile --target runtime-gpu -t russiandocs-go:gpu .
#
# Run:
#   docker run --rm -p 8003:8003 -e JWT_SECRET=... -e DEFAULT_API_KEY=... russiandocs-go:cpu
#   docker run --rm --gpus all -p 8003:8003 -e JWT_SECRET=... russiandocs-go:gpu
#
# No volume for /app/data on purpose — see DATA_WIPE_ON_START.
#
# **STATUS: BUILT AND RUN, both targets, 2026-08-05.** Both images serve the SPA,
# seed their log from service/seed_data and recognise documents; the GPU image picks
# `device=gpu` under `--gpus all` and keeps OCR on CPU as designed.
#
# The prediction that "the first docker build is part of the work" was right: it took
# eight fixes, and every one of them was in a place this header had flagged as
# unverified. Worth listing, because they are the traps a second port will meet:
#
#   1. `.dockerignore` excluded ports/ entirely — its allowlist predates the ports.
#   2. ORT 1.21.1 publishes NO linux-x64 CPU tarball; only the GPU one. Hence 1.21.0.
#   3. `curl` without -f wrote a 404 page into the .tgz; tar failed, blaming the archive.
#   4. golang:1.24 against `go 1.26.5` in go.mod — GOTOOLCHAIN=local means no rescue.
#   5. `... && go build ... || true` made a COMPILE ERROR a green layer. The smoke probe
#      is now its own RUN.
#   6. `-tags customenv` means gocv contributes no cgo flags — they are ours to pass.
#   7. Codec SONAMEs differ between the two runtime distros, so codecs are now built
#      into OpenCV. See the cmake flags.
#   8. RDOCS_MODELS_ROOT wants the repository root, not the models directory.
#
# What made 6 through 8 findable rather than mysterious is the ldd assertion in each
# runtime stage. Keep it.
# =============================================================================

# 1.21.x is REQUIRED, not preferred: onnxruntime_go v1.19.0 vendors ORT_API_VERSION 21, and a
# newer runtime rejects the handshake. The reference runs 1.21.1.
#
# **1.21.0 rather than 1.21.1, and only because 1.21.1 publishes no CPU tarball** — that release
# ships `onnxruntime-linux-x64-gpu-1.21.1.tgz` and no `onnxruntime-linux-x64-1.21.1.tgz`, so the
# CPU image cannot be built from it. Checked against the GitHub release API, after the download
# failed. The patch difference against the reference is safe on the evidence already gathered:
# the spike measured ORT 1.21.1 against 1.28.0 on CPU and got bit-identical tensors across seven
# outputs, so a patch bump inside 1.21 cannot move numbers that two minor versions did not.
ARG ONNXRUNTIME_VERSION=1.21.0

# =============================================================================
# Stage 1: build the SPA
#
# COPIED BYTE-FOR-BYTE from the root Dockerfile's stage 1, including the sed.
# Copied rather than rewritten on purpose: the two services serve the SAME
# frontend, and a second version of this stage would drift. If the root stage
# changes, change this one to match rather than improving it here.
# =============================================================================
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY web/package.json web/package-lock.json* ./

# A plain `npm ci`: the committed lockfile resolves every tarball from the public
# registry, so there is nothing to work around.
#
# This stage used to rewrite the lockfile at build time, because the lockfile had been
# generated against a private mirror and named its host 132 times. That was wrong twice
# over — it published internal infrastructure from a public repository, and anyone
# outside that network got a bare `E401` from a host they cannot resolve, with no hint
# why. The lockfile was rewritten once instead. Every version and `integrity` hash is
# unchanged: those hash the tarball contents and do not depend on who served them.
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build
# Output: /frontend/dist/

# =============================================================================
# Stage 2: OpenCV, built HEADLESS
#
# D-07, and it is worth the build time. gocv links highgui UNCONDITIONALLY
# (window.go exists in the package), so a service that will never draw a window
# drags Qt or GTK into the image otherwise — hundreds of megabytes for nothing.
# Built with WITH_QT=OFF and WITH_GTK=OFF, highgui becomes a stub that pulls
# nothing. This is exactly what `opencv-python-headless` does, which is what the
# Python service already uses, so the two match.
#
# The version is pinned to 4.13.0 because that is what the port is verified
# against. 4.6 from Debian's apt is NOT acceptable: contour approximation
# changed in 4.8, and the conformance goldens would stop matching for a reason
# nothing in the code explains.
# =============================================================================
# The tag must satisfy `go 1.26.5` in ports/go/go.mod. 1.24-bookworm was here and the build
# failed with `go.mod requires go >= 1.26.5 (running go 1.24.13; GOTOOLCHAIN=local)` — the
# official images set GOTOOLCHAIN=local, so the toolchain will NOT silently download a newer one
# to rescue the build. That is the right default and the reason the mismatch was loud.
FROM golang:1.26-bookworm AS opencv-builder

ARG OPENCV_VERSION=4.13.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config unzip curl ca-certificates \
        libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/opencv
RUN curl -sSL -o opencv.zip \
        "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip" \
    && curl -sSL -o contrib.zip \
        "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip" \
    && unzip -q opencv.zip && unzip -q contrib.zip && rm opencv.zip contrib.zip

# The module list is an ALLOWLIST rather than a set of exclusions. The pipeline
# uses roughly thirty functions across core, imgproc, imgcodecs and calib3d;
# building the rest costs time and image size and buys nothing. gocv still needs
# highgui to LINK, which is why it is present and headless rather than absent.
RUN mkdir -p build && cd build && cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv/opencv_contrib-${OPENCV_VERSION}/modules \
        -D BUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,highgui,video,videoio,objdetect,photo,dnn \
        -D WITH_QT=OFF -D WITH_GTK=OFF -D WITH_WIN32UI=OFF \
        -D WITH_FFMPEG=OFF -D WITH_GSTREAMER=OFF -D WITH_V4L=OFF \
        -D WITH_OPENEXR=OFF -D WITH_IPP=OFF -D WITH_TBB=OFF -D WITH_OPENCL=OFF \
        # Codecs built from OpenCV's own 3rdparty tree instead of linked against the distro's.
        # This removes a whole class of problem rather than an instance of it: the CPU runtime is
        # Debian bookworm and the GPU runtime is the CUDA image's Ubuntu jammy, and the two
        # disagree on codec SONAMEs — bookworm has libjpeg.so.62 and libtiff.so.6, jammy has
        # libjpeg.so.8 and libtiff.so.5. An OpenCV compiled on one therefore cannot load on the
        # other, and no choice of package names fixes it. The GPU image failed exactly there,
        # caught by the ldd assertion. Bundling is also what the opencv-python wheels do.
        -D BUILD_JPEG=ON -D BUILD_PNG=ON -D BUILD_TIFF=ON -D BUILD_WEBP=ON \
        -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
        -D BUILD_DOCS=OFF -D BUILD_opencv_apps=OFF -D BUILD_opencv_python3=OFF \
        -D BUILD_SHARED_LIBS=ON \
        /tmp/opencv/opencv-${OPENCV_VERSION} \
    && make -j"$(nproc)" && make install && ldconfig \
    && rm -rf /tmp/opencv

# =============================================================================
# Stage 3: ONNX Runtime, both flavours
#
# Downloaded rather than built: these are the same official binaries the Python
# wheel bundles, and the version must MATCH what the port is verified against.
# onnxruntime_go v1.19.0 vendors ORT_API_VERSION 21, and the C API is backward
# compatible one way only — a newer library serves an older request, never the
# reverse. Against a mismatch the failure is GetApi() returning NULL, which
# surfaces as "broken or missing library" and says nothing about versions.
# =============================================================================
FROM debian:bookworm-slim AS ort-downloader
ARG ONNXRUNTIME_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates tar \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /ort
# `curl -f` matters here and its absence cost a build: without it a 404 writes GitHub's
# HTML error page into cpu.tgz and the failure surfaces as `tar: Error is not recoverable`,
# which points at the archive instead of at the URL. Fail at the download.
RUN set -eu; \
    base="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}"; \
    curl -fsSL -o cpu.tgz "${base}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"; \
    curl -fsSL -o gpu.tgz "${base}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz"; \
    mkdir -p cpu gpu; \
    tar -xzf cpu.tgz -C cpu --strip-components=1; \
    tar -xzf gpu.tgz -C gpu --strip-components=1; \
    rm cpu.tgz gpu.tgz

# =============================================================================
# Stage 4: compile the Go binaries
#
# CGO_ENABLED=1 is not optional. onnxruntime_go IS a cgo wrapper and gocv IS a
# cgo wrapper; there is no pure-Go build of this port, not even in principle.
# The spike settled that as a question: pure Go gives 58-83% of pixels differing
# by up to 14 LSB, so "1e-3 on numeric outputs" is unreachable without OpenCV.
# =============================================================================
FROM opencv-builder AS go-builder

ENV CGO_ENABLED=1 \
    GOFLAGS=-mod=mod

# **`-tags customenv` means gocv supplies NO cgo flags of its own — they are mine to provide.**
# That is the entire purpose of the tag, and forgetting it here cost a build: every gocv header
# failed with `opencv2/opencv.hpp: No such file or directory`, which reads like a broken OpenCV
# install rather than a missing -I.
#
# Taken from pkg-config, exactly as build.ps1 does on Windows via pkgconf, so the tag means the
# same thing on both platforms and only the modules actually built get linked — the module list
# above is an allowlist, so gocv's built-in flag set would try to link libraries that do not
# exist in this image.
# The flags are computed inside the build RUN, not set as ENV: an ENV value cannot be produced by
# a command at build time.
#
# **Derived from the installed .so files, NOT from pkg-config and not from a hand-written list.**
# pkg-config was the first attempt and it silently produced empty flags: OpenCV 4.13 does not
# generate opencv4.pc unless built with OPENCV_GENERATE_PKGCONFIG=ON, and there is no .pc file
# anywhere in the image (checked, rather than assumed, after `cgo cppflags:` printed blank).
# Adding that cmake flag would work but costs a full OpenCV rebuild; enumerating the libraries
# that were actually installed costs nothing and keeps ONE source of truth — the BUILD_LIST
# allowlist above decides which .so files exist, and this loop links exactly those. A hardcoded
# `-lopencv_...` list would be a second copy of that allowlist, free to drift.
ENV CGO_CXXFLAGS="--std=c++11 -DNDEBUG"

WORKDIR /src
# Dependencies first, as their own layer: go.mod and go.sum change far less often
# than the source, so an edit to a .go file does not re-download the module graph.
COPY ports/go/go.mod ports/go/go.sum ./
RUN go mod download

COPY ports/go/ ./
# The ORT headers only have to satisfy the compiler; the shared library is
# resolved at RUN time from ORT_SO. Linking against the GPU build here and
# running the CPU library later is fine — the C API is the same.
COPY --from=ort-downloader /ort/cpu/include /usr/local/include/onnxruntime

# -s -w strips the symbol table and DWARF: about 30% off a cgo binary, and a
# stack trace from a stripped Go binary still names functions, because Go's own
# runtime metadata is separate from DWARF.
RUN set -eu; \
    libs=""; \
    for so in /usr/local/lib/libopencv_*.so; do \
        base="$(basename "$so" .so)"; \
        libs="$libs -l${base#lib}"; \
    done; \
    [ -n "$libs" ] || { echo "FAIL: no libopencv_*.so in /usr/local/lib"; exit 1; }; \
    export CGO_CPPFLAGS="-I/usr/local/include/opencv4"; \
    export CGO_LDFLAGS="-L/usr/local/lib$libs"; \
    echo "cgo cppflags: $CGO_CPPFLAGS"; \
    echo "cgo ldflags: $CGO_LDFLAGS"; \
    go build -tags customenv -ldflags="-s -w" -o /out/rdocs-service ./cmd/rdocs-service; \
    go build -tags customenv -ldflags="-s -w" -o /out/rdocs-conform ./cmd/rdocs-conform

# The smoke probe is a SEPARATE RUN, and that is the whole point. It used to be the third link
# of the chain above, ending in `|| true` — which in `A && B && C || true` makes the ENTIRE chain
# succeed, `go build` included. A compile error therefore produced a green layer, and the failure
# reappeared two stages later as `"/out/rdocs-service": not found`, which says nothing about a
# compiler. Build failures must be fatal; only this probe is advisory.
RUN /out/rdocs-service --help 2>&1 | head -1 || true

# =============================================================================
# Stage 5a: the CPU runtime
# =============================================================================
FROM debian:bookworm-slim AS runtime-cpu

ENV DEBIAN_FRONTEND=noninteractive \
    LOG_LEVEL=INFO \
    DATA_DIR=/app/data \
    DATA_WIPE_ON_START=true \
    COMPUTE_DEVICE=auto \
    ORT_SO=/usr/local/lib/libonnxruntime.so \
    # A ROOT that CONTAINS document_processing/models — not the models directory itself.
    # Set to the latter, the container started, served the SPA and failed every document with
    # `RDOCS_MODELS_ROOT=... has no document_processing/models`. Found on the first real run.
    RDOCS_MODELS_ROOT=/app \
    RDOCS_REPO_ROOT=/app

# **No codec packages here.** OpenCV bundles its own (BUILD_JPEG and friends, stage 2), which is
# what allows ONE OpenCV build to serve both this Debian runtime and the Ubuntu-based CUDA one.
#
# The history is worth keeping, because it explains why bundling and not a package list: this line
# first listed bookworm's codecs and the container died on start with `libwebpmux.so.3: cannot
# open shared object file`; adding the webp packages fixed THIS image and left the GPU image
# broken on `libjpeg.so.62`/`libtiff.so.6`, which jammy does not have under those names at all.
# Two runtimes on two distros cannot share one externally-linked OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencv-builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
COPY --from=ort-downloader /ort/cpu/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY --from=go-builder /out/rdocs-service /out/rdocs-conform /usr/local/bin/
# **A missing runtime library must fail the BUILD, not the deployment.** This is here because it
# did fail the deployment once: the image built clean and the container died on start with
# `libwebpmux.so.3: cannot open shared object file`. ldd resolves the whole graph, so one line
# covers OpenCV, ONNX Runtime and libstdc++ at once.
RUN set -eu;     missing="$(ldd /usr/local/bin/rdocs-service | grep "not found" || true)";     [ -z "$missing" ] || { echo "FAIL: unresolved shared libraries:"; echo "$missing"; exit 1; };     echo "shared libraries: all resolved"
COPY --from=frontend-builder /frontend/dist /app/web/dist
# The models come from the shared image, so this 215 MB layer is built and
# pulled once per host rather than once per port.
COPY --from=russiandocs-models:latest /models /app/document_processing/models
COPY --from=russiandocs-models:latest /config /app/document_processing/config
# Anonymised samples only — the warmup image is re-read at every start, so a real
# document here would be read on every boot of every deployment.
COPY samples /app/samples
# The pre-computed seed corpus. Generated by the PYTHON service
# (service/tools/build_seed_data.py) and read by both — one corpus, never a per-port copy, so
# the two services seed byte-identical rows. Without it the container starts with an empty log
# and `[SEED] no pre-computed data`, which is what the first run did.
COPY service/seed_data /app/service/seed_data

# Non-root. The data directory is created and owned before the drop, because a
# service that cannot write its own store fails on the first upload rather than
# at startup, which is the harder failure to read.
RUN useradd --system --create-home --uid 10001 rdocs \
    && mkdir -p /app/data && chown -R rdocs:rdocs /app/data
USER rdocs

EXPOSE 8003
# Reports OK while the models are still loading, deliberately: the service IS
# healthy then — it accepts uploads and queues them. Gating health on the
# runtime would make Docker kill the container during startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/rdocs-service", "--healthcheck"]
ENTRYPOINT ["/usr/local/bin/rdocs-service"]
CMD ["-addr", ":8003"]

# =============================================================================
# Stage 5b: the GPU runtime
#
# `cudnn-runtime`, NOT `base`, and the reason is the same one the Python image
# documents: onnxruntime 1.21 needs cuDNN 9 present. With `base` the CUDA
# provider fails to load, onnxruntime falls back to CPU, and the status page
# would still be able to list CUDAExecutionProvider — a slow service that looks
# healthy. This port narrows that particular trap by reporting OBSERVED
# providers (D-13), but the image still has to be right.
#
# The tag matches the Python image exactly, so a host running both pulls the
# CUDA layers once.
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runtime-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    LOG_LEVEL=INFO \
    DATA_DIR=/app/data \
    DATA_WIPE_ON_START=true \
    COMPUTE_DEVICE=auto \
    ORT_SO=/usr/local/lib/libonnxruntime.so \
    # A ROOT that CONTAINS document_processing/models — not the models directory itself.
    # Set to the latter, the container started, served the SPA and failed every document with
    # `RDOCS_MODELS_ROOT=... has no document_processing/models`. Found on the first real run.
    RDOCS_MODELS_ROOT=/app \
    RDOCS_REPO_ROOT=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencv-builder /usr/local/lib/libopencv_*.so* /usr/local/lib/
COPY --from=ort-downloader /ort/gpu/lib/libonnxruntime*.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY --from=go-builder /out/rdocs-service /out/rdocs-conform /usr/local/bin/
# **A missing runtime library must fail the BUILD, not the deployment.** This is here because it
# did fail the deployment once: the image built clean and the container died on start with
# `libwebpmux.so.3: cannot open shared object file`. ldd resolves the whole graph, so one line
# covers OpenCV, ONNX Runtime and libstdc++ at once.
RUN set -eu;     missing="$(ldd /usr/local/bin/rdocs-service | grep "not found" || true)";     [ -z "$missing" ] || { echo "FAIL: unresolved shared libraries:"; echo "$missing"; exit 1; };     echo "shared libraries: all resolved"
COPY --from=frontend-builder /frontend/dist /app/web/dist
COPY --from=russiandocs-models:latest /models /app/document_processing/models
COPY --from=russiandocs-models:latest /config /app/document_processing/config
COPY samples /app/samples
# The pre-computed seed corpus. Generated by the PYTHON service
# (service/tools/build_seed_data.py) and read by both — one corpus, never a per-port copy, so
# the two services seed byte-identical rows. Without it the container starts with an empty log
# and `[SEED] no pre-computed data`, which is what the first run did.
COPY service/seed_data /app/service/seed_data

RUN useradd --system --create-home --uid 10001 rdocs \
    && mkdir -p /app/data && chown -R rdocs:rdocs /app/data
USER rdocs

EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/rdocs-service", "--healthcheck"]
# --init is RECOMMENDED for this target specifically, and D-12 is why: the CUDA
# provider is reported to overwrite Go's signal handlers, in which case SIGTERM
# never reaches the process and the container is killed after the grace period
# instead of shutting down cleanly. tini forwards signals from PID 1 and reaps,
# which does not fix the in-process handler but does bound the damage.
#
#   docker run --init --gpus all ...
ENTRYPOINT ["/usr/local/bin/rdocs-service"]
CMD ["-addr", ":8003"]
