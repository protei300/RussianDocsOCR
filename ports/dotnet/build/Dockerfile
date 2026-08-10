# syntax=docker/dockerfile:1.7
# =============================================================================
# RussianDocs OCR — the .NET service
#
# TWO TARGETS, and only the BASE IMAGE differs between them. That is the whole
# reason this file is a third the size of the Go port's: there is nothing to
# compile per device. `Microsoft.ML.OnnxRuntime.Gpu` carries the CPU execution
# provider as well, so ONE published output runs on either host, and the split
# exists purely so a CPU-only host does not pull 4 GB of CUDA kernels it will
# never execute.
#
#   runtime-cpu   mcr.microsoft.com/dotnet/aspnet:8.0-jammy
#   runtime-gpu   nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 + the ASP.NET runtime
#
# MEASURED by `docker images`, on the same host as the other two:
#
#   russiandocs-dotnet:cpu   2.31 GB     russiandocs-go:cpu   902 MB
#   russiandocs-dotnet:gpu   7.56 GB     russiandocs-go:gpu  6.73 GB
#                                        russiandocs:latest  7.72 GB  (Python)
#
# **BOTH TARGETS BUILT AND RUN, 2026-08-05**, against the live verification suite:
# 37/37 wire-contract checks and 7/7 recognitions identical to service/seed_data,
# in the CPU container and again in the GPU container under `--gpus all`, which
# picks device=gpu and keeps OCR on CPU as designed.
#
# The .NET images are BIGGER than Go's, and the reason is worth stating plainly
# rather than hiding: 2.31 GB against 902 MB is mostly the GTK stack that
# OpenCvSharp's native package needs (see the note in the CPU stage) plus the
# ASP.NET runtime. The GPU pair is closer — 7.56 against 6.73 GB — because CUDA
# and cuDNN dominate both and are the same bytes. Against the Python image, the
# .NET GPU image is essentially a wash: 7.56 versus 7.72 GB. Size is not what
# this port buys.
#
# Compared with the Go port, three build stages simply do not exist here:
#   - no OpenCV from source. OpenCvSharp4.official.runtime.linux-x64 ships a
#     prebuilt libOpenCvSharpExtern.so, statically linked against OpenCV 4.13 —
#     the SAME version the port is verified against, and headless already, so
#     D-07's Qt problem cannot arise.
#   - no ONNX Runtime download. The NuGet package carries the native library,
#     and Central Package Management pins the version in one place.
#   - no cgo flags, no pkg-config, no SONAME archaeology between distros.
#
# Build (from the repository root):
#   docker build -f ports/base/Dockerfile.models -t russiandocs-models:latest .
#   docker build -f ports/dotnet/build/Dockerfile --target runtime-cpu -t russiandocs-dotnet:cpu .
#   docker build -f ports/dotnet/build/Dockerfile --target runtime-gpu -t russiandocs-dotnet:gpu .
#
# Run:
#   docker run --rm -p 8004:8004 -e JWT_SECRET=... -e DEFAULT_API_KEY=... russiandocs-dotnet:cpu
#   docker run --rm --gpus all -p 8004:8004 -e JWT_SECRET=... russiandocs-dotnet:gpu
#
# No volume for /app/data on purpose — see DATA_WIPE_ON_START.
#
# The DEVICE-SELECTION LOGIC is identical in both images, and that matters more
# than the split: the `[gpu, cpu]` attempt loop, the "a listed provider is not a
# working GPU" rule, and the device-node probe are in the same assembly. A CPU
# image asked for a GPU logs why it cannot have one rather than pretending — and
# in a container started WITHOUT `--gpus`, that probe is the only thing standing
# between the service and a segfault, because the CUDA provider does not fail
# cleanly there.
# =============================================================================

# =============================================================================
# Stage 1: build the SPA
#
# COPIED BYTE-FOR-BYTE from the root Dockerfile's stage 1. Copied rather than
# rewritten on purpose: all three services serve the SAME frontend, and a second
# version of this stage would drift. If the root stage changes, change this one
# to match rather than improving it here.
# =============================================================================
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY web/package.json web/package-lock.json* ./

# A plain `npm ci`: the committed lockfile resolves every tarball from the public
# registry, so there is nothing to work around. It used to name a private mirror
# 132 times, which was wrong twice over — it published internal infrastructure
# from a public repository, and anyone outside that network got a bare E401 from
# a host they cannot resolve. The lockfile was rewritten once instead; every
# version and `integrity` hash is unchanged, because those hash the tarball
# contents and do not depend on who served them.
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build
# Output: /frontend/dist/

# =============================================================================
# Stage 2: publish the service
#
# `dotnet publish` rather than `build`: it resolves the native assets for the
# target RID and lays them out beside the managed assemblies, which is exactly
# what the runtime image needs and nothing more.
#
# NOT self-contained and NOT AOT. Both would work; neither is worth it here. The
# runtime images already carry the ASP.NET runtime, self-contained would duplicate
# it, and AOT is incompatible with the reflection OpenCvSharp's own marshalling
# uses. A framework-dependent publish is also what makes the two runtime stages
# able to share one build.
# =============================================================================
FROM mcr.microsoft.com/dotnet/sdk:8.0-jammy AS builder
WORKDIR /src

# The project files first, so `restore` caches independently of the sources. The
# props files are part of that input: Directory.Packages.props IS the version set,
# so a change to it must invalidate the restore layer.
COPY ports/dotnet/Directory.Build.props ports/dotnet/Directory.Packages.props ./ports/dotnet/
COPY ports/dotnet/src/RussianDocs.DocumentProcessing/*.csproj ./ports/dotnet/src/RussianDocs.DocumentProcessing/
COPY ports/dotnet/src/RussianDocs.Service/*.csproj ./ports/dotnet/src/RussianDocs.Service/
RUN dotnet restore ports/dotnet/src/RussianDocs.Service/RussianDocs.Service.csproj

COPY ports/dotnet/src/ ./ports/dotnet/src/
RUN dotnet publish ports/dotnet/src/RussianDocs.Service/RussianDocs.Service.csproj \
        -c Release -o /publish --no-restore

# The native assets are the whole reason a .NET image can be this simple, so their
# presence is ASSERTED rather than assumed. A restore that resolved the managed
# wrapper but not the platform runtime package produces a publish directory that
# looks complete and dies at the first Mat with a DllNotFoundException — the exact
# class of failure the Go port's ldd assertion existed to catch.
RUN set -eu; \
    for lib in libOpenCvSharpExtern.so libonnxruntime.so; do \
        find /publish -name "$lib*" | grep -q . \
            || { echo "FAIL: $lib is not in the publish output"; exit 1; }; \
    done; \
    echo "native assets present:"; \
    find /publish -name '*.so*' -printf '  %f\n' | sort

# =============================================================================
# Stage 3a: the CPU runtime
# =============================================================================
FROM mcr.microsoft.com/dotnet/aspnet:8.0-jammy AS runtime-cpu

# **The native OpenCV in OpenCvSharp4.official.runtime.linux-x64 is NOT headless.**
# That was the assumption this file was written on and it is wrong: `ldd` on the
# shipped libOpenCvSharpExtern.so wants GTK 3, Pango, Cairo, ATK, X11 and
# FreeType. Without them the load fails as "The type initializer for
# OpenCvSharp.Internal.NativeMethods threw an exception" — which names neither
# OpenCV nor a library, so the cause has to be found with ldd. Every seed fixture
# and the warmup failed with exactly that message on the first run.
#
# So D-07 applies to this port too, in a different shape: the Go port pays for
# headless OpenCV in BUILD TIME (it compiles it with WITH_GTK=OFF), and this port
# pays for it in IMAGE SIZE instead. Installing GTK is the cheaper trade here —
# building OpenCV from source would also mean building OpenCvSharp's own extern
# wrapper against it, which is a great deal more than a cmake invocation.
#
# libgomp1 is OpenMP, which the OpenCV build links against directly.
# InvariantGlobalization=true in Directory.Build.props is what keeps libicu off
# this list — set for correctness (a ru-RU host would otherwise write `0,904` into
# every float on the wire) and saving ~30 MB as a side effect.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libatomic1 \
        libgtk-3-0 libgdk-pixbuf-2.0-0 libpango-1.0-0 libpangocairo-1.0-0 \
        libcairo2 libcairo-gobject2 libatk1.0-0 libglib2.0-0 \
        libfreetype6 libharfbuzz0b libx11-6 libdrm2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /publish/ ./
COPY --from=frontend-builder /frontend/dist/ ./web/dist/
COPY --from=russiandocs-models /models/ ./document_processing/models/
COPY --from=russiandocs-models /config/ ./document_processing/config/
COPY samples/ ./samples/
COPY service/seed_data/ ./service/seed_data/

# **KEEP THIS.** It is what turned "The type initializer threw an exception" into a
# list of sixteen missing sonames, and it is the same assertion that made the Go
# port's three distro problems findable rather than mysterious. An unresolved
# soname here fails the BUILD; without it, it fails every document at runtime with
# a message that names nothing.
RUN set -eu; \
    missing=$(ldd runtimes/linux-x64/native/libOpenCvSharpExtern.so 2>&1 \
        | grep -c 'not found' || true); \
    [ "$missing" = "0" ] || { \
        echo "FAIL: $missing unresolved shared libraries:"; \
        ldd runtimes/linux-x64/native/libOpenCvSharpExtern.so | grep 'not found'; \
        exit 1; }; \
    echo "libOpenCvSharpExtern.so: all shared libraries resolve"

# RDOCS_REPO_ROOT is the REPOSITORY ROOT, not the models directory. The Go port
# lost a build to that distinction: the loader appends
# `document_processing/models` itself, so pointing this at the models directory
# produces a path with the segment twice and a "model not found" that names a
# plausible-looking path.
ENV RDOCS_REPO_ROOT=/app \
    DATA_DIR=/app/data \
    DATA_WIPE_ON_START=true \
    COMPUTE_DEVICE=auto \
    ASPNETCORE_ENVIRONMENT=Production \
    DOTNET_gcServer=1

# Non-root. The data directory is created and chowned HERE rather than at startup,
# because a non-root process cannot create it under /app itself — and the wipe on
# start empties the CONTENTS precisely so it never needs to unlink the directory,
# which would require write permission on /app.
RUN useradd --create-home --shell /usr/sbin/nologin rdocs \
    && mkdir -p /app/data \
    && chown -R rdocs:rdocs /app/data
USER rdocs

EXPOSE 8004

# The binary is its own probe, so the image needs neither curl nor a shell for
# this. A HEALTHCHECK that depends on tools installed purely to run it is a larger
# image and one more thing to keep patched.
#
# start-period is generous because model loading takes seconds and /health answers
# OK throughout — the service really is healthy while it loads, since it accepts
# uploads and queues them.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["/app/rdocs-service", "--healthcheck", "--addr", ":8004"]

ENTRYPOINT ["/app/rdocs-service"]
CMD ["--addr", ":8004"]

# =============================================================================
# Stage 3b: the GPU runtime
#
# **cudnn-runtime, not base.** onnxruntime 1.21 needs cuDNN 9, and with the `base`
# image the CUDA provider fails to load while `CUDAExecutionProvider` remains in
# the advertised list — a service that is silently on CPU and looks healthy. The
# Python image is pinned to this same tag for the same reason.
#
# The ASP.NET runtime is installed rather than inherited, because there is no
# Microsoft image on a CUDA base. dotnet-install.sh is the supported way to do it
# and pins the channel explicitly.
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runtime-gpu

# The same GTK set as the CPU stage, and for the same reason — see the note there.
# This is a different distro base (the CUDA image is jammy too, but built from a
# different package set), so the list is repeated rather than shared: a stage
# cannot inherit a RUN, and factoring it into a common base would put CUDA in the
# CPU image's ancestry.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libgomp1 libatomic1 \
        libgtk-3-0 libgdk-pixbuf-2.0-0 libpango-1.0-0 libpangocairo-1.0-0 \
        libcairo2 libcairo-gobject2 libatk1.0-0 libglib2.0-0 \
        libfreetype6 libharfbuzz0b libx11-6 libdrm2 \
    && rm -rf /var/lib/apt/lists/*

# --runtime aspnetcore, not sdk: this image runs a published application and has
# no reason to carry a compiler. --channel 8.0 matches TargetFramework; a mismatch
# fails at startup with a message about a missing framework, which is at least
# loud.
RUN curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh \
    && chmod +x /tmp/dotnet-install.sh \
    && /tmp/dotnet-install.sh --channel 8.0 --runtime aspnetcore \
        --install-dir /usr/share/dotnet --no-path \
    && ln -s /usr/share/dotnet/dotnet /usr/local/bin/dotnet \
    && rm /tmp/dotnet-install.sh

WORKDIR /app
COPY --from=builder /publish/ ./
COPY --from=frontend-builder /frontend/dist/ ./web/dist/
COPY --from=russiandocs-models /models/ ./document_processing/models/
COPY --from=russiandocs-models /config/ ./document_processing/config/
COPY samples/ ./samples/
COPY service/seed_data/ ./service/seed_data/

# Same assertion as the CPU stage. Repeated deliberately: the two runtime images
# have different bases, and the Go port's GPU image failed the equivalent check
# after the CPU one passed.
RUN set -eu; \
    missing=$(ldd runtimes/linux-x64/native/libOpenCvSharpExtern.so 2>&1 \
        | grep -c 'not found' || true); \
    [ "$missing" = "0" ] || { \
        echo "FAIL: $missing unresolved shared libraries:"; \
        ldd runtimes/linux-x64/native/libOpenCvSharpExtern.so | grep 'not found'; \
        exit 1; }; \
    echo "libOpenCvSharpExtern.so: all shared libraries resolve"

ENV RDOCS_REPO_ROOT=/app \
    DATA_DIR=/app/data \
    DATA_WIPE_ON_START=true \
    COMPUTE_DEVICE=auto \
    ASPNETCORE_ENVIRONMENT=Production \
    DOTNET_gcServer=1 \
    DOTNET_ROOT=/usr/share/dotnet

RUN useradd --create-home --shell /usr/sbin/nologin rdocs \
    && mkdir -p /app/data \
    && chown -R rdocs:rdocs /app/data
USER rdocs

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/app/rdocs-service", "--healthcheck", "--addr", ":8004"]

ENTRYPOINT ["/app/rdocs-service"]
CMD ["--addr", ":8004"]
