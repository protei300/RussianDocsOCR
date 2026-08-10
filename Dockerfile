# syntax=docker/dockerfile:1.7
# =============================================================================
# RussianDocs OCR — recognition service
#
# ONE image that runs on a GPU host and on a CPU-only host unchanged. There is
# no separate CPU build: `onnxruntime-gpu` imports fine without a GPU, and
# `service/ml/runtime.py` tries `[gpu, cpu]` in order and logs which one it
# actually got. Building a real Pipeline is the only honest GPU test —
# `'CUDAExecutionProvider' in get_available_providers()` is true merely because
# the package is installed.
#
# Build:
#   docker build -t russiandocs:latest .
#
# Run (CPU):
#   docker run --rm -p 8002:8002 -e JWT_SECRET=... -e DEFAULT_API_KEY=... russiandocs:latest
# Run (GPU):
#   docker run --rm --gpus all -p 8002:8002 -e JWT_SECRET=... russiandocs:latest
#
# No volume for /app/data on purpose — see DATA_WIPE_ON_START below.
# Env var reference at the bottom of this file.
# =============================================================================

# =============================================================================
# Stage 1: build the SPA
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
# Stage 2: the service
#
# `cudnn-runtime`, not `base`. The sibling services get away with `base` because
# cuDNN and cuBLAS arrive inside the torch wheels; there is no torch here, and
# onnxruntime 1.21 needs cuDNN 9 present in the image. With `base` the CUDA
# provider fails to load, onnxruntime silently falls back to CPU, and the status
# page would still list CUDAExecutionProvider — a slow service that looks
# healthy. CUDA 12.6 matches onnxruntime-gpu 1.21's build.
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    DATA_DIR=/app/data \
    COMPUTE_DEVICE=auto

# -----------------------------------------------------------------------------
# 1. System packages
#
# Ubuntu 22.04's own python3.10 is used rather than a deadsnakes PPA: the pin set
# supports 3.10-3.12, so the extra repository would buy nothing and add a
# network dependency to the build.
#   libgomp1   - OpenMP runtime, required by onnxruntime and opencv
#   libglib2.0 - still linked by opencv-python-headless despite the name
#   unixodbc   - pyodbc links against libodbc at *import* time, so it is needed
#                even with no database configured. The Microsoft driver itself
#                is NOT installed: nothing here connects to MS SQL yet, and it
#                would add ~100 MB plus an EULA to every build.
#   curl       - the HEALTHCHECK below
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        libgomp1 libglib2.0-0 unixodbc curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# -----------------------------------------------------------------------------
# 2. Dependencies, before the code, so edits do not re-run pip.
#
# `onnxruntime` is swapped for `onnxruntime-gpu` at the same pin. Both provide
# the same `onnxruntime` module, and having both installed leaves whichever
# landed last in site-packages — so the CPU wheel is uninstalled explicitly
# rather than left to shadow the GPU one.
# -----------------------------------------------------------------------------
COPY requirements2026.txt requirements-service.txt ./
RUN python -m pip install --no-cache-dir -r requirements2026.txt \
    && python -m pip install --no-cache-dir -r requirements-service.txt \
    && python -m pip uninstall -y onnxruntime \
    && python -m pip install --no-cache-dir onnxruntime-gpu==1.21.1

# -----------------------------------------------------------------------------
# 3. Application
#
# GIT_COMMIT is declared here, after every cached layer, so bumping it does not
# invalidate the apt and pip work above.
# -----------------------------------------------------------------------------
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT

COPY document_processing/ ./document_processing/
COPY service/ ./service/
COPY samples/ ./samples/
COPY --from=frontend-builder /frontend/dist/ ./web/dist/

# -----------------------------------------------------------------------------
# 4. Build-time assertions.
#
# Both guard against a .dockerignore that quietly drops model files: the service
# would otherwise build and push cleanly and die on its first document. Cheap
# here, expensive to diagnose in a registry.
# -----------------------------------------------------------------------------
RUN <<'PY' python
import pathlib
root = pathlib.Path('document_processing/models')
onnx = sorted(root.rglob('*.onnx'))
cfg = sorted(root.rglob('model.json'))
print(f'models: {len(onnx)} .onnx, {len(cfg)} model.json')
# A model.json carries the input shapes, class list and pre/post-processing
# steps. Weights without it are useless, and it is the file a broad
# `*.json` ignore rule silently eats — hence counting both, not just .onnx.
assert len(onnx) >= 10, f'only {len(onnx)} .onnx weights in the image'
assert len(cfg) >= 10, f'only {len(cfg)} model.json configs in the image'
for required in ('web/dist/index.html', 'service/seed_data/manifest.json'):
    assert pathlib.Path(required).is_file(), f'missing {required}'
import document_processing, service.ml.runtime  # noqa: F401 - import smoke test
print('build assertions OK')
PY

# -----------------------------------------------------------------------------
# Drop privileges. Docker runs a container as root unless told otherwise, and
# this service feeds ATTACKER-SUPPLIED files to an image decoder and to ONNX
# Runtime — native code, and the likeliest place a bug becomes code execution.
# Root there is the difference between a shell as `rdocs` and a shell that can
# start looking for a way out of the container.
#
# The three port images already do this; this one was the last to run
# privileged.
#
# A fixed UID rather than whatever `useradd` picks: a bind-mounted $DATA_DIR
# must be writable by it, and the operator needs a number to `chown` to. If you
# mount a host directory over /app/data, either `chown -R 10001:10001` it on the
# host or run with `--user "$(id -u):$(id -g)"` — otherwise the first upload
# fails with a permission error, which reads as a service bug rather than a
# mount problem.
# -----------------------------------------------------------------------------
RUN useradd --system --create-home --uid 10001 rdocs \
    && mkdir -p /app/data \
    && chown -R rdocs:rdocs /app/data
USER rdocs

# -----------------------------------------------------------------------------
# Health check. `start-period` is generous: 215 MB of ONNX sessions plus a real
# warmup document take several seconds, and the runtime loads in the background
# so /health answers before recognition is ready.
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

EXPOSE 8002

# `--workers 1` is not a tuning choice: the store index and the pipeline
# singleton live per process. main.py refuses to start with more.
ENTRYPOINT ["python", "-m", "uvicorn", "service.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8002", "--workers", "1"]

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
# --- Auth ---
# AUTH_PIN=1234                  # website only; the API uses keys
# JWT_SECRET=<32+ random chars>  # CHANGE THIS
# DEFAULT_API_KEY=rdk_...        # bootstrap key; if unset a random one is
#                                # generated at each start and logged, so
#                                # integrations must be re-pointed after every
#                                # restart
#
# --- Storage ---
# RUSSIANDOCS_DATABASE_CONNECTIONSTRING=  # unset -> temporary storage, wiped at
#                                # every start, and the startup banner says so.
#                                # MS SQL additionally needs msodbcsql18 added
#                                # to stage 2 above.
# DATA_DIR=/app/data
# DATA_WIPE_ON_START=true        # Absence of a volume is NOT enough: `docker
#                                # restart` keeps the writable layer, so without
#                                # this the "data disappears on restart"
#                                # behaviour would silently not hold.
#
# --- Recognition ---
# COMPUTE_DEVICE=auto            # auto | gpu | cpu
# OCR_MODE=accurate              # accurate | fast
# WARMUP_IMAGE=samples/INTPASSPORT_2011/12_CR_INTPASSPORT_2011.jpg
#                                # Only ever an anonymised repository sample.
# =============================================================================
