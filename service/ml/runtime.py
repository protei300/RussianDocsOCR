"""Owning and safely calling the recognition ``Pipeline``.

This is the reference part of the reference project: everything below encodes a
rule that is easy to get wrong and expensive to debug. Each one was verified
against ``document_processing`` 3.0.2, not inferred from documentation.

1.  **A ``Pipeline`` instance is not re-entrant.** ``process_img`` rebinds
    ``self.results`` (``pipeline.py:450``) and ``self.ocr_options``
    (``pipeline.py:465``) on every call, so two concurrent calls on one instance
    silently return each other's fields. This does not crash and does not
    reproduce in single-user testing — it corrupts data under load. Hence
    ``lease_pipeline``: never call ``process_img`` outside it.

2.  **The per-session CUDA lock does not help with (1).** The ``threading.Lock``
    added in commit ``bbf2cfc`` (``processing/inference.py:73``) serialises
    individual ONNX ``Run()`` calls on GPU — it fixes CUDA wedging, not
    pipeline re-entrancy. Different problem, different scope.

3.  **Transform the result before releasing the lease.** ``results`` *is*
    ``pipeline.results``; the next ``process_img`` replaces it. ``recognise``
    below does the whole read-and-convert inside the lease for this reason.

4.  **``Pipeline.warmup()`` cannot report failure.** It swallows every exception
    and ``print``s it (``pipeline.py:417-420``), so a failed warmup looks like a
    successful one and the message never reaches the JSON log. We call
    ``process_img`` directly instead.

5.  **Warmup needs a real document.** With no argument, ``warmup`` builds a
    synthetic grey frame that classifies as ``'NONE'`` and short-circuits before
    the border/field/OCR stages — warming perhaps a fifth of the graph. Pass a
    real sample.

6.  **The library prints to stdout.** Model-loading banners and warmup chatter
    would corrupt a JSON log stream, so stdout is captured and re-emitted
    through the logger at debug level.

7.  **``'CUDAExecutionProvider' in get_available_providers()`` does not mean the
    GPU works.** The provider is listed whenever ``onnxruntime-gpu`` is merely
    installed — including when cuDNN is missing and every session silently
    falls back to CPU. The only honest probe is constructing a real ``Pipeline``
    (which eagerly builds all 12 sessions) and catching the failure, which is
    what ``init_runtime``'s ``[gpu, cpu]`` attempt loop does.

8.  **GPU does not mean GPU OCR.** With ``ocr_gpu_batch=False`` (the correct
    default) the detectors run on CUDA while the OCR engines are forced to CPU
    (``pipeline.py:325-330``), because dynamic-width per-word calls are far
    slower on CUDA. ``DeviceInfo`` reports ``device`` and ``ocr_device``
    separately so the status page can say so instead of claiming "GPU active".

9.  **Models load eagerly and cost 215 MB.** All twelve are built in
    ``Pipeline.__init__`` (~15 s). Instances are expensive; a second one on the
    same card also means a second CUDA context. Hence a pool of size 1 by
    default.

10. **Only this module imports ``document_processing``.** That keeps the rest of
    the service testable without the models, and bounds the work of porting the
    service to another language.
"""
from __future__ import annotations

import contextlib
import ctypes
import io
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from service.ml import transform
from service.ml.errors import PipelineBusy, RuntimeNotReady

log = logging.getLogger(__name__)

#: Seconds a caller waits for a free pipeline before giving up. Short on
#: purpose: a queued job that cannot get a pipeline should go back on the queue
#: (and surface as "degraded"), not block a worker indefinitely.
LEASE_TIMEOUT_SEC = 5.0


@dataclass
class DeviceInfo:
    """What the recognition runtime actually ended up doing.

    Reported verbatim by ``GET /status`` — an operator needs the real answer,
    not the configured intent, because the two differ whenever a GPU was asked
    for and not obtained.
    """

    state: str = "initializing"          # initializing | ready | error
    providers: list[str] = field(default_factory=list)
    device: str | None = None            # what the detectors use
    ocr_device: str | None = None        # differs from `device` by design — see rule 8
    model_format: str | None = None
    ocr_mode: str | None = None
    requested_device: str | None = None
    fell_back: bool = False              # asked for gpu, got cpu
    warmup_ms: int | None = None
    load_ms: int | None = None
    library_version: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "providers": list(self.providers),
            "device": self.device,
            "ocr_device": self.ocr_device,
            "model_format": self.model_format,
            "ocr_mode": self.ocr_mode,
            "requested_device": self.requested_device,
            "fell_back": self.fell_back,
            "warmup_ms": self.warmup_ms,
            "load_ms": self.load_ms,
            "library_version": self.library_version,
            "error": self.error,
        }


_POOL: "queue.Queue[Any]" = queue.Queue()
_INFO = DeviceInfo()
_INFO_LOCK = threading.Lock()


def device_info() -> DeviceInfo:
    with _INFO_LOCK:
        return DeviceInfo(**_INFO.__dict__)


def _set_info(**kwargs: Any) -> None:
    with _INFO_LOCK:
        for key, value in kwargs.items():
            setattr(_INFO, key, value)


def is_ready() -> bool:
    return device_info().state == "ready"


def detect_providers() -> list[str]:
    """Execution providers onnxruntime reports.

    Informational only — see rule 7 for why this cannot decide the device.
    """
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover - depends on the install
        log.warning("[RUNTIME] onnxruntime unavailable: %s", exc)
        return []


def gpu_visible() -> bool:
    """Whether a GPU is actually reachable from this process.

    Needed because rule 7's "just try to build it and catch the exception" has a
    hole: in a container started **without** ``--gpus``, onnxruntime's CUDA
    provider does not raise — it **segfaults**, and the process dies at exit 139
    with no traceback and no chance to fall back. Verified in exactly that setup.

    So the attempt list is gated on evidence a device exists. NVML is the primary
    probe; the device nodes are checked too, because NVML's shared library is
    absent from a plain CUDA runtime image even when a GPU is passed through.
    Both are cheap and neither can crash the process.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                return True
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    # Linux/WSL2 container: the driver is passed through as device nodes.
    return any(Path(node).exists() for node in
               ("/dev/nvidiactl", "/dev/dxg", "/dev/nvidia0"))


def log_environment() -> None:
    """Log everything needed to explain a device decision, before making it.

    Written for containers. Locally you can poke at the interpreter when GPU
    silently becomes CPU; in a container all you get is the log, and the useful
    facts are exactly the ones nothing else prints: whether the installed wheel
    is the GPU build at all, whether cuDNN is actually present (the difference
    between the `base` and `cudnn-runtime` CUDA images), and whether a device was
    passed through to the container.

    Diagnostics only — every probe is individually guarded, because a service
    that cannot report its environment must still start and recognise documents.
    """
    log.info("[ENV] python=%s platform=%s", sys.version.split()[0], sys.platform)

    try:
        import onnxruntime as ort
        # The distribution name is the honest answer to "is this the GPU build":
        # both wheels import as `onnxruntime`, and the CPU one still lists no
        # CUDA provider — which reads identically to a GPU build with broken CUDA.
        try:
            from importlib.metadata import distributions
            names = sorted(d.metadata["Name"] for d in distributions()
                           if (d.metadata["Name"] or "").startswith("onnxruntime"))
        except Exception:
            names = []
        log.info("[ENV] onnxruntime=%s wheel=%s device=%s providers=%s",
                 ort.__version__, ",".join(names) or "unknown",
                 ort.get_device(), ort.get_available_providers())
    except Exception as exc:
        log.error("[ENV] onnxruntime import failed: %s", exc)

    # cuDNN, checked by loader rather than by filename: onnxruntime 1.21 needs
    # cuDNN 9, and its absence is the single most common reason a GPU image falls
    # back to CPU with no error of its own (the `base` vs `cudnn-runtime` choice
    # of CUDA image). Linux only — on Windows these come from `torch/lib` via the
    # library's own DLL shim, under different names, so probing SO names there
    # would report three failures that mean nothing.
    if sys.platform.startswith("linux"):
        for lib in ("libcudnn.so.9", "libcublas.so.12", "libcudart.so.12"):
            try:
                ctypes.CDLL(lib)
                log.info("[ENV] %s loadable", lib)
            except OSError as exc:
                log.warning("[ENV] %s NOT loadable: %s", lib, exc)

    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            name = name.decode() if isinstance(name, bytes) else name  # differs by version
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            log.info("[ENV] gpu%d=%s vram=%.1fGB", index, name, memory.total / 1e9)
        pynvml.nvmlShutdown()
        if not count:
            log.warning("[ENV] NVML reports no GPU — was the container started "
                        "with --gpus all?")
    except Exception as exc:
        log.warning("[ENV] no GPU visible via NVML (%s) — CPU mode expected", exc)

    for key in ("COMPUTE_DEVICE", "OCR_MODE", "NVIDIA_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        if os.environ.get(key):
            log.info("[ENV] %s=%s", key, os.environ[key])


@contextlib.contextmanager
def _captured_stdout(*, on_error: bool = False) -> Iterator[io.StringIO]:
    """Keep the library's ``print`` output out of the JSON log stream (rule 6).

    The library reports its own troubles by printing, so when the wrapped call
    raises, that text is the only description of what went wrong — and it gets
    promoted to WARNING. On the happy path it stays at DEBUG: with ``on_error``
    keyed off the flag alone, twelve routine "Loading model X!" lines were logged
    as warnings at every successful start.
    """
    buf = io.StringIO()
    failed = False
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    except BaseException:
        failed = True
        raise
    finally:
        text = buf.getvalue().strip()
        if text:
            flat = text.replace("\n", " | ")
            log.warning("[RUNTIME] library stdout: %s", flat) if (on_error and failed) \
                else log.debug("[RUNTIME] library stdout: %s", flat)


def _warm(pipeline: Any, sample: Path) -> int:
    """Run one real document through the pipeline; return elapsed milliseconds.

    Deliberately not ``Pipeline.warmup()`` — see rules 4 and 5. Exceptions
    propagate so ``init_runtime`` can fall back to CPU or record an error.
    """
    started = time.perf_counter()
    with _captured_stdout():
        pipeline.process_img(str(sample))
    return int((time.perf_counter() - started) * 1000)


def _find_warmup_sample(configured: str | None) -> Path | None:
    """A real, non-personal document to warm the full graph with.

    Only anonymised repository samples are eligible. Uploaded documents and any
    local personal files must never be used for warmup — they would be read at
    every service start, which is not something a document the user handed over
    for a single recognition has consented to.
    """
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return candidate
        log.warning("[RUNTIME] configured warmup image not found: %s", candidate)

    samples = Path(__file__).resolve().parents[2] / "samples"
    for sub in ("INTPASSPORT_2011", "DL_2011", "SNILS_1996"):
        found = next(iter(sorted((samples / sub).glob("*.jpg"))), None)
        if found:
            return found
    return next(iter(sorted(samples.glob("*/*.jpg"))), None)


def init_runtime(*, compute_device: str = "auto", model_format: str = "ONNX",
                 ocr_mode: str = "accurate", warmup_image: str | None = None,
                 pool_size: int = 1) -> DeviceInfo:
    """Build the pipeline(s), warm them, and publish what actually happened.

    Blocking and slow (~15 s to load, plus a warmup document per instance) — the
    caller should run it off the event loop. Never raises: a failure is recorded
    as ``state='error'`` so the service can serve its status page and explain
    itself rather than refusing to start.
    """
    from document_processing import Pipeline, __version__ as lib_version

    log_environment()
    providers = detect_providers()
    _set_info(providers=providers, model_format=model_format, ocr_mode=ocr_mode,
              requested_device=compute_device, library_version=lib_version)

    # Two independent conditions, and BOTH are required. The provider list says
    # the GPU build is installed; `gpu_visible()` says a device was actually
    # passed through. With the first true and the second false, constructing a
    # CUDA pipeline segfaults the process instead of raising — see gpu_visible().
    has_provider = "CUDAExecutionProvider" in providers
    has_device = gpu_visible()

    if compute_device == "auto":
        wanted = "gpu" if (has_provider and has_device) else "cpu"
        if has_provider and not has_device:
            log.warning("[RUNTIME] CUDAExecutionProvider is installed but no GPU is "
                        "visible — choosing CPU. In Docker, pass --gpus all.")
    else:
        wanted = compute_device
        if wanted == "gpu" and not has_device:
            log.error("[RUNTIME] compute_device=gpu but no GPU is visible to this "
                      "process — refusing to build a CUDA pipeline, which would "
                      "segfault rather than fail cleanly. Using CPU. In Docker, "
                      "pass --gpus all.")
            wanted = "cpu"
        elif wanted == "gpu" and not has_provider:
            log.error("[RUNTIME] compute_device=gpu but CUDAExecutionProvider is not "
                      "available (providers=%s) — will try anyway, then fall back",
                      providers)

    attempts = ["gpu", "cpu"] if wanted == "gpu" else ["cpu"]
    sample = _find_warmup_sample(warmup_image)
    if sample is None:
        log.warning("[RUNTIME] no warmup sample found; first real document will pay "
                    "the cold-start cost")

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            log.info("[RUNTIME] building pipeline (device=%s, format=%s, ocr=%s)",
                     attempt, model_format, ocr_mode)
            load_started = time.perf_counter()
            built = []
            for _ in range(max(1, pool_size)):
                # on_error: the library prints its own diagnostics, and if this
                # constructor throws that text is the only account of why.
                with _captured_stdout(on_error=True):
                    built.append(Pipeline(model_format=model_format, device=attempt,
                                          ocr=ocr_mode, verbose=False, ocr_gpu_batch=False))
            load_ms = int((time.perf_counter() - load_started) * 1000)
            log.info("[RUNTIME] %d instance(s) constructed on %s in %dms; warming up",
                     len(built), attempt, load_ms)

            warmup_ms = None
            if sample is not None:
                warmup_ms = sum(_warm(p, sample) for p in built) // len(built)

            for p in built:
                _POOL.put(p)

            first = built[0]
            _set_info(state="ready", device=first.device, ocr_device=first.ocr_device,
                      fell_back=(wanted == "gpu" and attempt == "cpu"),
                      load_ms=load_ms, warmup_ms=warmup_ms, error=None)
            log.info("[RUNTIME] ready: device=%s ocr_device=%s load=%dms warmup=%sms "
                     "instances=%d", first.device, first.ocr_device, load_ms,
                     warmup_ms, len(built))
            if wanted == "gpu" and attempt == "cpu":
                log.error("[RUNTIME] GPU was requested but only CPU worked — "
                          "check CUDA/cuDNN. Recognition will be slower.")
            return device_info()

        except Exception as exc:
            last_error = exc
            remaining = attempts[attempts.index(attempt) + 1:]
            log.exception("[RUNTIME] pipeline init FAILED on device=%s (%s: %s); %s",
                          attempt, type(exc).__name__, exc,
                          f"falling back to {remaining[0]}" if remaining
                          else "no fallback left")

    log.error("[RUNTIME] recognition unavailable after trying %s. The service will "
              "start and accept uploads, but every document will fail — see the "
              "[ENV] lines above for the environment it saw.", attempts)
    _set_info(state="error", error=f"{type(last_error).__name__}: {last_error}")
    return device_info()


def shutdown_runtime() -> None:
    """Drop the pipelines. Called from the lifespan's shutdown path."""
    drained = 0
    while True:
        try:
            _POOL.get_nowait()
            drained += 1
        except queue.Empty:
            break
    _set_info(state="initializing", device=None, ocr_device=None)
    log.info("[RUNTIME] released %d pipeline instance(s)", drained)


@contextlib.contextmanager
def lease_pipeline(timeout: float = LEASE_TIMEOUT_SEC) -> Iterator[Any]:
    """Exclusive use of one ``Pipeline``. **Never call ``process_img`` outside this.**

    Raises ``RuntimeNotReady`` before the models finish loading, and
    ``PipelineBusy`` if none becomes free in ``timeout`` seconds — both
    transient, so the caller requeues instead of failing the job.
    """
    info = device_info()
    if info.state == "error":
        raise RuntimeNotReady(f"recognition runtime failed to start: {info.error}")
    if info.state != "ready":
        raise RuntimeNotReady("recognition runtime is still loading models")

    try:
        pipeline = _POOL.get(timeout=timeout)
    except queue.Empty:
        raise PipelineBusy(f"no pipeline became available within {timeout}s") from None

    try:
        yield pipeline
    finally:
        _POOL.put(pipeline)


def recognise(image_path: str | Path, *, include_debug: bool = False,
              docconf: float = 0.5, img_size: int = 1500,
              lease_timeout: float = LEASE_TIMEOUT_SEC) -> tuple[dict[str, Any], Any]:
    """Recognise one document. The whole public surface of this package.

    Returns ``(viewmodel, canvas_rgb)`` — a JSON-safe dict plus the corrected
    canvas image as an RGB numpy array. The caller persists the array; note it
    is **RGB**, and OpenCV writes BGR, so it needs converting before ``imwrite``
    (the artifact layer does this).

    Blocking and CPU-bound: call it from a worker thread, not the event loop.
    """
    with lease_pipeline(timeout=lease_timeout) as pipeline:
        with _captured_stdout():
            results = pipeline.process_img(
                str(image_path),
                ocr=True, get_doc_borders=True, find_text_fields=True,
                check_quality=True, low_quality=True,
                docconf=docconf, img_size=img_size,
            )
        # Inside the lease, deliberately — see rule 3.
        payload = transform.build_viewmodel(results, device=pipeline.device,
                                            include_debug=include_debug)
        try:
            canvas = results.img_with_fixed_perspective
        except (KeyError, AttributeError):
            canvas = None
        return payload, canvas
