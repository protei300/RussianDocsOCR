"""Background recognition worker.

One asyncio drain loop pulls queued documents and runs them through the
pipeline. Deliberate choices, with the reasoning attached because several look
arbitrary until something breaks:

* **Event-driven, not fixed-interval polling.** Recognition takes ~0.4 s; a
  10-second poll would dominate the latency a user perceives. Uploads set
  ``_WAKE``, and a 2-second timeout is only a safety net for anything that
  enqueues without signalling.

* **A dedicated executor, not the default one.** ``run_in_executor(None, ...)``
  shares asyncio's default pool, which sizes itself to ``min(32, cpu+4)``.
  Twenty threads all racing for one pipeline lease is not useful; a bounded
  pool sized to the pipeline pool enforces the invariant at a second layer.

* **``asyncio.wait_for`` cannot kill the executor thread.** This is the sharpest
  edge here. On timeout the coroutine is cancelled but the thread keeps running
  ``process_img`` and keeps holding its lease. We mark the job failed and move
  on; the lease is released whenever that thread finally finishes. Subsequent
  jobs then get ``PipelineBusy`` (a bounded wait) and requeue rather than
  blocking forever. A genuinely hung ONNX call needs a process restart — the
  container's restart policy is the last line of defence.

* **Transient vs deterministic failures.** Retrying a corrupt JPEG forever is
  as wrong as giving up on a CUDA hiccup, so the two are separated explicitly
  and only transient ones consume a retry.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from service.core.config import get_settings
from service.core.database import db_session, get_store
from service.ml import runtime
from service.ml.errors import PipelineBusy, RecognitionError, RuntimeNotReady
from service.ml.transform import build_search_text
from service.repositories import artifacts
from service.repositories import documents as repo
from service.repositories import settings as settings_repo

log = logging.getLogger(__name__)

_WAKE = asyncio.Event()
_TASKS: list[asyncio.Task] = []
_EXECUTOR: ThreadPoolExecutor | None = None
_RUNTIME_READY = asyncio.Event()

#: Fallback poll interval. Normal flow is driven by ``_WAKE``; this only
#: catches anything that enqueues without signalling.
QUEUE_POLL_SEC = 2.0


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
# Two honest steps rather than five interpolated ones. The pipeline exposes no
# progress callbacks, so a finer breakdown would be pure theatre — and at
# ~0.4 s per document a five-segment animated bar is a lie the user can see
# through. `recognizing` self-calibrates from real completions instead of using
# a hardcoded constant.
_STEP_CONFIG: dict[str, dict[str, Any]] = {
    "loading": {"label": "Loading models", "pct_start": 0, "pct_end": 90, "duration": 20.0},
    "recognizing": {"label": "Recognising document", "pct_start": 5, "pct_end": 95,
                    "duration": 0.6},
}
_processing_state: dict[int, tuple[str, float]] = {}
_duration_ema = 0.6


def _set_step(doc_id: int, step: str) -> None:
    _processing_state[doc_id] = (step, time.time())


def _clear_step(doc_id: int) -> None:
    _processing_state.pop(doc_id, None)


def _record_duration(seconds: float) -> None:
    """Exponential moving average of real completions, so the ETA tracks reality."""
    global _duration_ema
    _duration_ema = 0.7 * _duration_ema + 0.3 * max(seconds, 0.05)


def average_duration_sec() -> float:
    return _duration_ema


def get_document_progress(doc_id: int) -> dict[str, Any] | None:
    """Live progress, or ``None`` when the document is not being processed."""
    state = _processing_state.get(doc_id)
    if state is None:
        return None
    step, started = state
    config = _STEP_CONFIG.get(step, _STEP_CONFIG["recognizing"])
    duration = _duration_ema if step == "recognizing" else config["duration"]
    elapsed = time.time() - started
    fraction = min(elapsed / max(duration, 0.05), 0.95)
    pct = config["pct_start"] + fraction * (config["pct_end"] - config["pct_start"])
    return {
        "step": step,
        "label": config["label"],
        "pct": round(pct, 1),
        "eta_sec": round(max(0.0, duration - elapsed), 1),
        "queue_position": None,
    }


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------
def _classify_failure(exc: BaseException) -> tuple[str, bool]:
    """``(error_code, transient)``.

    Only transient failures consume a retry — see the module docstring.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return "TIMEOUT", True
    if isinstance(exc, (PipelineBusy, RuntimeNotReady)):
        return "BUSY", True
    if isinstance(exc, RecognitionError):
        return "RECOGNITION_FAILED", getattr(exc, "transient", False)
    text = f"{type(exc).__name__}: {exc}".lower()
    if "cuda" in text or "cudnn" in text or "out of memory" in text:
        return "GPU_ERROR", True
    if isinstance(exc, OSError):
        return "IO_ERROR", True
    return "INTERNAL", False


def _process_sync(doc_id: int) -> tuple[dict[str, Any], Any, int]:
    """Runs on an executor thread. All CPU-bound work lives here.

    Returns ``(viewmodel, canvas_rgb, elapsed_ms)``. Note ``runtime.recognise``
    does the transform inside the pipeline lease — see its docstring for why
    that is not optional.
    """
    db = get_store()
    record = repo.get_by_id(db, doc_id)
    if record is None:
        raise RecognitionError(f"document {doc_id} vanished before processing")

    found = artifacts.open_artifact(db, doc_id, "original")
    if found is None:
        raise RecognitionError(f"document {doc_id} has no stored original")
    path, _ = found

    docconf = settings_repo.get_value(db, "docconf", 0.5)
    img_size = settings_repo.get_value(db, "img_size", 1500)

    started = time.perf_counter()
    payload, canvas = runtime.recognise(path, docconf=float(docconf),
                                        img_size=int(img_size))
    return payload, canvas, int((time.perf_counter() - started) * 1000)


async def _process_document(doc_id: int) -> None:
    loop = asyncio.get_running_loop()
    with db_session() as db:
        record = repo.get_by_id(db, doc_id)
        if record is None or record.status != "queued":
            return  # deleted or claimed elsewhere between the scan and now
        record = repo.update_status(db, record, "processing")
        timeout = float(settings_repo.get_value(db, "job_timeout_sec", 120))
        max_retries = int(settings_repo.get_value(db, "max_retries", 2))

    _set_step(doc_id, "recognizing")
    try:
        payload, canvas, elapsed_ms = await asyncio.wait_for(
            loop.run_in_executor(_EXECUTOR, _process_sync, doc_id), timeout=timeout,
        )
    except BaseException as exc:  # noqa: BLE001 - classified below, never swallowed
        code, transient = _classify_failure(exc)
        if isinstance(exc, asyncio.CancelledError):
            raise
        log.warning("[WORKER] doc=%s failed (%s, transient=%s): %s",
                    doc_id, code, transient, exc)
        with db_session() as db:
            record = repo.get_by_id(db, doc_id)
            if record is None:
                return
            if transient and record.retry_count < max_retries:
                repo.update(db, record, status="queued",
                            retry_count=record.retry_count + 1,
                            error=str(exc), error_code=code, started_at=None)
                _WAKE.set()
            else:
                repo.update_status(db, record, "failed", error=str(exc), error_code=code)
        return
    finally:
        _clear_step(doc_id)

    _record_duration(elapsed_ms / 1000)

    # Persist artifacts outside any lock — these are the slow writes.
    with db_session() as db:
        record = repo.get_by_id(db, doc_id)
        if record is None:
            return  # deleted while we were recognising
        if canvas is not None:
            try:
                artifacts.save_canvas(db, doc_id, canvas)
                artifacts.save_thumbnail(db, doc_id, canvas)
            except Exception:
                # A missing preview must not fail an otherwise good recognition.
                log.exception("[WORKER] doc=%s canvas write failed", doc_id)
        search_text = build_search_text(record.filename, payload)
        repo.save_result(db, record, payload, search_text=search_text,
                         processing_ms=elapsed_ms)

    log.info("[WORKER] doc=%s done in %dms type=%s fields=%d",
             doc_id, elapsed_ms, payload.get("doc_type"), len(payload.get("fields") or []))


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------
async def _drain_loop() -> None:
    log.info("[WORKER] drain loop started")
    await _RUNTIME_READY.wait()
    while True:
        try:
            with db_session() as db:
                doc_id = repo.next_queued(db)
            if doc_id is None:
                _WAKE.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(_WAKE.wait(), timeout=QUEUE_POLL_SEC)
                continue
            try:
                await _process_document(doc_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Belt and braces: _process_document already handles its own
                # failures, so reaching here means a bug in that handling.
                log.exception("[WORKER] unhandled error on doc=%s", doc_id)
        except asyncio.CancelledError:
            log.info("[WORKER] drain loop cancelled")
            raise
        except Exception:
            log.exception("[WORKER] drain loop error")
            await asyncio.sleep(5)


async def _init_runtime_bg() -> None:
    """Load the models off the event loop, then release the drain loop.

    Startup must not block on this: 215 MB of ONNX sessions plus a warmup
    document take seconds, and blocking the lifespan would delay ``/health``
    and fight Docker's healthcheck. Uploads are accepted immediately and wait
    in the queue — which is exactly what the async design is for.
    """
    settings = get_settings()
    loop = asyncio.get_running_loop()
    try:
        with db_session() as db:
            compute_device = settings_repo.get_value(db, "compute_device",
                                                     settings.compute_device)
            ocr_mode = settings_repo.get_value(db, "ocr_mode", settings.ocr_mode)
        info = await loop.run_in_executor(
            None,
            lambda: runtime.init_runtime(
                compute_device=str(compute_device), model_format=settings.model_format,
                ocr_mode=str(ocr_mode), warmup_image=settings.warmup_image or None,
                pool_size=settings.pipeline_pool_size),
        )
        if info.state != "ready":
            log.error("[WORKER] recognition runtime failed to start: %s", info.error)
    except Exception:
        log.exception("[WORKER] runtime initialisation crashed")
    finally:
        # Released either way: with a broken runtime the drain loop still needs
        # to run so queued documents fail with a clear message instead of
        # sitting in 'queued' forever with no explanation.
        _RUNTIME_READY.set()
        _WAKE.set()


def notify_new_work() -> None:
    """Called by the upload/reprocess endpoints to wake the drain loop."""
    _WAKE.set()


async def start_worker() -> None:
    global _EXECUTOR
    settings = get_settings()
    _EXECUTOR = ThreadPoolExecutor(max_workers=max(1, settings.pipeline_pool_size),
                                   thread_name_prefix="recognise")

    with db_session() as db:
        recovered = repo.reset_stale_processing(db)
    if recovered:
        log.info("[WORKER] requeued %d document(s) left mid-processing", recovered)

    _TASKS.append(asyncio.create_task(_init_runtime_bg(), name="runtime_init"))
    _TASKS.append(asyncio.create_task(_drain_loop(), name="drain"))


async def stop_worker() -> None:
    for task in _TASKS:
        task.cancel()
    if _TASKS:
        await asyncio.gather(*_TASKS, return_exceptions=True)
    _TASKS.clear()
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
    runtime.shutdown_runtime()
    log.info("[WORKER] stopped")
