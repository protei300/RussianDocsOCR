"""Recognition in memory: one image in, one answer out, nothing kept.

``POST /recognize`` reads the image into a buffer, decodes it, recognises it inside
the request, answers, and wipes what it owns. No record, no file, no artifact, no
queue; one log line and one journal line with the outcome and the time — nothing
about the document. Available in both processing modes; in ``PROCESSING_MODE=memory``
it is the only way a document gets into the service.

What "nothing kept" takes, because each item below was a real way to keep something:

* **The body is never spooled to disk.** FastAPI's ``UploadFile`` is a
  ``SpooledTemporaryFile`` that moves to a temporary file on disk past 1 MB, so a
  normal passport photo would touch the disk before any code of ours ran. This
  route reads the raw request stream into a ``bytearray`` itself: either the image
  as the body (``Content-Type: image/jpeg`` …), or a ``multipart/form-data`` body
  that is split in memory. ``UploadFile`` is not used here at all.
* **Buffers we own are zeroed**, not just dropped: the request body, the image
  bytes, the decoded pixels, the canvas PNG. The pipeline's own copies are zeroed
  inside the lease (runtime rule 11).
* **Strings cannot be wiped.** The recognised text and the JSON answer are
  immutable strings in Python (and in Go, .NET and the JVM); they are released and
  the memory is reused, but not overwritten. The documentation says so rather
  than promising more.
* **Nothing about the document reaches a log**: not the filename, not the type,
  not the size, not a library message, not an exception text (only its type, and
  only the outcome is logged). The journal line is outcome + status + milliseconds.
* **The answer is not cacheable**: ``Cache-Control: no-store``.
"""
from __future__ import annotations

import asyncio
import base64
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from service.api.deps import require_api_or_role
from service.core.config import get_settings
from service.core.database import DbSession, get_db
from service.ml import runtime
from service.ml.errors import PipelineBusy, RuntimeNotReady
from service.repositories import artifacts
from service.repositories import recognitions
from service.repositories import settings as settings_repo

log = logging.getLogger(__name__)
router = APIRouter(tags=["recognize"])

require_api_or_operator = require_api_or_role("operator")
require_api_or_viewer = require_api_or_role("viewer")

_EXECUTOR: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    """Separate from the background worker's: an in-memory request must not queue
    behind a stored document's thread. The pipeline pool is still shared and the
    lease is what serialises access to it."""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=max(1, get_settings().pipeline_pool_size),
                                       thread_name_prefix="recognize-mem")
    return _EXECUTOR


def _wipe(buf: bytearray | None) -> None:
    if buf is not None:
        buf[:] = b"\0" * len(buf)


class _Refused(Exception):
    def __init__(self, status_code: int, detail: str, outcome: str = "rejected",
                 headers: dict[str, str] | None = None):
        super().__init__(detail)
        self.status_code, self.detail, self.outcome, self.headers = status_code, detail, outcome, headers


async def _read_body(request: Request, limit: int) -> bytearray:
    """The whole request body into one bytearray, refusing more than ``limit`` bytes."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + 64 * 1024:
        raise _Refused(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image exceeds the upload limit")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit + 64 * 1024:          # multipart framing is a few hundred bytes
            _wipe(body)
            raise _Refused(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image exceeds the upload limit")
    return body


def _boundary(content_type: str) -> bytes | None:
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "boundary" and value:
            return value.strip('"').encode("latin-1")
    return None


def _image_from_multipart(body: bytearray, boundary: bytes) -> bytearray:
    """The first file part of a multipart body, copied into its own bytearray.

    Deliberately small: one part with a filename, or a part named ``file``. The
    body buffer is wiped by the caller afterwards; this copy is wiped later too.
    """
    delimiter = b"--" + boundary
    pos = body.find(delimiter)
    while pos != -1:
        start = pos + len(delimiter)
        if body[start:start + 2] == b"--":
            break
        header_end = body.find(b"\r\n\r\n", start)
        if header_end == -1:
            break
        headers = bytes(body[start:header_end]).decode("latin-1").lower()
        data_start = header_end + 4
        nxt = body.find(b"\r\n" + delimiter, data_start)
        if nxt == -1:
            break
        if "filename=" in headers or 'name="file"' in headers:
            return bytearray(body[data_start:nxt])
        pos = nxt + 2
    raise _Refused(status.HTTP_400_BAD_REQUEST,
                   "No image in the multipart body — send it as the 'file' part")


def _recognise_sync(image: bytearray, docconf: float, img_size: int,
                    with_canvas: bool) -> tuple[dict[str, Any], bytearray | None]:
    """Decode, recognise, wipe. Runs on the executor; every pixel buffer it creates
    is zeroed here, in ``finally``, even when recognition fails or times out."""
    import cv2
    import numpy as np

    bgr = rgb = None
    try:
        bgr = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise _Refused(status.HTTP_422_UNPROCESSABLE_ENTITY,
                           "The image could not be decoded — it may be corrupt")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr.fill(0)
        return runtime.recognise_image(rgb, docconf=docconf, img_size=img_size,
                                       with_canvas=with_canvas,
                                       lease_timeout=float(get_settings().recognize_wait_sec))
    finally:
        for arr in (bgr, rgb):
            if arr is not None:
                arr.fill(0)
        _wipe(image)


def _response_body(payload: dict[str, Any], png: bytearray | None, ms: int) -> dict[str, Any]:
    canvas = dict(payload.get("canvas") or {})
    if png is not None:
        canvas["image_png_base64"] = base64.b64encode(png).decode("ascii")
    return {
        "stored": False,
        "doc_type": payload.get("doc_type"),
        "doc_type_base": payload.get("doc_type_base"),
        "doc_type_era": payload.get("doc_type_era"),
        "recognised": payload.get("recognised"),
        "doc_conf": (payload.get("quality") or {}).get("DocConf"),
        "device": payload.get("device"),
        "canvas": canvas,
        "coord_space": payload.get("coord_space"),
        "coord_space_note": payload.get("coord_space_note"),
        "boxes": payload.get("boxes") or [],
        "fields": payload.get("fields") or [],
        "ocr": payload.get("ocr") or {},
        # Same split as a stored document's row: DocConf on its own, the rest as quality.
        "quality": {k: v for k, v in (payload.get("quality") or {}).items() if k != "DocConf"},
        "address": payload.get("address"),
        "timings": payload.get("timings") or {},
        "processing_ms": ms,
    }


def _finish(outcome: str, status_code: int, started: float) -> int:
    ms = int((time.perf_counter() - started) * 1000)
    recognitions.record(outcome, status_code, ms)
    # The one log line of an in-memory recognition: outcome, status, time. Nothing else.
    log.info("[API] recognize: outcome=%s status=%d ms=%d", outcome, status_code, ms)
    return ms


@router.post("/recognize")
async def recognize(
    request: Request,
    images: bool = Query(False, description="also return the rectified document as base64 PNG"),
    db: DbSession = Depends(get_db),
    _identity=Depends(require_api_or_operator),
) -> JSONResponse:
    """Recognise one image in memory and answer. Nothing is stored."""
    started = time.perf_counter()
    settings = get_settings()
    limit = settings.max_upload_mb * 1024 * 1024
    body = image = png = None
    try:
        body = await _read_body(request, limit)
        content_type = request.headers.get("content-type", "")
        if content_type.lower().startswith("multipart/form-data"):
            boundary = _boundary(content_type)
            if not boundary:
                raise _Refused(status.HTTP_400_BAD_REQUEST, "Malformed multipart body")
            image = _image_from_multipart(body, boundary)
            _wipe(body)
        else:
            image, body = body, None
        if not image:
            raise _Refused(status.HTTP_400_BAD_REQUEST, "Empty upload")
        if len(image) > limit:
            raise _Refused(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                           f"File exceeds the {settings.max_upload_mb} MB limit")
        if artifacts.is_pdf(image):
            raise _Refused(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                           "PDF is not supported — send a JPEG, PNG, WEBP, BMP or TIFF image")
        if artifacts.sniff_image(image) is None:
            raise _Refused(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                           "Unsupported file type — expected an image")

        docconf = float(settings_repo.get_value(db, "docconf", 0.5))
        img_size = int(settings_repo.get_value(db, "img_size", 1500))
        timeout = float(settings_repo.get_value(db, "job_timeout_sec", 120))
        loop = asyncio.get_running_loop()
        try:
            payload, png = await asyncio.wait_for(
                loop.run_in_executor(_executor(), _recognise_sync, image, docconf, img_size, images),
                timeout=timeout + settings.recognize_wait_sec)
        except PipelineBusy:
            raise _Refused(status.HTTP_503_SERVICE_UNAVAILABLE,
                           "All recognition pipelines are busy — retry shortly", "busy",
                           {"Retry-After": "5"}) from None
        except RuntimeNotReady:
            raise _Refused(status.HTTP_503_SERVICE_UNAVAILABLE,
                           "The recognition runtime is not ready yet", "busy",
                           {"Retry-After": "10"}) from None
        except asyncio.TimeoutError:
            raise _Refused(status.HTTP_504_GATEWAY_TIMEOUT,
                           "Recognition took too long", "error") from None
        image = None                                # wiped by _recognise_sync

        ms = int((time.perf_counter() - started) * 1000)
        content = _response_body(payload, png, ms)
        outcome = "recognised" if payload.get("recognised") else "not_recognised"
        _finish(outcome, 200, started)
        return JSONResponse(content, headers={"Cache-Control": "no-store"})
    except _Refused as refused:
        _finish(refused.outcome, refused.status_code, started)
        raise HTTPException(refused.status_code, detail=refused.detail,
                            headers=refused.headers) from None
    except HTTPException:
        raise
    except Exception as error:                       # noqa: BLE001 — logged by type only
        _finish("error", 500, started)
        log.warning("[API] recognize failed: %s", type(error).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Recognition failed") from None
    finally:
        _wipe(body)
        _wipe(image)
        _wipe(png)
        payload = None
        gc.collect()


@router.get("/recognitions")
def list_recognitions(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                      _identity=Depends(require_api_or_viewer)) -> dict[str, Any]:
    """The journal of in-memory recognitions: when, how it ended, how long. Nothing
    else exists to show — these lines cannot be opened."""
    page, total = recognitions.recent(limit=limit, offset=offset)
    return {"items": [e.public() for e in page], "total": total,
            "counts": recognitions.counts()}
