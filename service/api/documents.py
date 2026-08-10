"""The document resource: upload, browse, inspect, re-run, delete.

Serialisation is hand-written ``_row``/``_detail`` functions rather than
Pydantic response models — the reference project's convention, and it keeps the
wire format visible in one place instead of spread across model definitions.
Pydantic is still used for request bodies, where validation earns its keep.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import (APIRouter, Depends, File, HTTPException, Query, Response,
                     UploadFile, status)
from fastapi.responses import FileResponse

from service import worker
from service.api.deps import require_api_or_session, require_session
from service.core.config import get_settings
from service.core.database import DbSession, get_db
from service.core.models import VALID_STATUSES, iso
from service.repositories import artifacts
from service.repositories import documents as repo

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILENAME_LEN = 200


def _safe_filename(raw: str | None) -> str:
    """Keep a display name only — it never touches the filesystem.

    Stored artifacts always use a fixed name (``original.jpg``), so even a
    hostile filename cannot escape the document directory. This is purely so
    the UI shows something sensible and bounded.
    """
    name = (raw or "upload").replace("\\", "/").split("/")[-1].strip()
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '<>:"|?*')
    return (name or "upload")[:MAX_FILENAME_LEN]


def _row(record) -> dict[str, Any]:
    """One line of the document log."""
    return {
        "id": record.id,
        "filename": record.filename,
        "size_bytes": record.size_bytes,
        "status": record.status,
        "doc_type": record.doc_type,
        "doc_type_base": (record.doc_type or "").rsplit("_", 1)[0] or None
        if record.doc_type else None,
        "doc_type_era": (record.doc_type or "").rsplit("_", 1)[1]
        if record.doc_type and "_" in record.doc_type else None,
        "recognised": record.recognised,
        "doc_conf": record.doc_conf,
        "quality": record.quality or {},
        "field_count": record.field_count,
        "device": record.device,
        "processing_ms": record.processing_ms,
        "error": record.error,
        "error_code": record.error_code,
        "retry_count": record.retry_count,
        "has_canvas": record.has_canvas,
        "created_at": iso(record.created_at),
        "started_at": iso(record.started_at),
        "finished_at": iso(record.finished_at),
    }


def _detail(record) -> dict[str, Any]:
    """The row plus the full recognition view model, flattened into it.

    The stored ``result`` already has the client-facing shape (boxes, fields,
    canvas dimensions, coordinate-space notes) — see ``service/ml/transform.py``.
    """
    payload = _row(record)
    result = record.result or {}
    payload.update({
        "canvas": {
            **(result.get("canvas") or {}),
            "url": f"/api/v1/documents/{record.id}/image/canvas",
        },
        "original": {
            "url": f"/api/v1/documents/{record.id}/image/original",
            "width": record.original_w,
            "height": record.original_h,
            "content_type": record.content_type,
        },
        "coord_space": result.get("coord_space"),
        "coord_space_note": result.get("coord_space_note"),
        "boxes": result.get("boxes") or [],
        "fields": result.get("fields") or [],
        "ocr": result.get("ocr") or {},
        "quality": result.get("quality") or {},
        "timings": result.get("timings") or {},
        "address": result.get("address"),
    })
    return payload


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    db: DbSession = Depends(get_db),
    _identity=Depends(require_api_or_session),
) -> dict[str, Any]:
    """Accept one image and queue it.

    Everything that can be checked cheaply is checked here, so a bad upload
    fails immediately with an actionable message instead of becoming a
    mysterious failed job a minute later.
    """
    settings = get_settings()
    limit = settings.max_upload_mb * 1024 * 1024

    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File exceeds the {settings.max_upload_mb} MB limit")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Empty upload")

    if artifacts.is_pdf(data):
        # Called out separately because people will try it, and "unsupported
        # image type" would not tell them what to do about it.
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="PDF is not supported — upload a JPEG, PNG, WEBP, BMP or TIFF image")

    sniffed = artifacts.sniff_image(data)
    if sniffed is None:
        # Sniffed from magic bytes, not the client's Content-Type, which is
        # attacker-controlled and wrong often enough to be useless.
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Unsupported file type — expected an image")
    ext, media_type = sniffed

    dimensions = artifacts.decode_dimensions(data)
    if dimensions is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="The image could not be decoded — it may be corrupt")

    filename = _safe_filename(file.filename)

    # Bytes first, row second. The record is what makes the document visible to
    # the worker, so writing it before the file leaves a window in which the
    # drain loop can claim a document whose original does not exist yet and mark
    # a perfectly good upload as failed. See repo.reserve_id.
    doc_id = repo.reserve_id(db)
    artifacts.save_original(db, doc_id, data, ext)
    record = repo.create(db, doc_id=doc_id, filename=filename,
                         content_type=media_type,
                         size_bytes=len(data), original_ext=ext,
                         original_w=dimensions[0], original_h=dimensions[1],
                         search_text=filename.lower())

    worker.notify_new_work()
    log.info("[API] queued doc=%s (%s, %d bytes)", record.id, filename, len(data))

    row = _row(record)
    row["queue_position"] = repo.queue_position(db, record.id)
    return row


@router.get("")
def list_documents(
    status_filter: str | None = Query(None, alias="status"),
    doc_type: str | None = Query(None),
    search: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at"),
    sort_dir: str = Query("desc"),
    db: DbSession = Depends(get_db),
    _identity=Depends(require_api_or_session),
) -> dict[str, Any]:
    if status_filter and status_filter not in VALID_STATUSES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid status")
    if sort_dir not in ("asc", "desc"):
        sort_dir = "desc"

    rows, total = repo.get_all(db, status=status_filter, doc_type=doc_type,
                               search=search, date_from=date_from, date_to=date_to,
                               page=page, page_size=page_size,
                               sort_by=sort_by, sort_dir=sort_dir)
    return {
        "items": [_row(r) for r in rows],
        "total": total, "page": page, "page_size": page_size,
        "stats": repo.stats(db),
    }


@router.get("/{doc_id}")
def get_document(doc_id: int, db: DbSession = Depends(get_db),
                 _identity=Depends(require_api_or_session)) -> dict[str, Any]:
    record = repo.get_by_id(db, doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return _detail(record)


@router.get("/{doc_id}/progress")
def get_progress(doc_id: int, db: DbSession = Depends(get_db),
                 _identity=Depends(require_api_or_session)) -> dict[str, Any] | None:
    """Live progress, queue position, or a terminal state.

    Returns ``200`` with a JSON ``null`` body when there is nothing to report —
    not 404, which would make the polling client raise an error toast every two
    seconds.
    """
    record = repo.get_by_id(db, doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    live = worker.get_document_progress(doc_id)
    if live is not None:
        return live

    if record.status == "queued":
        position = repo.queue_position(db, doc_id) or 0
        return {
            "step": "queued",
            "label": f"Queued (#{position + 1})",
            "pct": 0,
            "eta_sec": round(position * worker.average_duration_sec(), 1),
            "queue_position": position,
        }
    if record.status in ("done", "failed"):
        return {"step": record.status, "label": record.status.capitalize(),
                "pct": 100 if record.status == "done" else 0,
                "eta_sec": None, "queue_position": None}
    return None


@router.get("/{doc_id}/image/{kind}")
def get_image(doc_id: int, kind: str, db: DbSession = Depends(get_db),
              _identity=Depends(require_api_or_session)) -> Response:
    if kind not in ("original", "canvas", "thumb"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown image kind")
    found = artifacts.open_artifact(db, doc_id, kind)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image not available")
    path, media_type = found
    # `no-cache` means "revalidate", not "do not store": FileResponse still sends
    # an ETag and Last-Modified, so a repeat request costs a 304 with no body.
    # `max-age=3600` was wrong here — Reprocess overwrites canvas.png and thumb.jpg
    # at the *same* URL, so the browser kept showing the previous recognition's
    # image for up to an hour while the field table beside it was already new.
    return FileResponse(path, media_type=media_type,
                        headers={"Cache-Control": "private, no-cache"})


@router.post("/{doc_id}/reprocess")
def reprocess_document(doc_id: int, db: DbSession = Depends(get_db),
                       _identity=Depends(require_api_or_session)) -> dict[str, Any]:
    record = repo.get_by_id(db, doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    if record.status in ("queued", "processing"):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail=f"Document is already {record.status}")
    record = repo.requeue(db, record)
    worker.notify_new_work()
    return _row(record)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: int, db: DbSession = Depends(get_db),
                    _identity=Depends(require_api_or_session)) -> Response:
    record = repo.get_by_id(db, doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    repo.delete(db, record)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/purge", status_code=status.HTTP_200_OK)
def purge_documents(db: DbSession = Depends(get_db),
                    _user=Depends(require_session)) -> dict[str, int]:
    """Clear the scratch store. Session-only — not something an integration does."""
    removed = 0
    for record in db.all_records():
        if record.status != "processing":  # leave an in-flight job alone
            repo.delete(db, record)
            removed += 1
    log.info("[API] purged %d document(s)", removed)
    return {"deleted": removed}
