"""Document records: query, create, mutate.

Thin by design. Every function takes the store as ``db`` and delegates the
actual query to it, because the two backends must express the same question
differently — Python filtering over JSON files versus real SQL. What lives here
is the part that is genuinely shared: validation, timestamp rules, and the
denormalisation performed when a result is saved.

Mutating functions return a **new** record; callers rebind
(``rec = repo.update(db, rec, status="done")``). Both backends hand out copies,
so mutating the returned object never touches storage on its own.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from service.core.models import VALID_STATUSES, DocumentRecord, utcnow
from service.core.store import SORT_COLUMNS

__all__ = [
    "SORT_COLUMNS", "ACTIVE_STATUSES", "get_all", "get_by_id", "create", "update",
    "update_status", "save_result", "requeue", "delete", "next_queued",
    "queue_position", "reset_stale_processing", "count_by_status", "stats",
]

ACTIVE_STATUSES = frozenset({"queued", "processing"})


def get_all(db, *, status: str | None = None, doc_type: str | None = None,
            search: str | None = None, date_from: str | None = None,
            date_to: str | None = None, page: int = 1, page_size: int = 20,
            sort_by: str = "created_at",
            sort_dir: str = "desc") -> tuple[list[DocumentRecord], int]:
    return db.query_documents(status=status, doc_type=doc_type, search=search,
                              date_from=date_from, date_to=date_to, page=page,
                              page_size=page_size, sort_by=sort_by, sort_dir=sort_dir)


def get_by_id(db, doc_id: int) -> DocumentRecord | None:
    """Full record, including the recognition result."""
    return db.get_record(doc_id)


def reserve_id(db) -> int:
    """Claim an id without inserting a row yet.

    Exists so a caller can write the upload's bytes *before* the document becomes
    visible to the worker. Inserting first looks harmless but is a real race: the
    row lands in ``queued``, the drain loop polls on its own timer, and if it
    claims the document in the window before the file is written the job fails
    with "has no stored original" — a good upload reported as a failed document.
    """
    return db.next_document_id()


def create(db, *, doc_id: int | None = None, **fields: Any) -> DocumentRecord:
    """Insert a record. Pass ``doc_id`` from :func:`reserve_id` when the
    artifacts were written first (see the note there)."""
    return db.put_record(
        DocumentRecord(id=doc_id if doc_id is not None else db.next_document_id(),
                       **fields))


def update(db, record: DocumentRecord, **fields: Any) -> DocumentRecord:
    updated = dataclasses.replace(record, **fields, updated_at=utcnow())
    # `result` is stored separately; carry it so a plain field update does not
    # look like a request to clear it.
    updated.result = record.result
    return db.put_record(updated)


def update_status(db, record: DocumentRecord, status: str, *,
                  error: str | None = None,
                  error_code: str | None = None) -> DocumentRecord:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}")
    extra: dict[str, Any] = {"status": status, "error": error, "error_code": error_code}
    if status == "processing":
        extra["started_at"] = utcnow()
    elif status in ("done", "failed"):
        extra["finished_at"] = utcnow()
    return update(db, record, **extra)


def save_result(db, record: DocumentRecord, payload: dict[str, Any], *,
                search_text: str, processing_ms: int) -> DocumentRecord:
    """Store the view model and denormalise the columns the list page needs.

    The denormalisation is the point: without it, filtering or sorting the log
    would mean opening every result blob.
    """
    db.save_result_payload(record.id, payload)
    quality = payload.get("quality") or {}
    canvas = payload.get("canvas") or {}
    return update(
        db, record,
        status="done", error=None, error_code=None,
        doc_type=payload.get("doc_type"),
        doc_conf=quality.get("DocConf"),
        quality={k: v for k, v in quality.items() if k != "DocConf"},
        recognised=bool(payload.get("recognised")),
        field_count=len(payload.get("fields") or []),
        device=payload.get("device"),
        processing_ms=processing_ms,
        canvas_w=canvas.get("width"), canvas_h=canvas.get("height"),
        has_canvas=bool(canvas.get("width")),
        search_text=search_text,
        finished_at=utcnow(),
    )


def requeue(db, record: DocumentRecord) -> DocumentRecord:
    return update(db, record, status="queued", retry_count=0, error=None,
                  error_code=None, started_at=None, finished_at=None)


def delete(db, record: DocumentRecord) -> None:
    db.drop_record(record.id)


def next_queued(db) -> int | None:
    return db.next_queued_id()


def queue_position(db, doc_id: int) -> int | None:
    return db.queue_position(doc_id)


def reset_stale_processing(db) -> int:
    """Recover jobs interrupted mid-flight by a restart.

    Without this a document caught in ``processing`` when the process died would
    sit there forever: the drain loop only ever claims ``queued`` rows.
    """
    count = 0
    for record in db.all_records():
        if record.status == "processing":
            update(db, record, status="queued", started_at=None)
            count += 1
    return count


def count_by_status(db) -> dict[str, int]:
    return db.count_by_status()


def stats(db) -> dict[str, Any]:
    return db.aggregate_stats()
