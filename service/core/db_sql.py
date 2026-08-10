"""SQLAlchemy-backed store. Works on any dialect SQLAlchemy supports.

Tested against MS SQL Server and PostgreSQL. There is deliberately **no
dialect-specific SQL anywhere** in this file or in the migrations — only generic
SQLAlchemy types and operations, which the dialect layer translates (``String``
becomes ``NVARCHAR`` on MS SQL and ``VARCHAR`` on PostgreSQL, ``Text`` becomes
``NVARCHAR(MAX)`` / ``TEXT``). That is what lets one schema definition serve both.

Column names match ``DocumentRecord``/``ApiKey`` field names exactly, so
conversion between ORM rows and the dataclasses the rest of the service uses is
mechanical and there is one obvious place to look when something is missing.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (Boolean, DateTime, Float, Integer, String, Text, create_engine,
                        func, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from service.core.models import ApiKey, DocumentRecord, utcnow
from service.core.store import SORT_COLUMNS

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "rd_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), index=True)

    doc_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    doc_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    recognised: Mapped[bool] = mapped_column(Boolean, default=False)
    field_count: Mapped[int] = mapped_column(Integer, default=0)
    #: Small JSON blob (the four quality verdicts). Kept as text rather than a
    #: JSON column so the schema stays dialect-neutral.
    quality_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    device: Mapped[str | None] = mapped_column(String(16), nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    original_ext: Mapped[str] = mapped_column(String(16), default=".jpg")
    original_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    original_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    canvas_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    canvas_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    has_canvas: Mapped[bool] = mapped_column(Boolean, default=False)

    #: Lowercased haystack for free-text search: filename + doc type + every
    #: recognised value. Precomputed so a search never parses result_json.
    search_text: Mapped[str] = mapped_column(Text, default="")
    #: The full recognition view model. Deliberately one denormalised blob: it
    #: is written once, read whole, and never queried into.
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class ApiKeyRow(Base):
    __tablename__ = "rd_api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    label: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(32))
    key_hash: Mapped[str] = mapped_column(String(64), index=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SettingRow(Base):
    __tablename__ = "rd_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def _naive_utc(value: datetime | None) -> datetime | None:
    """Strip the tzinfo for storage.

    Not every dialect has a portable timezone-aware type, and mixing aware and
    naive values is a reliable source of comparison bugs. Everything is UTC by
    construction (``models.utcnow``), so store naive UTC and re-attach the zone
    on the way out.
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _row_to_record(row: DocumentRow, *, with_result: bool = False) -> DocumentRecord:
    record = DocumentRecord(
        id=row.id, filename=row.filename, content_type=row.content_type,
        size_bytes=row.size_bytes, status=row.status,
        doc_type=row.doc_type, doc_conf=row.doc_conf, recognised=bool(row.recognised),
        field_count=row.field_count,
        quality=json.loads(row.quality_json) if row.quality_json else {},
        device=row.device, processing_ms=row.processing_ms,
        error=row.error, error_code=row.error_code, retry_count=row.retry_count,
        original_ext=row.original_ext, original_w=row.original_w, original_h=row.original_h,
        canvas_w=row.canvas_w, canvas_h=row.canvas_h, has_canvas=bool(row.has_canvas),
        search_text=row.search_text or "",
        created_at=_aware_utc(row.created_at) or utcnow(),
        started_at=_aware_utc(row.started_at),
        finished_at=_aware_utc(row.finished_at),
        updated_at=_aware_utc(row.updated_at) or utcnow(),
    )
    if with_result and row.result_json:
        try:
            record.result = json.loads(row.result_json)
        except ValueError:
            log.exception("[STORE] unreadable result_json for document %s", row.id)
    return record


def _apply_record(row: DocumentRow, record: DocumentRecord) -> None:
    row.filename = record.filename
    row.content_type = record.content_type
    row.size_bytes = record.size_bytes
    row.status = record.status
    row.doc_type = record.doc_type
    row.doc_conf = record.doc_conf
    row.recognised = record.recognised
    row.field_count = record.field_count
    row.quality_json = json.dumps(record.quality, ensure_ascii=False) if record.quality else None
    row.device = record.device
    row.processing_ms = record.processing_ms
    row.error = record.error
    row.error_code = record.error_code
    row.retry_count = record.retry_count
    row.original_ext = record.original_ext
    row.original_w = record.original_w
    row.original_h = record.original_h
    row.canvas_w = record.canvas_w
    row.canvas_h = record.canvas_h
    row.has_canvas = record.has_canvas
    row.search_text = record.search_text
    row.created_at = _naive_utc(record.created_at)
    row.started_at = _naive_utc(record.started_at)
    row.finished_at = _naive_utc(record.finished_at)
    row.updated_at = _naive_utc(record.updated_at)


def _parse_day(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None  # same silent-ignore contract as the file backend


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class SqlStore:
    """Implements ``core.store.DocumentStore`` against a SQL database."""

    def __init__(self, url: str, data_dir: Path):
        self.url = url
        self.root = data_dir
        self.docs_dir = data_dir / "documents"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        # `pool_pre_ping` matters here: a corporate SQL Server will drop idle
        # connections, and without it the first request after a quiet spell
        # fails instead of transparently reconnecting.
        self._engine = create_engine(url, pool_pre_ping=True, pool_recycle=1800,
                                     future=True)
        self._session = sessionmaker(bind=self._engine, expire_on_commit=False)
        #: Guards id allocation only. Everything else relies on the database's
        #: own transactions.
        self._id_lock = threading.Lock()

    @property
    def backend(self) -> str:
        return "sql"

    @property
    def is_ephemeral(self) -> bool:
        return False

    @property
    def engine(self):
        return self._engine

    def dialect(self) -> str:
        return self._engine.dialect.name

    # -- ids -----------------------------------------------------------------
    # Explicit ids rather than an identity column, so both backends allocate
    # the same way and a record can be created before it is inserted.
    def _next_id(self, model: type, column) -> int:
        with self._id_lock, self._session() as session:
            highest = session.execute(select(func.max(column))).scalar()
            return int(highest or 0) + 1

    def next_document_id(self) -> int:
        return self._next_id(DocumentRow, DocumentRow.id)

    def next_api_key_id(self) -> int:
        return self._next_id(ApiKeyRow, ApiKeyRow.id)

    # -- documents -----------------------------------------------------------
    def get_record(self, doc_id: int) -> DocumentRecord | None:
        with self._session() as session:
            row = session.get(DocumentRow, doc_id)
            return _row_to_record(row, with_result=True) if row else None

    def put_record(self, record: DocumentRecord) -> DocumentRecord:
        with self._session() as session:
            row = session.get(DocumentRow, record.id)
            if row is None:
                row = DocumentRow(id=record.id)
                session.add(row)
            _apply_record(row, record)
            # `result` travels separately (save_result_payload); never clear an
            # already-stored blob just because this record instance lacks it.
            if record.result is not None:
                row.result_json = json.dumps(record.result, ensure_ascii=False)
            session.commit()
        return record

    def drop_record(self, doc_id: int) -> None:
        with self._session() as session:
            row = session.get(DocumentRow, doc_id)
            if row is not None:
                session.delete(row)
                session.commit()
        import shutil
        shutil.rmtree(self.doc_dir(doc_id), ignore_errors=True)

    def query_documents(self, *, status=None, doc_type=None, search=None,
                        date_from=None, date_to=None, page=1, page_size=20,
                        sort_by="created_at", sort_dir="desc"):
        column_name = sort_by if sort_by in SORT_COLUMNS else "created_at"
        column = getattr(DocumentRow, column_name)

        def constrain(stmt):
            if status:
                stmt = stmt.where(DocumentRow.status == status)
            if doc_type == "__none__":
                stmt = stmt.where(DocumentRow.recognised.is_(False))
            elif doc_type:
                stmt = stmt.where(DocumentRow.doc_type.like(f"{doc_type}%"))
            start = _parse_day(date_from)
            if start:
                stmt = stmt.where(DocumentRow.created_at >= start)
            end = _parse_day(date_to)
            if end:
                stmt = stmt.where(DocumentRow.created_at < end + timedelta(days=1))
            if search:
                stmt = stmt.where(DocumentRow.search_text.like(f"%{search.lower().strip()}%"))
            return stmt

        with self._session() as session:
            total = session.execute(
                constrain(select(func.count()).select_from(DocumentRow))).scalar() or 0
            # NULLs last in both directions: a queued document has no doc_conf
            # and must not lead an ascending sort.
            order = column.asc() if sort_dir == "asc" else column.desc()
            stmt = (constrain(select(DocumentRow))
                    .order_by(column.is_(None), order, DocumentRow.id.desc())
                    .offset(max(0, (page - 1) * page_size)).limit(page_size))
            rows = session.execute(stmt).scalars().all()
            return [_row_to_record(r) for r in rows], int(total)

    def all_records(self) -> list[DocumentRecord]:
        with self._session() as session:
            rows = session.execute(select(DocumentRow)).scalars().all()
            return [_row_to_record(r) for r in rows]

    def next_queued_id(self) -> int | None:
        with self._session() as session:
            stmt = (select(DocumentRow.id).where(DocumentRow.status == "queued")
                    .order_by(DocumentRow.created_at.asc(), DocumentRow.id.asc()).limit(1))
            return session.execute(stmt).scalar()

    def queue_position(self, doc_id: int) -> int | None:
        with self._session() as session:
            row = session.get(DocumentRow, doc_id)
            if row is None or row.status != "queued":
                return None
            ahead = session.execute(
                select(func.count()).select_from(DocumentRow)
                .where(DocumentRow.status == "queued")
                .where(DocumentRow.created_at < row.created_at)).scalar() or 0
            return int(ahead)

    def count_by_status(self) -> dict[str, int]:
        with self._session() as session:
            rows = session.execute(
                select(DocumentRow.status, func.count()).group_by(DocumentRow.status)).all()
        counts = {"queued": 0, "processing": 0, "done": 0, "failed": 0}
        counts.update({str(status): int(count) for status, count in rows})
        return counts

    def aggregate_stats(self) -> dict[str, Any]:
        with self._session() as session:
            total = session.execute(select(func.count()).select_from(DocumentRow)).scalar() or 0
            recognised = session.execute(
                select(func.count()).select_from(DocumentRow)
                .where(DocumentRow.recognised.is_(True))).scalar() or 0
            average = session.execute(
                select(func.avg(DocumentRow.processing_ms))
                .where(DocumentRow.status == "done")
                .where(DocumentRow.processing_ms.is_not(None))).scalar()
        return {
            **self.count_by_status(),
            "total": int(total),
            "recognised": int(recognised),
            "avg_processing_ms": round(float(average)) if average is not None else None,
        }

    # -- results -------------------------------------------------------------
    def save_result_payload(self, doc_id: int, payload: dict[str, Any]) -> None:
        with self._session() as session:
            row = session.get(DocumentRow, doc_id)
            if row is None:
                return
            row.result_json = json.dumps(payload, ensure_ascii=False)
            session.commit()

    def load_result_payload(self, doc_id: int) -> dict[str, Any] | None:
        with self._session() as session:
            row = session.get(DocumentRow, doc_id)
            if row is None or not row.result_json:
                return None
            try:
                return json.loads(row.result_json)
            except ValueError:
                return None

    # -- api keys ------------------------------------------------------------
    def all_api_keys(self) -> list[ApiKey]:
        with self._session() as session:
            rows = session.execute(
                select(ApiKeyRow).order_by(ApiKeyRow.created_at.asc())).scalars().all()
            return [ApiKey(id=r.id, label=r.label, prefix=r.prefix, key_hash=r.key_hash,
                           is_default=bool(r.is_default),
                           created_at=_aware_utc(r.created_at) or utcnow(),
                           last_used_at=_aware_utc(r.last_used_at)) for r in rows]

    def put_api_key(self, key: ApiKey) -> ApiKey:
        with self._session() as session:
            row = session.get(ApiKeyRow, key.id)
            if row is None:
                row = ApiKeyRow(id=key.id)
                session.add(row)
            row.label = key.label
            row.prefix = key.prefix
            row.key_hash = key.key_hash
            row.is_default = key.is_default
            row.created_at = _naive_utc(key.created_at)
            row.last_used_at = _naive_utc(key.last_used_at)
            session.commit()
        return key

    def drop_api_key(self, key_id: int) -> bool:
        with self._session() as session:
            row = session.get(ApiKeyRow, key_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    # -- settings ------------------------------------------------------------
    def all_settings(self) -> dict[str, str]:
        with self._session() as session:
            rows = session.execute(select(SettingRow)).scalars().all()
            return {r.key: r.value for r in rows}

    def set_settings(self, values: dict[str, str]) -> dict[str, str]:
        with self._session() as session:
            for key, value in values.items():
                row = session.get(SettingRow, key)
                if row is None:
                    session.add(SettingRow(key=str(key), value=str(value)))
                else:
                    row.value = str(value)
            session.commit()
        return self.all_settings()

    # -- artifacts -----------------------------------------------------------
    def doc_dir(self, doc_id: int) -> Path:
        return self.docs_dir / str(doc_id)

    def disk_usage_bytes(self) -> int:
        total = 0
        for path in self.docs_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total

    def dispose(self) -> None:
        self._engine.dispose()
