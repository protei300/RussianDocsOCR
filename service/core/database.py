"""Filesystem-backed store, shaped like a database session.

The service has no database on purpose — data is a scratch pad that dies with
the container. But "no database" must not mean "no abstraction", or moving to
SQL later becomes a rewrite. So this module presents the same surface a
SQLAlchemy session would, and the repositories on top of it use the same
plain-function style (``db`` first argument) as the reference project.

**SQL swap point.** Replacing this file with ``create_engine`` /
``SessionLocal`` / a generator ``get_db``, and setting ``DbSession = Session``,
is the whole migration as far as callers are concerned. Router and worker code
does not change.

On-disk layout::

    $DATA_DIR/
      documents/42/
        record.json     the "row"
        original.jpg    exactly the bytes uploaded
        canvas.png      the deskewed/rectified canvas
        result.json     the full recognition view model
      api_keys.json
      settings.json

Design notes worth knowing before changing anything here:

* **The index lives in memory; disk is scanned once at startup.** The service
  is pinned to one process (the pipeline singleton and this index both are), so
  a shared in-memory index is legitimate. The startup rescan is what keeps
  ``uvicorn --reload`` from losing everything mid-session.
* **Writes are atomic** (temp file + ``os.replace``, atomic on NTFS and ext4).
  A half-written ``record.json`` would survive a crash and poison the next boot.
* **Reads return copies.** SQLAlchemy hands back live identity-mapped objects;
  we cannot, so ``update()`` returns a *new* record and callers must rebind.
  That is already how the reference project's routers are written
  (``m = repo.update_status(db, m, ...)``), so the idiom ports cleanly.
* **``result`` is not held in the index** — it can be 100 KB of boxes per
  document. ``get_by_id`` loads it lazily; list queries never touch it.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import dataclasses

from service.core.models import ApiKey, DocumentRecord
from service.core.store import SORT_COLUMNS

log = logging.getLogger(__name__)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON so a crash can never leave a partial file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class FileStore:
    """Stand-in for a SQLAlchemy ``Session``.

    Deliberately named and shaped like one so repository signatures and
    ``Depends(get_db)`` survive the eventual move to SQL. Unlike a real session
    it is a process-wide singleton with no per-request state: there is nothing
    to commit and nothing to close.

    Thread safety: one ``RLock`` guards the index and all mutations. Both the
    worker thread and FastAPI's request threads write here. ``RLock`` rather
    than ``Lock`` because some operations nest (``create`` can touch other
    records). Long I/O — writing a 2 MB PNG — happens *outside* the lock; only
    the rename and the index update are inside it.
    """

    def __init__(self, root: Path):
        self.root = root
        self.docs_dir = root / "documents"
        self.lock = threading.RLock()
        self._records: dict[int, DocumentRecord] = {}
        self._api_keys: dict[int, ApiKey] = {}
        self._settings: dict[str, str] = {}
        self._next_doc_id = 1
        self._next_key_id = 1
        self._scan()

    # -- paths ---------------------------------------------------------------
    @property
    def api_keys_path(self) -> Path:
        return self.root / "api_keys.json"

    @property
    def settings_path(self) -> Path:
        return self.root / "settings.json"

    def doc_dir(self, doc_id: int) -> Path:
        return self.docs_dir / str(doc_id)

    # -- startup -------------------------------------------------------------
    def _scan(self) -> None:
        """Rebuild the in-memory index from disk. Cheap: N small JSON reads."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        loaded = 0
        for entry in sorted(self.docs_dir.iterdir(), key=lambda p: p.name):
            record_file = entry / "record.json"
            if not entry.is_dir() or not record_file.is_file():
                continue
            try:
                record = DocumentRecord.from_json(json.loads(record_file.read_text("utf-8")))
            except Exception:
                # A corrupt record must not stop the service from starting; the
                # rest of the scratch data is still perfectly usable.
                log.exception("[STORE] skipping unreadable record %s", record_file)
                continue
            self._records[record.id] = record
            self._next_doc_id = max(self._next_doc_id, record.id + 1)
            loaded += 1

        if self.api_keys_path.is_file():
            try:
                for raw in json.loads(self.api_keys_path.read_text("utf-8")):
                    key = ApiKey.from_json(raw)
                    self._api_keys[key.id] = key
                    self._next_key_id = max(self._next_key_id, key.id + 1)
            except Exception:
                log.exception("[STORE] api_keys.json unreadable — starting with none")

        if self.settings_path.is_file():
            try:
                self._settings = {str(k): str(v) for k, v in
                                  json.loads(self.settings_path.read_text("utf-8")).items()}
            except Exception:
                log.exception("[STORE] settings.json unreadable — using defaults")

        if loaded:
            log.info("[STORE] recovered %d document(s) from %s", loaded, self.docs_dir)

    # -- documents -----------------------------------------------------------
    def next_document_id(self) -> int:
        with self.lock:
            doc_id = self._next_doc_id
            self._next_doc_id += 1
            return doc_id

    def all_records(self) -> list[DocumentRecord]:
        with self.lock:
            return list(self._records.values())

    def get_record(self, doc_id: int) -> DocumentRecord | None:
        """A copy, with the lazily-stored result attached.

        A copy rather than the indexed instance: callers mutate what they get
        back, and a shared instance would let one request's edit leak into
        another's view.
        """
        with self.lock:
            record = self._records.get(doc_id)
        if record is None:
            return None
        loaded = dataclasses.replace(record)
        loaded.result = self.load_result_payload(doc_id)
        return loaded

    def put_record(self, record: DocumentRecord) -> DocumentRecord:
        """Persist a record and index it. Returns the stored instance."""
        directory = self.doc_dir(record.id)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "record.json", record.to_json())
        with self.lock:
            self._records[record.id] = record
            self._next_doc_id = max(self._next_doc_id, record.id + 1)
        return record

    def drop_record(self, doc_id: int) -> None:
        with self.lock:
            self._records.pop(doc_id, None)
        shutil.rmtree(self.doc_dir(doc_id), ignore_errors=True)

    # -- api keys ------------------------------------------------------------
    def all_api_keys(self) -> list[ApiKey]:
        with self.lock:
            return list(self._api_keys.values())

    def next_api_key_id(self) -> int:
        with self.lock:
            key_id = self._next_key_id
            self._next_key_id += 1
            return key_id

    def put_api_key(self, key: ApiKey) -> ApiKey:
        with self.lock:
            self._api_keys[key.id] = key
            self._next_key_id = max(self._next_key_id, key.id + 1)
            self._flush_api_keys_locked()
        return key

    def drop_api_key(self, key_id: int) -> bool:
        with self.lock:
            if key_id not in self._api_keys:
                return False
            del self._api_keys[key_id]
            self._flush_api_keys_locked()
            return True

    def _flush_api_keys_locked(self) -> None:
        atomic_write_json(self.api_keys_path,
                          [k.to_json() for k in self._api_keys.values()])

    # -- settings ------------------------------------------------------------
    def all_settings(self) -> dict[str, str]:
        with self.lock:
            return dict(self._settings)

    def set_settings(self, values: dict[str, str]) -> dict[str, str]:
        with self.lock:
            self._settings.update({str(k): str(v) for k, v in values.items()})
            atomic_write_json(self.settings_path, self._settings)
            return dict(self._settings)


    # -- identity ------------------------------------------------------------
    @property
    def backend(self) -> str:
        return "files"

    @property
    def is_ephemeral(self) -> bool:
        return True

    # -- queries -------------------------------------------------------------
    # Implemented in Python over the in-memory index. Correct at this scale
    # (a few hundred records) and honest about it: the SQL backend answers the
    # same questions with real queries.
    def query_documents(self, *, status=None, doc_type=None, search=None,
                        date_from=None, date_to=None, page=1, page_size=20,
                        sort_by="created_at", sort_dir="desc"):
        from datetime import datetime, timedelta, timezone

        def parse_day(value):
            if not value:
                return None
            try:
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return None  # a half-typed date disables the filter, not the page

        rows = self.all_records()
        if status:
            rows = [r for r in rows if r.status == status]
        if doc_type == "__none__":
            rows = [r for r in rows if not r.recognised]
        elif doc_type:
            rows = [r for r in rows if (r.doc_type or "").startswith(doc_type)]
        start = parse_day(date_from)
        if start:
            rows = [r for r in rows if r.created_at >= start]
        end = parse_day(date_to)
        if end:
            rows = [r for r in rows if r.created_at < end + timedelta(days=1)]
        if search:
            needle = search.lower().strip()
            rows = [r for r in rows if needle in r.search_text]

        total = len(rows)
        column = sort_by if sort_by in SORT_COLUMNS else "created_at"
        # None last in both directions, matching the SQL backend: a queued
        # document has no doc_conf and must not lead an ascending sort.
        rows.sort(key=lambda r: (getattr(r, column) is None,
                                 getattr(r, column) or (r.created_at if column == "created_at" else 0)),
                  reverse=(sort_dir != "asc"))
        offset = max(0, (page - 1) * page_size)
        return rows[offset:offset + page_size], total

    def next_queued_id(self):
        queued = [r for r in self.all_records() if r.status == "queued"]
        return min(queued, key=lambda r: r.created_at).id if queued else None

    def queue_position(self, doc_id: int):
        queued = sorted((r for r in self.all_records() if r.status == "queued"),
                        key=lambda r: r.created_at)
        for index, record in enumerate(queued):
            if record.id == doc_id:
                return index
        return None

    def count_by_status(self) -> dict[str, int]:
        counts = {"queued": 0, "processing": 0, "done": 0, "failed": 0}
        for record in self.all_records():
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def aggregate_stats(self) -> dict[str, Any]:
        records = self.all_records()
        timed = [r.processing_ms for r in records if r.status == "done" and r.processing_ms]
        return {
            **self.count_by_status(),
            "total": len(records),
            "recognised": sum(1 for r in records if r.recognised),
            "avg_processing_ms": round(sum(timed) / len(timed)) if timed else None,
        }

    # -- results -------------------------------------------------------------
    def save_result_payload(self, doc_id: int, payload: dict[str, Any]) -> None:
        directory = self.doc_dir(doc_id)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "result.json", payload)

    def load_result_payload(self, doc_id: int) -> dict[str, Any] | None:
        path = self.doc_dir(doc_id) / "result.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            log.exception("[STORE] unreadable result.json for document %s", doc_id)
            return None

    # -- housekeeping --------------------------------------------------------
    def disk_usage_bytes(self) -> int:
        total = 0
        for path in self.docs_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total


_STORE: FileStore | None = None


def set_store(store: Any) -> Any:
    """Install the process-wide store. Called once by the lifespan.

    Accepts either backend — ``get_db``/``get_store`` are backend-agnostic, which
    is what keeps routers and the worker identical in both modes.
    """
    global _STORE
    _STORE = store
    return store


def init_store(data_dir: str | Path, *, wipe: bool = False) -> FileStore:
    """Create the process-wide store. Called once from the lifespan."""
    global _STORE
    root = Path(data_dir).resolve()
    if wipe and root.exists():
        # Deliberate: see Settings.data_wipe_on_start. `docker restart` keeps
        # the writable layer, so without this the "ephemeral" store would
        # quietly persist across restarts.
        size_mb = 0
        try:
            size_mb = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) // (1024 * 1024)
        except OSError:
            pass
        shutil.rmtree(root, ignore_errors=True)
        log.info("[STORE] wiped data directory on startup (%d MB) — %s", size_mb, root)
    root.mkdir(parents=True, exist_ok=True)
    _STORE = FileStore(root)
    return _STORE


def get_store() -> Any:
    if _STORE is None:
        raise RuntimeError("store not initialised — init_store() must run in the lifespan")
    return _STORE


def get_db() -> Iterator[Any]:
    """FastAPI dependency.

    Generator-shaped on purpose: the SQLAlchemy version
    (``db = SessionLocal(); try: yield db; finally: db.close()``) is a literal
    drop-in, so no router signature changes.
    """
    yield get_store()


@contextmanager
def db_session() -> Iterator[Any]:
    """For background code, which has no request scope to hang a dependency on."""
    yield get_store()


#: Router annotation. Intentionally the protocol rather than a concrete class,
#: because either backend can be behind it — see ``core.store.DocumentStore``.
DbSession = Any
