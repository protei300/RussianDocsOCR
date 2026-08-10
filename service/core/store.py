"""The storage contract both backends implement.

Two backends exist:

* ``FileStore`` (``core.database``) — JSON files on disk, wiped at every start.
  The zero-configuration default so the service runs out of the box.
* ``SqlStore`` (``core.db_sql``) — any SQLAlchemy-supported database. What you
  use when the data matters.

Everything above this layer — routers, worker, repositories — is written against
the protocol below and cannot tell them apart. That is what makes swapping the
backend a configuration change rather than a rewrite, and
``tests/service/test_repository_contract.py`` is what keeps the two honest by
running the same suite against both.

Query methods live *here*, not in the repository functions, on purpose. Filtering
a list in Python is correct for a few hundred JSON files and wrong for a table:
each backend needs to express "the newest twenty matching rows" in its own
terms. Putting the queries behind the protocol lets it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from service.core.models import ApiKey, DocumentRecord


@runtime_checkable
class DocumentStore(Protocol):
    """Everything the service needs from a storage backend."""

    # -- identity ------------------------------------------------------------
    @property
    def backend(self) -> str:
        """``'files'`` or ``'sql'`` — surfaced on the status page."""

    @property
    def is_ephemeral(self) -> bool:
        """True when the contents do not survive a restart."""

    # -- documents -----------------------------------------------------------
    def next_document_id(self) -> int: ...

    def get_record(self, doc_id: int) -> DocumentRecord | None: ...

    def put_record(self, record: DocumentRecord) -> DocumentRecord: ...

    def drop_record(self, doc_id: int) -> None: ...

    def query_documents(self, *, status: str | None = None, doc_type: str | None = None,
                        search: str | None = None, date_from: str | None = None,
                        date_to: str | None = None, page: int = 1, page_size: int = 20,
                        sort_by: str = "created_at",
                        sort_dir: str = "desc") -> tuple[list[DocumentRecord], int]:
        """One page of matching records, plus the unpaged total."""

    def all_records(self) -> list[DocumentRecord]:
        """Every record. Only for small-scale operations (purge, recovery)."""

    def next_queued_id(self) -> int | None: ...

    def queue_position(self, doc_id: int) -> int | None: ...

    def count_by_status(self) -> dict[str, int]: ...

    def aggregate_stats(self) -> dict[str, Any]: ...

    # -- results -------------------------------------------------------------
    def save_result_payload(self, doc_id: int, payload: dict[str, Any]) -> None:
        """Persist the recognition view model for a document."""

    def load_result_payload(self, doc_id: int) -> dict[str, Any] | None: ...

    # -- api keys ------------------------------------------------------------
    def all_api_keys(self) -> list[ApiKey]: ...

    def next_api_key_id(self) -> int: ...

    def put_api_key(self, key: ApiKey) -> ApiKey: ...

    def drop_api_key(self, key_id: int) -> bool: ...

    # -- settings ------------------------------------------------------------
    def all_settings(self) -> dict[str, str]: ...

    def set_settings(self, values: dict[str, str]) -> dict[str, str]: ...

    # -- artifacts -----------------------------------------------------------
    # Binary files stay on the filesystem regardless of backend, so this is a
    # plain directory in both cases.
    def doc_dir(self, doc_id: int) -> Path: ...

    def disk_usage_bytes(self) -> int: ...


#: Sortable columns, shared by both backends so they cannot drift apart.
#: A whitelist rather than dynamic attribute access — in the SQL backend that
#: difference is an injection vector.
SORT_COLUMNS = frozenset({
    "created_at", "filename", "status", "doc_type", "doc_conf",
    "processing_ms", "size_bytes",
})
