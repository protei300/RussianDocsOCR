"""Record shapes for the filesystem store.

**SQL swap point.** These dataclasses become SQLAlchemy ``DeclarativeBase``
models when a real database arrives. Field names are therefore chosen to be
valid SQL column names, and the API layer depends on *these names only* — so
the swap touches this file, ``database.py`` and the repository bodies, and
nothing else.

Two denormalisations are deliberate and would be kept in SQL:

* ``doc_type`` / ``doc_conf`` / ``processing_ms`` / ``canvas_*`` are columns, so
  the list page can filter and sort without parsing the stored result blob.
* ``search_text`` is a precomputed lowercase haystack (filename + doc type +
  every recognised value). Without it, "search by recognised surname" means
  parsing every result blob on every keystroke. In SQL this becomes an indexed
  computed column.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The only statuses a document can hold. Same set as the internal reference service
#: pattern so the frontend's badge classes map one-to-one.
VALID_STATUSES = frozenset({"queued", "processing", "done", "failed"})


def utcnow() -> datetime:
    """Timezone-aware UTC. Serialised with an explicit ``Z`` on the wire.

    Naive timestamps are how you end up with a frontend guessing the zone; the
    reference project papers over it client-side and we would rather not.
    """
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class DocumentRecord:
    """One uploaded document and everything known about it."""

    id: int
    filename: str            # sanitised, display only — never a filesystem path
    content_type: str
    size_bytes: int
    status: str = "queued"

    doc_type: str | None = None
    doc_conf: float | None = None
    recognised: bool = False
    field_count: int = 0
    #: Denormalised quality verdicts (Glare/Blur/PrintSpoofing/LCDSpoofing) so
    #: the list page can show them without loading each result blob. Values are
    #: whatever the library reports — currently 'good'/'bad' for the first two
    #: and 'REAL'/'FAKE' for the spoofing checks, so clients must not assume a
    #: single vocabulary.
    quality: dict[str, Any] = field(default_factory=dict)

    device: str | None = None
    processing_ms: int | None = None
    error: str | None = None
    error_code: str | None = None
    retry_count: int = 0

    original_ext: str = ".jpg"
    original_w: int | None = None
    original_h: int | None = None
    canvas_w: int | None = None
    canvas_h: int | None = None
    has_canvas: bool = False

    search_text: str = ""

    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime = field(default_factory=utcnow)

    #: Full recognition view model. Kept OUT of the in-memory index (it can be
    #: 100 KB of boxes) and loaded lazily by ``get_by_id``.
    result: dict[str, Any] | None = None

    # -- persistence helpers -------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        """Everything except ``result``, which is stored in its own file."""
        data = dataclasses.asdict(self)
        data.pop("result", None)
        for key in ("created_at", "started_at", "finished_at", "updated_at"):
            data[key] = iso(data[key])
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DocumentRecord":
        parsed = dict(data)
        for key in ("created_at", "started_at", "finished_at", "updated_at"):
            raw = parsed.get(key)
            parsed[key] = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None
        parsed.setdefault("created_at", utcnow())
        parsed.setdefault("updated_at", utcnow())
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in parsed.items() if k in known})


@dataclass
class ApiKey:
    """An API key for machine callers.

    The plaintext key is shown **once**, at creation, and only its hash is kept
    — same reasoning as any password store: a leaked data directory should not
    hand over working credentials.
    """

    id: int
    label: str
    prefix: str              # first few chars, for identifying a key in the UI
    key_hash: str            # sha256 of the full key
    is_default: bool = False  # comes from the environment; cannot be deleted
    created_at: datetime = field(default_factory=utcnow)
    last_used_at: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["created_at"] = iso(self.created_at)
        data["last_used_at"] = iso(self.last_used_at)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ApiKey":
        parsed = dict(data)
        for key in ("created_at", "last_used_at"):
            raw = parsed.get(key)
            parsed[key] = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None
        parsed.setdefault("created_at", utcnow())
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in parsed.items() if k in known})

    def public(self) -> dict[str, Any]:
        """What the UI may see — never the hash."""
        return {
            "id": self.id,
            "label": self.label,
            "prefix": self.prefix,
            "masked": f"{self.prefix}{'•' * 8}",
            "is_default": self.is_default,
            "created_at": iso(self.created_at),
            "last_used_at": iso(self.last_used_at),
        }
