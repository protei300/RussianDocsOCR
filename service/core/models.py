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


#: The three roles, from least to most privileged. Ordered on purpose: the
#: permission check is a comparison against this list, not a table of every
#: (role, endpoint) pair — a table grows a hole the day someone adds an endpoint
#: and forgets a row, and that hole is silent.
ROLES = ("viewer", "operator", "admin")


def role_at_least(role: str, required: str) -> bool:
    """True when ``role`` is ``required`` or higher in the ROLES ordering."""
    try:
        return ROLES.index(role) >= ROLES.index(required)
    except ValueError:                      # unknown role: deny, never default to allow
        return False


@dataclass
class User:
    """A named account for the website, used when ``AUTH_MODE=users``.

    Only the hash is stored, like ``ApiKey`` — but the algorithm is different
    and the reason is in ``core/passwords.py``: an API key is random, a password
    is not.

    ``token_version`` is the part worth reading twice. It is embedded in every
    issued JWT and compared on each request. Bumping it — on a password change,
    on deactivation, on deletion — invalidates every token already handed out
    for that account, including ones on other devices. Without it a disabled
    administrator keeps working access until their token expires, which on this
    service is eight hours. It costs one integer and closes a real hole.

    ``must_change_password`` exists because the first account ships with a
    known, deliberately weak password. A default credential that can be used
    indefinitely is the single most common way a demo turns into an incident.
    """

    id: int
    username: str
    role: str = "viewer"
    password_hash: str = ""
    display_name: str = ""
    is_active: bool = True
    must_change_password: bool = False
    token_version: int = 1
    created_at: datetime = field(default_factory=utcnow)
    last_login_at: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["created_at"] = iso(self.created_at)
        data["last_login_at"] = iso(self.last_login_at)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "User":
        parsed = dict(data)
        for key in ("created_at", "last_login_at"):
            raw = parsed.get(key)
            parsed[key] = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None
        parsed.setdefault("created_at", utcnow())
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in parsed.items() if k in known})

    def public(self) -> dict[str, Any]:
        """What the UI may see — never the hash, never the token version."""
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name or self.username,
            "role": self.role,
            "is_active": self.is_active,
            "must_change_password": self.must_change_password,
            "created_at": iso(self.created_at),
            "last_login_at": iso(self.last_login_at),
        }


@dataclass
class AuditEntry:
    """One recorded action: who did what, to which object, when.

    **No personal data goes in here, and that is a hard rule rather than a
    preference.** Documents are erased at every restart in temporary mode; the
    audit log deliberately is not. Writing a filename such as
    ``Ivanov_passport.jpg`` — let alone a recognised field — would quietly carry
    personal data across the very erasure that the ephemeral store promises. So
    the target is an *id*, and the detail field is for things like a role name
    or a username, never document content.

    Not tamper-proof, and the documentation says so plainly: anyone with write
    access to the data directory can edit this file. Real immutability needs
    signed records or an append-only store outside the service.
    """

    id: int
    action: str                      # 'login', 'login_failed', 'user.create', 'document.delete', …
    actor: str = ""                  # username, or 'pin' for the shared PIN session
    target_type: str = ""            # 'document' | 'user' | 'api_key' | 'settings'
    target_id: str = ""              # id only — never a filename
    detail: str = ""                 # short, non-personal: a role, a username, a status
    at: datetime = field(default_factory=utcnow)

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["at"] = iso(self.at)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AuditEntry":
        parsed = dict(data)
        raw = parsed.get("at")
        parsed["at"] = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else utcnow()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in parsed.items() if k in known})

    def public(self) -> dict[str, Any]:
        return self.to_json()
