"""API key storage and verification.

Only hashes are persisted; the plaintext key exists exactly once, in the
response to the create call.

The environment-provided default key is **synthesised at startup, not stored**.
That keeps one awkward case honest: runtime-created keys live in the ephemeral
store and vanish on restart, so if the default were also merely stored, a
restart could leave the API with no working credential at all. Deriving it from
the environment on every boot means it is always present, and deleting it is
refused rather than silently undone by the next restart.
"""
from __future__ import annotations

import secrets
from typing import Any

from service.core import auth
from service.core.config import get_settings
from service.core.database import FileStore
from service.core.models import ApiKey, utcnow

#: Reserved id for the environment key, so it never collides with stored ones.
DEFAULT_KEY_ID = 0


def _default_key() -> ApiKey:
    raw, generated = auth.resolve_default_key()
    return ApiKey(
        id=DEFAULT_KEY_ID,
        label="Default (generated at startup)" if generated else "Default (environment)",
        prefix=auth.key_prefix(raw),
        key_hash=auth.hash_api_key(raw),
        is_default=True,
    )


def get_all(db: FileStore) -> list[ApiKey]:
    """Every usable key, default first."""
    stored = sorted(db.all_api_keys(), key=lambda k: k.created_at)
    return [_default_key(), *stored]


def create(db: FileStore, label: str) -> tuple[ApiKey, str]:
    """Mint a key. Returns ``(record, plaintext)`` — the plaintext is shown once."""
    raw = auth.generate_api_key()
    record = ApiKey(
        id=db.next_api_key_id() or 1,
        label=(label or "").strip() or "Unnamed key",
        prefix=auth.key_prefix(raw),
        key_hash=auth.hash_api_key(raw),
        is_default=False,
    )
    if record.id == DEFAULT_KEY_ID:      # never shadow the environment key
        record.id = 1
    db.put_api_key(record)
    return record, raw


def delete(db: FileStore, key_id: int) -> bool:
    return db.drop_api_key(key_id)


def verify(db: FileStore, candidate: str) -> ApiKey | None:
    """Match a presented key against every known hash, in constant time.

    ``compare_digest`` on each candidate rather than a dict lookup: a plain
    ``==`` leaks how much of the hash matched via timing. The list is tiny, so
    scanning it costs nothing.
    """
    if not candidate:
        return None
    digest = auth.hash_api_key(candidate)
    for key in get_all(db):
        if secrets.compare_digest(digest, key.key_hash):
            return key
    return None


def touch(db: FileStore, key: ApiKey) -> None:
    """Record last use. The environment key is not persisted, so skip it."""
    if key.is_default:
        return
    key.last_used_at = utcnow()
    db.put_api_key(key)


def public_list(db: FileStore) -> list[dict[str, Any]]:
    """Keys for the UI.

    The generated default is returned **in full**: it exists only in this
    process's memory, so masking it would make it unusable — the operator would
    have no way to learn a key the service invented. A key supplied via
    ``DEFAULT_API_KEY`` stays masked, because whoever set it already has it and
    echoing a configured secret back into a browser is gratuitous.
    """
    raw, generated = auth.resolve_default_key()
    out = []
    for key in get_all(db):
        entry = key.public()
        if key.is_default:
            entry["is_generated"] = generated
            if generated:
                entry["key"] = raw
        out.append(entry)
    return out
