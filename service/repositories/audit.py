"""The action log: who did what, to which object, when.

Thin on purpose — the rules that matter live in ``AuditEntry`` (no personal data)
and in ``FileStore.append_audit`` (append a line, never rewrite the file, never
let a logging failure break the action being logged).

``record`` takes the actor as a string rather than a ``User`` so the PIN path can
use it too: in PIN mode there is no account, and the actor is the literal
``'pin'``. That keeps one log covering both modes instead of a second mechanism
that only exists in one of them.
"""
from __future__ import annotations

from service.core.database import FileStore
from service.core.models import AuditEntry


def record(db: FileStore, *, action: str, actor: str = "", target_type: str = "",
           target_id: str | int = "", detail: str = "") -> AuditEntry:
    return db.append_audit(AuditEntry(
        id=0,                       # assigned under the store's lock
        action=action,
        actor=actor or "anonymous",
        target_type=target_type,
        target_id=str(target_id or ""),
        detail=detail,
    ))


def recent(db: FileStore, *, limit: int = 200, action: str | None = None,
           actor: str | None = None) -> list[AuditEntry]:
    return db.recent_audit(limit=limit, action=action, actor=actor)
