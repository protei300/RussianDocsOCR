"""Authentication dependencies.

Three levels, because two kinds of caller share one API:

``require_session``
    Browser only. Backed by the PIN-issued JWT. Guards anything that manages
    the service itself — API keys, settings, logs — because those are operator
    concerns and an integration has no business touching them.

``require_api_or_session``
    Either a valid ``X-API-Key`` or a valid session JWT. Guards the working
    endpoints (upload, list, detail, artifacts), so the same routes serve the
    bundled UI and third-party integrations without duplicating them.

``optional_identity``
    Never rejects. For endpoints that vary their response by caller but must
    stay reachable, such as ``/auth/config``.

Why not one scheme for both: a four-digit PIN is a human affordance and a poor
service credential — shared, guessable, and it would have to be embedded in
every integration. An API key is the opposite. Conflating them would force one
of the two use cases into the wrong shape.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from service.core.auth import decode_access_token
from service.core.database import DbSession, get_db
from service.repositories import api_keys as key_repo

_bearer = HTTPBearer(auto_error=False)

#: Shape of the single operator identity. There are no user accounts — the PIN
#: authenticates "whoever is at the console", nothing finer.
SESSION_USER = {"name": "Operator", "role": "admin"}


def _session_from_bearer(
    credentials: HTTPAuthorizationCredentials | None,
) -> dict[str, Any] | None:
    if credentials is None:
        return None
    return decode_access_token(credentials.credentials)


def optional_identity(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: DbSession = Depends(get_db),
) -> dict[str, Any] | None:
    """Best-effort identification. Returns ``None`` for anonymous callers."""
    claims = _session_from_bearer(credentials)
    if claims:
        return {"kind": "session", **claims}

    if x_api_key:
        key = key_repo.verify(db, x_api_key)
        if key is not None:
            key_repo.touch(db, key)
            return {"kind": "api_key", "key_id": key.id, "name": key.label,
                    "role": "service"}
    return None


def require_session(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    claims = _session_from_bearer(credentials)
    if not claims:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Sign in with the PIN to use this endpoint",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"kind": "session", **claims}


def require_api_or_session(
    identity: dict[str, Any] | None = Depends(optional_identity),
) -> dict[str, Any]:
    if identity is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Provide an API key in X-API-Key, or sign in with the PIN",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return identity
