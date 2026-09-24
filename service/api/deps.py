"""Authentication and authorisation dependencies — the single gate.

Two kinds of caller share one API, and now two authentication modes share one
gate. Everything below exists so that **one function decides**, because the
alternative — each endpoint checking for itself — grows a hole the day someone
adds a route and forgets a line, and that hole is silent.

``require_api_or_session``
    Either a valid ``X-API-Key`` or a valid browser session. Guards the working
    endpoints (upload, list, detail, artifacts) so the bundled UI and
    third-party integrations use the same routes.

``require_session``
    Browser only. Guards service management — API keys, settings, logs, users.

``require_role('admin' | 'operator' | 'viewer')``
    Browser only, plus a minimum role. In PIN mode there are no roles and the
    single operator identity satisfies all of them; in users mode the account's
    role is compared against the ordering in ``models.ROLES``.

``require_session_allow_password_change``
    The one dependency that lets a *restricted* session through — see below.

``optional_identity``
    Never rejects. For endpoints that vary by caller but must stay reachable.

**Fail-safe by default, and that shape is deliberate.** An account flagged
``must_change_password`` gets a real token, but every dependency here refuses it
except the explicitly named one used by the change-password endpoint. So a route
added later without thinking about it blocks the restricted session rather than
serving it. Naming the permissive dependency after what it permits is the point:
you cannot use it by accident.

**Why the session is re-checked against the store on every request.** A JWT is
valid until it expires — eight hours here — which means a disabled account, a
changed password or a demoted role would otherwise keep their old authority for
the rest of that window. Each request therefore loads the user and compares
``token_version``; anything that changes authority bumps it and every issued
token dies at once. The cost is one dictionary lookup in an in-memory index.
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from service.core import auth as auth_core
from service.core.auth import decode_access_token
from service.core.database import DbSession, get_db
from service.core.models import role_at_least
from service.repositories import api_keys as key_repo
from service.repositories import users as user_repo

_bearer = HTTPBearer(auto_error=False)

#: The PIN identity. There are no accounts in that mode — the PIN authenticates
#: "whoever is at the console", nothing finer, so it is granted the top role and
#: the UI hides everything user-related.
SESSION_USER = {"name": "Operator", "role": "admin"}

#: Machine-readable reason on the 403 that means "change your password first".
#: The UI routes on this string rather than on the message text, which is prose
#: and will be reworded.
PASSWORD_CHANGE_REQUIRED = "password_change_required"


def _unauthorised(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail=detail,
                         headers={"WWW-Authenticate": "Bearer"})


def _session_identity(
    credentials: HTTPAuthorizationCredentials | None,
    db: DbSession,
) -> dict[str, Any] | None:
    """Decode a bearer token into an identity, or ``None`` if it is not usable."""
    if credentials is None:
        return None
    claims = decode_access_token(credentials.credentials)
    if not claims:
        return None

    if not auth_core.users_enabled():
        # PIN mode accepts only PIN tokens. A token minted while the service ran
        # with named accounts carries a uid, and it must be REJECTED, not merely
        # have its uid ignored: every PIN session is the administrator, so
        # "ignoring" the uid would promote a viewer's still-valid token to full
        # control the moment the service is switched back to PIN. The earlier
        # version of this branch did exactly that while its comment claimed the
        # opposite.
        if "uid" in claims:
            return None
        return {"kind": "session", "sub": claims.get("sub", "operator"), **SESSION_USER}

    user_id = claims.get("uid")
    if not isinstance(user_id, int):
        return None                       # a PIN-era token, worthless in users mode
    user = user_repo.get(db, user_id)
    if user is None or not user.is_active:
        return None
    if claims.get("tv") != user.token_version:
        # Password changed, role changed or the account was disabled since this
        # token was issued. This is the whole reason token_version exists.
        return None

    return {
        "kind": "session",
        "sub": user.username,
        "user_id": user.id,
        "username": user.username,
        "name": user.display_name or user.username,
        "role": user.role,
        "must_change_password": user.must_change_password,
    }


def optional_identity(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: DbSession = Depends(get_db),
) -> dict[str, Any] | None:
    """Best-effort identification. Returns ``None`` for anonymous callers."""
    identity = _session_identity(credentials, db)
    if identity is not None:
        return identity

    if x_api_key:
        key = key_repo.verify(db, x_api_key)
        if key is not None:
            key_repo.touch(db, key)
            return {"kind": "api_key", "key_id": key.id, "name": key.label,
                    "sub": key.label, "role": "service"}
    return None


def require_session_allow_password_change(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: DbSession = Depends(get_db),
) -> dict[str, Any]:
    """A valid session, **including** one that still owes a password change.

    Used by exactly two endpoints: change-my-password and who-am-I. Everything
    else must use ``require_session`` or ``require_role``.
    """
    identity = _session_identity(credentials, db)
    if identity is None:
        raise _unauthorised("Sign in to use this endpoint")
    return identity


def _reject_if_password_change_pending(identity: dict[str, Any]) -> None:
    if identity.get("must_change_password"):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=PASSWORD_CHANGE_REQUIRED,
        )


def require_session(
    identity: dict[str, Any] = Depends(require_session_allow_password_change),
) -> dict[str, Any]:
    _reject_if_password_change_pending(identity)
    return identity


def require_api_or_session(
    identity: dict[str, Any] | None = Depends(optional_identity),
) -> dict[str, Any]:
    if identity is None:
        raise _unauthorised("Provide an API key in X-API-Key, or sign in")
    if identity.get("kind") == "session":
        _reject_if_password_change_pending(identity)
    return identity


def require_role(minimum: str) -> Callable[..., dict[str, Any]]:
    """Dependency factory: a session whose role is ``minimum`` or higher.

    In PIN mode every session is the single operator identity with the top role,
    so these gates are satisfied and the service behaves exactly as it did
    before named accounts existed. That is what keeps the change backward
    compatible rather than merely additive.
    """

    def dependency(identity: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
        if not role_at_least(str(identity.get("role", "")), minimum):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail=f"This action requires the {minimum} role",
            )
        return identity

    # Named after the role it demands, so the route table test can read what
    # each endpoint declares instead of only whether it declares anything.
    dependency.__name__ = f"require_{minimum}"
    return dependency


require_admin = require_role("admin")
require_operator = require_role("operator")
require_viewer = require_role("viewer")


def require_api_or_role(minimum: str) -> Callable[..., dict[str, Any]]:
    """An API key, or a session whose role is ``minimum`` or higher.

    For the document endpoints, which serve both the bundled UI and third-party
    integrations. An API key is admitted at any level because its scope is the
    document API and nothing else — it cannot reach users, keys, settings or logs,
    which all use ``require_role`` and therefore refuse API keys outright.

    **Why this exists as a separate factory rather than a flag on require_role.**
    The first version of named accounts guarded these routes with plain
    ``require_api_or_session``, which checks that a caller is *someone* and never
    what they may do. A viewer could upload, reprocess, delete and purge — and the
    route-coverage test was green the whole time, because it asked whether a guard
    was present, not whether it was the right one. Every document route now names
    the role it needs, and the test checks that name against an explicit table.
    """

    def dependency(identity: dict[str, Any] = Depends(require_api_or_session)) -> dict[str, Any]:
        if identity.get("kind") == "api_key":
            return identity
        if not role_at_least(str(identity.get("role", "")), minimum):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail=f"This action requires the {minimum} role",
            )
        return identity

    dependency.__name__ = f"require_api_or_{minimum}"
    return dependency


def actor_of(identity: dict[str, Any] | None) -> str:
    """The name to put in the audit log. ``'pin'`` when there is no account."""
    if not identity:
        return "anonymous"
    if identity.get("kind") == "api_key":
        return f"api_key:{identity.get('name', '?')}"
    return str(identity.get("username") or "pin")
