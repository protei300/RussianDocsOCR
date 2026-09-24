"""Sign-in for the browser UI: a shared PIN, or a named account.

Which one is live is decided by ``AUTH_MODE`` and resolved in ``core/auth.py``.
This module never reads the setting directly — it asks for the effective mode, so
a misconfiguration that was downgraded to PIN behaves consistently everywhere
instead of half-working.

``GET /auth/config`` is the only endpoint here that answers before anyone has
authenticated, and it is what makes one frontend serve both modes: the login page
asks what to render rather than guessing. The Go, .NET and Kotlin services
implement the same endpoint (``ports/AUTH.md``), so one shared UI serves all four;
a service that did not would fall back to the PIN keypad.

Security notes on this file specifically:

* Failed sign-ins are **throttled per (identity, address)** and recorded in the
  audit log. The submitted PIN or password is never logged — writing rejected
  credentials to disk is its own small leak, and rejected ones are often a typo
  away from the real one.
* The reply to a bad username and to a bad password is identical, including its
  timing (``users.authenticate`` verifies against a decoy hash when the account
  does not exist). Distinguishing them hands over a list of valid usernames.
* A successful sign-in for an account that still owes a password change returns a
  **restricted** token: it is real, but every dependency except the password
  change refuses it. The response says so with ``must_change_password`` so the UI
  can route straight to the change form instead of bouncing off a 403.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from service.api.deps import (SESSION_USER, require_session_allow_password_change)
from service.core import auth as auth_core
from service.core import passwords
from service.core.auth import create_access_token, verify_pin
from service.core.config import DEFAULT_ADMIN_PASSWORD, get_settings
from service.core.database import DbSession, get_db
from service.repositories import audit as audit_repo
from service.repositories import users as user_repo

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class PinRequest(BaseModel):
    pin: str = Field(min_length=1, max_length=32)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


def _client(request: Request) -> str:
    return request.client.host if request.client else "-"


def _token_for(user) -> str:
    return create_access_token({
        "sub": user.username,
        "uid": user.id,
        "tv": user.token_version,       # see deps: this is what kills stale sessions
        "role": user.role,
        "name": user.display_name or user.username,
    })


def _demo_credentials_to_advertise(db, settings) -> dict | None:
    """The seeded credentials, **only while publishing them is harmless.**

    Two conditions, and both are necessary.

    *The configured password is the built-in demo one.* The first version of this
    endpoint returned ``settings.admin_password`` unconditionally — so an operator
    who set ``ADMIN_PASSWORD`` to a real secret had it printed on the login page
    for every anonymous visitor. A value from the environment is a secret by
    default; only the documented demo value is not.

    *The seeded account still owes its password change.* After that, "admin/1234"
    is false, and advertising a credential that does not work is noise at best and
    a hint about the account name at worst.
    """
    if settings.admin_password != DEFAULT_ADMIN_PASSWORD:
        return None
    seeded = user_repo.find(db, settings.admin_username)
    if seeded is None or not seeded.must_change_password:
        return None
    return {"username": seeded.username, "password": DEFAULT_ADMIN_PASSWORD}


@router.get("/config")
def auth_config(db: DbSession = Depends(get_db)) -> dict:
    """What the login page needs before anyone has authenticated.

    Deliberately reachable without a token, and deliberately says nothing that
    is not already visible: the mode, the password rules, and — only in the demo
    default — the seeded credentials, which are printed on the page anyway.
    """
    mode, downgrade_reason = auth_core.resolve_auth_mode()
    settings = get_settings()
    payload: dict = {
        "mode": mode,
        "pin_required": mode == auth_core.PIN_MODE,
        # Named so the UI can hide user management without inferring it from the
        # mode string; the two are the same today and need not stay that way.
        "users_enabled": mode == auth_core.USERS_MODE,
        "downgrade_reason": downgrade_reason,
    }
    if mode == auth_core.USERS_MODE:
        payload["password_rules"] = passwords.rules_for_ui()
        # Shown on the login page on purpose: this is a demonstration service
        # whose first account is seeded with a known password, and hiding it
        # would only mean the person testing cannot get in. What makes it
        # defensible is that the account can do nothing until it is changed.
        demo = _demo_credentials_to_advertise(db, settings)
        if demo is not None:
            payload["demo_credentials"] = demo
    return payload


@router.post("/pin-login")
def pin_login(body: PinRequest, request: Request, db: DbSession = Depends(get_db)) -> dict:
    if auth_core.users_enabled():
        # Not 401: the credential is not wrong, the endpoint is not in service.
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail="This service is configured for named accounts; "
                                   "sign in with a username and password")
    client = _client(request)
    blocked = auth_core.login_blocked_for("pin", client)
    if blocked:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Too many attempts. Try again in {blocked} s",
                            headers={"Retry-After": str(blocked)})

    if not verify_pin(body.pin):
        # Logged without the attempted value — writing rejected PINs to disk
        # would be its own small credential leak.
        auth_core.note_failed_login("pin", client)
        audit_repo.record(db, action="login_failed", actor="pin")
        log.warning("[API] rejected PIN sign-in attempt")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Wrong PIN")

    auth_core.clear_failed_logins("pin", client)
    audit_repo.record(db, action="login", actor="pin")
    token = create_access_token({"sub": "operator", **SESSION_USER})
    return {"access_token": token, "token_type": "bearer", "user": SESSION_USER}


@router.post("/login")
def login(body: LoginRequest, request: Request, db: DbSession = Depends(get_db)) -> dict:
    if not auth_core.users_enabled():
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail="This service is configured for PIN sign-in")
    client = _client(request)
    blocked = auth_core.login_blocked_for(body.username, client)
    if blocked:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Too many attempts. Try again in {blocked} s",
                            headers={"Retry-After": str(blocked)})

    user = user_repo.authenticate(db, body.username, body.password)
    if user is None:
        auth_core.note_failed_login(body.username, client)
        # The username is recorded because it is what makes the log useful when
        # someone is walking a list of accounts. The password never is.
        # The client address is deliberately NOT recorded. It is personal data,
        # and the audit log's rule is that nothing personal goes in — an earlier
        # version wrote it on every sign-in and broke that rule four times. The
        # throttle still uses the address, in memory, for the life of the process.
        audit_repo.record(db, action="login_failed", actor=body.username.strip())
        log.warning("[API] rejected sign-in for %r", body.username)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            detail="Wrong username or password")

    auth_core.clear_failed_logins(body.username, client)
    audit_repo.record(db, action="login", actor=user.username)
    return {
        "access_token": _token_for(user),
        "token_type": "bearer",
        "user": user.public(),
        "must_change_password": user.must_change_password,
    }


@router.get("/me")
def whoami(identity: dict = Depends(require_session_allow_password_change)) -> dict:
    """Who the current token belongs to.

    Uses the permissive dependency so the change-password screen can show who is
    signed in while the session is still restricted.
    """
    return {
        "mode": auth_core.auth_mode(),
        "user": {k: identity.get(k) for k in
                 ("username", "name", "role", "user_id", "must_change_password")},
    }


@router.post("/change-password")
def change_password(body: ChangePasswordRequest,
                    identity: dict = Depends(require_session_allow_password_change),
                    db: DbSession = Depends(get_db)) -> dict:
    """Change your own password. Ends every session you have, including this one.

    The current password is required even though the caller is already
    authenticated: a token left open on an unattended machine should not be
    enough to take an account over permanently.
    """
    if not auth_core.users_enabled():
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail="There are no accounts in PIN mode")
    user = user_repo.get(db, int(identity["user_id"]))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Session no longer valid")

    try:
        user_repo.change_password(db, user, body.new_password,
                                  current_password=body.current_password)
    except user_repo.UserError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    audit_repo.record(db, action="password.change", actor=user.username,
                      target_type="user", target_id=user.id)
    # No fresh token on purpose: the version bump has just invalidated this one,
    # and handing back a new one would quietly defeat the point of asking the
    # user to sign in again with the password they have just chosen.
    return {"status": "ok", "reauthenticate": True}
