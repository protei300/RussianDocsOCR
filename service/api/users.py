"""User management — administrators only, and only in ``AUTH_MODE=users``.

**In PIN mode every route here answers 404, not an empty list.** "There are no
users" and "users are not a concept in this configuration" are different
answers, and a UI that cannot tell them apart shows an empty management page
that nobody can make work. The frontend asks ``/auth/config`` and hides the
section entirely; the 404 is the backstop for anyone calling the API directly.

Every mutation is audited, with the acting administrator as the actor and the
affected account as the target — an id and a username, never a password.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from service.api.deps import require_admin
from service.core import auth as auth_core
from service.core import passwords
from service.core.database import DbSession, get_db
from service.core.models import ROLES
from service.repositories import audit as audit_repo
from service.repositories import users as user_repo

log = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    role: str = Field(default="viewer")
    display_name: str = Field(default="", max_length=128)


class UpdateUserRequest(BaseModel):
    role: str | None = None
    display_name: str | None = Field(default=None, max_length=128)
    is_active: bool | None = None


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(min_length=1, max_length=256)


def _require_users_mode() -> None:
    if not auth_core.users_enabled():
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail="User accounts are disabled (AUTH_MODE=pin)")


def _load(db: DbSession, user_id: int):
    user = user_repo.get(db, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such user")
    return user


def _bad_request(error: Exception) -> HTTPException:
    # UserError messages are written to be shown: "Cannot demote the last active
    # administrator" is the whole explanation, and swallowing it into a generic
    # 400 would leave the operator guessing which rule they hit.
    return HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(error))


@router.get("")
def list_users(db: DbSession = Depends(get_db), _admin=Depends(require_admin)) -> dict:
    _require_users_mode()
    return {
        "items": [u.public() for u in user_repo.get_all(db)],
        "roles": list(ROLES),
        "password_rules": passwords.rules_for_ui(),
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_user(body: CreateUserRequest, db: DbSession = Depends(get_db),
                admin: dict = Depends(require_admin)) -> dict:
    _require_users_mode()
    try:
        user = user_repo.create(db, username=body.username, password=body.password,
                                role=body.role, display_name=body.display_name,
                                must_change_password=True)
    except user_repo.UserError as error:
        raise _bad_request(error) from error

    audit_repo.record(db, action="user.create", actor=admin.get("username", "?"),
                      target_type="user", target_id=user.id,
                      detail=f"{user.username} as {user.role}")
    return user.public()


@router.patch("/{user_id}")
def update_user(user_id: int, body: UpdateUserRequest, db: DbSession = Depends(get_db),
                admin: dict = Depends(require_admin)) -> dict:
    _require_users_mode()
    user = _load(db, user_id)
    was = (user.role, user.is_active)
    try:
        user = user_repo.update(db, user, role=body.role,
                                display_name=body.display_name,
                                is_active=body.is_active)
    except user_repo.UserError as error:
        raise _bad_request(error) from error

    changes = []
    if was[0] != user.role:
        changes.append(f"role {was[0]}->{user.role}")
    if was[1] != user.is_active:
        changes.append("activated" if user.is_active else "deactivated")
    audit_repo.record(db, action="user.update", actor=admin.get("username", "?"),
                      target_type="user", target_id=user.id,
                      detail=f"{user.username}: {', '.join(changes) or 'profile'}")
    return user.public()


@router.post("/{user_id}/password")
def reset_password(user_id: int, body: ResetPasswordRequest,
                   db: DbSession = Depends(get_db),
                   admin: dict = Depends(require_admin)) -> dict:
    """Set someone else's password. They must change it at their next sign-in."""
    _require_users_mode()
    user = _load(db, user_id)
    try:
        user = user_repo.reset_password(db, user, body.new_password)
    except user_repo.UserError as error:
        raise _bad_request(error) from error

    audit_repo.record(db, action="user.password_reset", actor=admin.get("username", "?"),
                      target_type="user", target_id=user.id, detail=user.username)
    return user.public()


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: DbSession = Depends(get_db),
                admin: dict = Depends(require_admin)) -> Response:
    _require_users_mode()
    user = _load(db, user_id)
    try:
        user_repo.delete(db, user, acting_user_id=admin.get("user_id"))
    except user_repo.UserError as error:
        raise _bad_request(error) from error

    audit_repo.record(db, action="user.delete", actor=admin.get("username", "?"),
                      target_type="user", target_id=user_id, detail=user.username)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/audit/entries")
def list_audit(limit: int = 200, action: str | None = None, actor: str | None = None,
               db: DbSession = Depends(get_db), _admin=Depends(require_admin)) -> dict:
    """The action log. Administrator only — it names who did what.

    Available in **both** modes: in PIN mode the actor is the literal 'pin', and
    knowing that a document was deleted at a given moment is still worth more
    than nothing. Only the user-management part of this module is mode-gated.
    """
    entries = audit_repo.recent(db, limit=max(1, min(limit, 1000)),
                                action=action, actor=actor)
    return {"items": [e.public() for e in entries], "count": len(entries)}
