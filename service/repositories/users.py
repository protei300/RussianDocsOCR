"""User accounts: the whole lifecycle, and the rules that keep it usable.

Only reachable when ``AUTH_MODE=users``. In PIN mode nothing here is called and
the store stays empty — see ``api/users.py``, which answers 404 rather than
returning an empty list.

**The rules worth stating before the code, because each is the kind that only
bites in production.**

*You cannot lock everyone out.* Deleting, deactivating or demoting the last
active administrator is refused. It sounds like an edge case until someone
demotes themselves to "viewer" to test the role and discovers there is no longer
an account that can undo it.

*That check is only a check if it is atomic with the change.* Two administrators
demoting each other at the same moment each see the other still active, both
checks pass, and the service ends with none. So every mutation below runs under
one lock and re-reads the account inside it: check and change are a single step,
not two with a gap between them. The first version had the gap — and handed out
live objects from the store, so the gap was also a window in which one request
could see another's half-applied edit.

*Changing anything that affects authority bumps the token version.* Password,
role, active flag — each one invalidates every token already issued for that
account. Skipping it on a role change is the subtle case: a demoted
administrator otherwise keeps administrator rights for the rest of the token's
eight-hour life, and nothing in the UI hints at it.

*Usernames are ASCII.* A name is an identity, and Unicode lets two different
identities look the same: "admin" and "аdmin" (Cyrillic а) pass a case-insensitive
uniqueness check and are indistinguishable on screen. Display names can be
anything — they are for reading, not for trusting.

Signatures follow the project's repository contract (``db`` first, plain
functions), so a SQL-backed implementation is a body swap. There the lock
becomes a transaction — ``SELECT … FOR UPDATE`` on the rows involved, or a
serialisable transaction around the admin count — which is spelled out next to
the DDL in ``core/db_sql.py``.
"""
from __future__ import annotations

import re
import threading

from service.core import passwords
from service.core.database import FileStore
from service.core.models import ROLES, User, utcnow

#: One lock for every change to an account. Re-entrant because a mutation may
#: call a helper that also takes it. Logins do not hold it while hashing — see
#: ``authenticate`` — so a slow Argon2 verification never blocks administration.
_WRITE = threading.RLock()

#: Letters, digits, dot, underscore, hyphen; starting with a letter or digit.
_USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class UserError(Exception):
    """A rule was violated. The message is safe to show the caller."""


def get_all(db: FileStore) -> list[User]:
    return db.all_users()


def get(db: FileStore, user_id: int) -> User | None:
    return db.get_user(user_id)


def find(db: FileStore, username: str) -> User | None:
    return db.find_user(username)


def count_active_admins(db: FileStore, *, excluding: int | None = None) -> int:
    return sum(1 for u in db.all_users()
               if u.role == "admin" and u.is_active and u.id != excluding)


def _require_not_last_admin(db: FileStore, user: User, what: str) -> None:
    if user.role == "admin" and user.is_active and count_active_admins(db, excluding=user.id) == 0:
        raise UserError(f"Cannot {what} the last active administrator")


def _normalise_username(username: str) -> str:
    name = (username or "").strip()
    if not name:
        raise UserError("Username is required")
    if not _USERNAME.match(name):
        raise UserError("Username may contain only Latin letters, digits, '.', '_' and '-', "
                        "must start with a letter or digit, and be at most 64 characters")
    return name


def _fresh(db: FileStore, user_id: int) -> User:
    """Re-read inside the lock. The copy the caller holds may already be stale."""
    user = db.get_user(user_id)
    if user is None:
        raise UserError("No such user")
    return user


def create(db: FileStore, *, username: str, password: str, role: str,
           display_name: str = "", must_change_password: bool = True) -> User:
    name = _normalise_username(username)
    if role not in ROLES:
        raise UserError(f"Unknown role {role!r}")
    complaint = passwords.validate_password(password)
    if complaint:
        raise UserError(complaint)
    # Hashed before taking the lock: it is the slow part, and it does not depend
    # on anything the lock protects.
    password_hash = passwords.hash_password(password)

    with _WRITE:
        # Uniqueness and id allocation are checked and used in one step, or two
        # concurrent creations of "petrov" would both succeed.
        if db.find_user(name) is not None:
            raise UserError(f"User {name!r} already exists")
        user = User(
            id=db.next_user_id(),
            username=name,
            role=role,
            display_name=(display_name or "").strip(),
            password_hash=password_hash,
            must_change_password=must_change_password,
        )
        return db.put_user(user)


def seed_admin(db: FileStore, *, username: str, password: str) -> User | None:
    """Create the bootstrap administrator when there are no users at all.

    Returns the created user, or ``None`` when users already exist.

    This is the one place that bypasses the password policy, and it has to: the
    default password is printed on the login page precisely so the demo can be
    used, and it fails every composition rule. ``must_change_password`` is what
    makes that defensible — the account can do nothing else until it is changed.
    """
    name = _normalise_username(username)
    password_hash = passwords.hash_password(password)
    with _WRITE:
        if db.all_users():
            return None
        user = User(
            id=db.next_user_id(),
            username=name,
            role="admin",
            display_name="Administrator",
            password_hash=password_hash,
            must_change_password=True,
        )
        return db.put_user(user)


def authenticate(db: FileStore, username: str, password: str) -> User | None:
    """Verify credentials. ``None`` for wrong user, wrong password or disabled.

    One return value for all three on purpose: an endpoint that distinguished
    "no such user" from "wrong password" would let anyone enumerate accounts.
    The password is still verified for a missing user — against a throwaway hash
    — so the response time does not reveal which case it was.

    The verification runs **outside** the write lock (it is the slow part); only
    the bookkeeping afterwards takes it, and it re-reads the account first. Writing
    back the copy loaded before hashing would silently undo anything an
    administrator changed during those ~80 ms — a demotion, say.
    """
    user = db.find_user(username)
    if user is None:
        passwords.verify_password(_TIMING_DECOY, password)
        return None
    if not passwords.verify_password(user.password_hash, password):
        return None

    rehash = passwords.hash_password(password) if passwords.needs_rehash(user.password_hash) else None
    with _WRITE:
        fresh = db.get_user(user.id)
        # Re-checked on the fresh copy: the account may have been disabled, or
        # its password changed, while this request was busy hashing.
        if fresh is None or not fresh.is_active or fresh.password_hash != user.password_hash:
            return None
        if rehash:
            fresh.password_hash = rehash
        fresh.last_login_at = utcnow()
        return db.put_user(fresh)


def change_password(db: FileStore, user: User, new_password: str,
                    *, current_password: str | None = None) -> User:
    """Set a new password. Bumps the token version, killing existing sessions."""
    complaint = passwords.validate_password(new_password)
    if complaint:
        raise UserError(complaint)
    new_hash = passwords.hash_password(new_password)

    with _WRITE:
        fresh = _fresh(db, user.id)
        if current_password is not None and not passwords.verify_password(
                fresh.password_hash, current_password):
            raise UserError("Current password is incorrect")
        if passwords.verify_password(fresh.password_hash, new_password):
            raise UserError("The new password must differ from the current one")
        fresh.password_hash = new_hash
        fresh.must_change_password = False
        fresh.token_version += 1
        return db.put_user(fresh)


def reset_password(db: FileStore, user: User, new_password: str) -> User:
    """Administrator sets someone else's password; they must change it at next login."""
    complaint = passwords.validate_password(new_password)
    if complaint:
        raise UserError(complaint)
    new_hash = passwords.hash_password(new_password)
    with _WRITE:
        fresh = _fresh(db, user.id)
        fresh.password_hash = new_hash
        fresh.must_change_password = True
        fresh.token_version += 1
        return db.put_user(fresh)


def update(db: FileStore, user: User, *, role: str | None = None,
           display_name: str | None = None, is_active: bool | None = None) -> User:
    """Change role, name or active flag. Any authority change invalidates tokens."""
    if role is not None and role not in ROLES:
        raise UserError(f"Unknown role {role!r}")

    with _WRITE:
        fresh = _fresh(db, user.id)
        authority_changed = False

        if role is not None and role != fresh.role:
            if role != "admin":
                _require_not_last_admin(db, fresh, "demote")
            fresh.role = role
            authority_changed = True

        if is_active is not None and is_active != fresh.is_active:
            if not is_active:
                _require_not_last_admin(db, fresh, "deactivate")
            fresh.is_active = is_active
            authority_changed = True

        if display_name is not None:
            fresh.display_name = display_name.strip()

        if authority_changed:
            fresh.token_version += 1
        return db.put_user(fresh)


def delete(db: FileStore, user: User, *, acting_user_id: int | None = None) -> None:
    if acting_user_id is not None and user.id == acting_user_id:
        # Not a safety rule so much as a usability one: there is no undo, and
        # deleting the account you are signed in as is never what was meant.
        raise UserError("You cannot delete your own account")
    with _WRITE:
        fresh = _fresh(db, user.id)
        _require_not_last_admin(db, fresh, "delete")
        db.drop_user(fresh.id)


#: A real Argon2 hash of a value nobody knows, used only to spend the same time
#: verifying a password for a username that does not exist. Computed once at
#: import: hashing on every failed login would itself be a timing signal.
_TIMING_DECOY = (passwords.hash_password("no-such-user-timing-decoy")
                 if passwords.AVAILABLE else "")
