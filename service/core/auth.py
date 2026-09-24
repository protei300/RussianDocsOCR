"""Two authentication paths, for two different callers.

* **The website** signs in with a PIN and gets a short-lived JWT. This is a
  single shared operator identity — there are no user accounts.
* **Machine callers** send an API key in ``X-API-Key``. Keys are managed from
  the UI at runtime, plus one bootstrap key from the environment.

Why the split: a PIN is a human affordance and a terrible service credential
(it is four digits, it is shared, and it would have to be embedded in every
integration). An API key is the opposite. Endpoints that both kinds of caller
use accept either — see ``api/deps.py``.

Security notes, honestly:

* Key comparison uses ``secrets.compare_digest``. The PIN comparison does too,
  though against a four-digit space that is mostly symbolic — there is no rate
  limiting or lockout here, and a PIN is not a defence against an attacker who
  can reach the port. It keeps honest people out of the browser UI; the network
  boundary is the real control.
* Only key *hashes* are stored. A leaked data directory should not yield
  working credentials.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt

from service.core import passwords
from service.core.config import get_settings

#: Prefix makes keys greppable in logs and recognisable when pasted somewhere
#: they shouldn't be — the same reason GitHub uses ``ghp_``.
KEY_PREFIX = "rdk_"
KEY_PREFIX_DISPLAY_LEN = 10  # 'rdk_' + 6 chars, enough to tell keys apart


#: The value shipped in config.py. Kept here so the check below names it once.
DEFAULT_JWT_SECRET = "changeme-in-production"

_process_secret: str | None = None


def jwt_secret() -> str:
    """The signing secret actually in use.

    **A known secret is not a secret.** The default in config.py is public — it
    is in this repository — so with it anyone can mint a token. In users mode that
    is a full takeover: the administrator is uid 1 and token_version starts at 1,
    so a forged ``{"uid": 1, "tv": 1}`` is a guess, not an attack.

    So when the secret is unset or still the default, a random one is generated
    for the life of the process. The only thing it costs is that sessions do not
    survive a restart — and on this service nothing does: the store is wiped at
    every start, so a session outliving it would point at an account that no
    longer exists anyway. Set JWT_SECRET explicitly when that is not what you want.
    """
    global _process_secret
    configured = (get_settings().jwt_secret or "").strip()
    if configured and configured != DEFAULT_JWT_SECRET:
        return configured
    if _process_secret is None:
        _process_secret = secrets.token_urlsafe(48)
    return _process_secret


def jwt_secret_is_ephemeral() -> bool:
    configured = (get_settings().jwt_secret or "").strip()
    return not configured or configured == DEFAULT_JWT_SECRET


def create_access_token(payload: dict[str, Any]) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode({**payload, "exp": expire}, jwt_secret(),
                      algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Returns the claims, or ``None`` for anything invalid or expired."""
    settings = get_settings()
    try:
        # The algorithm list is pinned rather than read from the token header, so
        # a token claiming alg=none, or an asymmetric algorithm keyed with our
        # own secret, is refused rather than negotiated.
        return jwt.decode(token, jwt_secret(), algorithms=[settings.jwt_algorithm])
    except Exception:
        return None


def verify_pin(candidate: str) -> bool:
    return secrets.compare_digest(str(candidate), get_settings().auth_pin)


def generate_api_key() -> str:
    """A fresh key. Shown to the user exactly once, then only its hash remains."""
    return f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"


# --- the bootstrap key ------------------------------------------------------
# Resolved once per process. Two cases:
#
#   DEFAULT_API_KEY set    -> use it. Stable across restarts, so integrations
#                             keep working. Treated as a secret the operator
#                             already holds, so the UI shows it masked.
#   DEFAULT_API_KEY unset  -> generate a random one and log it. Nobody could
#                             know it otherwise, so the UI *does* reveal it in
#                             full; that is the deliberate trade, and it only
#                             happens when no explicit key was configured.
#
# The alternative — a constant fallback in the source — would give every
# unconfigured deployment the same publicly-known key. That is worse than
# either branch here.
_default_key: str | None = None
_default_is_generated = False


def resolve_default_key() -> tuple[str, bool]:
    """``(key, was_generated)``. Idempotent; safe to call from anywhere."""
    global _default_key, _default_is_generated
    if _default_key is None:
        configured = get_settings().default_api_key.strip()
        if configured:
            _default_key, _default_is_generated = configured, False
        else:
            _default_key, _default_is_generated = generate_api_key(), True
    return _default_key, _default_is_generated


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_prefix(key: str) -> str:
    return key[:KEY_PREFIX_DISPLAY_LEN]


# --- Which authentication mode is in force ----------------------------------
#
# Resolved here, once, and never by reading ``settings.auth_mode`` directly
# elsewhere: the whole point is that one function decides, so there is one place
# where an unusable configuration turns into a usable one.

PIN_MODE = "pin"
USERS_MODE = "users"
AUTH_MODES = (PIN_MODE, USERS_MODE)

#: Set once at startup by ``core/storage_mode.py``. Named accounts are
#: implemented for the file store only — the database backend's user methods are
#: documented stubs — so the storage choice decides whether users mode is even
#: possible, and that has to be known here rather than discovered at the first
#: login attempt.
_storage_backend: str = "files"


def configure_storage_backend(backend: str) -> None:
    global _storage_backend
    _storage_backend = backend or "files"


def resolve_auth_mode() -> tuple[str, str | None]:
    """Return ``(mode, downgrade_reason)``.

    **This function never raises and never refuses to serve.** An existing
    deployment that pulls this version must keep working exactly as before, and
    "as before" is the PIN. So every way of getting it wrong — an unset value, a
    typo, a mode whose dependency is missing — resolves to ``pin`` with a reason
    attached, rather than a service that will not start.

    The reason is returned rather than only logged so the status page and the
    startup banner can show it. A silent downgrade is the failure mode to avoid
    here: somebody configured named accounts, and if they are quietly given a
    shared four-digit PIN instead, nothing in the interface would say so.
    """
    raw = (get_settings().auth_mode or "").strip().lower()

    if not raw:
        return PIN_MODE, None                       # unset is not a mistake, it is the default
    if raw not in AUTH_MODES:
        return PIN_MODE, (f"AUTH_MODE={raw!r} is not one of {', '.join(AUTH_MODES)} — "
                          f"falling back to PIN authentication")
    if raw == USERS_MODE and not passwords.AVAILABLE:
        return PIN_MODE, ("AUTH_MODE=users needs argon2-cffi, which is not installed "
                          "(pip install -r requirements-service.txt) — "
                          "falling back to PIN authentication")
    if raw == USERS_MODE and _storage_backend != "files":
        return PIN_MODE, ("AUTH_MODE=users is implemented for the temporary file store "
                          "only; the database backend's user methods are stubs you are "
                          "expected to implement (see docs/auth.md) — "
                          "falling back to PIN authentication")
    return raw, None


def auth_mode() -> str:
    """The effective mode. Use this, not ``settings.auth_mode``."""
    return resolve_auth_mode()[0]


def users_enabled() -> bool:
    return auth_mode() == USERS_MODE


# --- Failed-login throttling -------------------------------------------------
#
# In memory, per (identity, client address), and that is honest about what it is:
# a single-process service pinned to --workers 1, so a dict is the whole
# mechanism. A real deployment behind several instances needs shared state —
# Redis, or the reverse proxy's own rate limiting — and docs/auth.md says so
# rather than leaving the reader to discover it.
#
# Why it exists at all: SECURITY.md admits the PIN has no brute-force protection,
# and four digits is 10 000 guesses. Adding named accounts without throttling
# would have made that worse, not better, because a username is a longer-lived
# secret than a PIN nobody expected to hold.

import threading as _threading
import time as _time

_attempts: dict[tuple[str, str], list[float]] = {}
_attempts_lock = _threading.Lock()


def _throttle_key(identity: str, client: str) -> tuple[str, str]:
    return ((identity or "").strip().casefold(), client or "-")


#: The key under which failures from one address are counted regardless of the
#: username tried. Not a valid username (see repositories/users.py), so it can
#: never collide with a real account's counter.
_ANY_IDENTITY = "*"

#: How many failures one address may make across ALL usernames before it is
#: blocked, as a multiple of the per-account limit. Higher than the per-account
#: limit because several people can share one address (an office NAT), lower
#: than unlimited because otherwise rotating usernames is a free pass.
ADDRESS_LIMIT_FACTOR = 3


def _blocked(key: tuple[str, str], limit: int, window: int, now: float) -> int:
    recent = [t for t in _attempts.get(key, []) if now - t < window]
    if len(recent) < limit:
        return 0
    return max(1, int(window - (now - recent[0])))


def login_blocked_for(identity: str, client: str) -> int:
    """Seconds remaining in a lockout, or 0 when the caller may try again.

    Two counters, and both are needed. **Per (account, address)** — locking by
    account alone would let anyone lock a known user out of the service by
    failing on purpose from anywhere. **Per address, across every account** —
    without it the first counter is defeated by trying a different username each
    time, which is exactly what a password-spraying run does. The first version
    had only the first counter.
    """
    settings = get_settings()
    window = settings.login_lockout_seconds
    now = _time.monotonic()
    with _attempts_lock:
        return max(
            _blocked(_throttle_key(identity, client), settings.login_max_attempts, window, now),
            _blocked(_throttle_key(_ANY_IDENTITY, client),
                     settings.login_max_attempts * ADDRESS_LIMIT_FACTOR, window, now),
        )


def note_failed_login(identity: str, client: str) -> None:
    settings = get_settings()
    window = settings.login_lockout_seconds
    now = _time.monotonic()
    with _attempts_lock:
        for key in (_throttle_key(identity, client), _throttle_key(_ANY_IDENTITY, client)):
            recent = [t for t in _attempts.get(key, []) if now - t < window]
            recent.append(now)
            _attempts[key] = recent
        # Opportunistic sweep: without it the dict grows once per distinct
        # (identity, address) pair for the life of the process.
        if len(_attempts) > 10_000:
            for stale_key, times in list(_attempts.items()):
                if not any(now - t < window for t in times):
                    _attempts.pop(stale_key, None)


def clear_failed_logins(identity: str, client: str) -> None:
    """Called after a successful sign-in, so one typo does not linger.

    Clears the account's counter only, never the address-wide one: otherwise an
    attacker holding one valid account could reset their budget by signing in
    between guesses at everyone else's.
    """
    with _attempts_lock:
        _attempts.pop(_throttle_key(identity, client), None)
