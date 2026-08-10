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

from service.core.config import get_settings

#: Prefix makes keys greppable in logs and recognisable when pasted somewhere
#: they shouldn't be — the same reason GitHub uses ``ghp_``.
KEY_PREFIX = "rdk_"
KEY_PREFIX_DISPLAY_LEN = 10  # 'rdk_' + 6 chars, enough to tell keys apart


def create_access_token(payload: dict[str, Any]) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode({**payload, "exp": expire}, settings.jwt_secret,
                      algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Returns the claims, or ``None`` for anything invalid or expired."""
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
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
