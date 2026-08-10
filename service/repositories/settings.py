"""Runtime settings: stored as strings, validated against the schema.

The worker reads these fresh on every loop iteration, so an operator change
takes effect without a restart — except for the ones flagged
``restart_required``, which are baked into ``Pipeline.__init__``.
"""
from __future__ import annotations

import logging
from typing import Any

from service.core.config import get_settings
from service.core.database import FileStore
from service.core.settings_schema import (
    SCHEMA_BY_KEY, SETTINGS_SCHEMA, UI_KEYS, SettingValidationError, coerce, typed,
)

log = logging.getLogger(__name__)


def effective_default(key: str) -> str:
    """The default for ``key`` **after** the environment has had its say.

    Precedence is *stored value → environment → schema default*, and it is
    resolved here so no caller can get it wrong. Every schema key that is also
    configurable by env deliberately shares its name with the ``Settings`` field
    (``compute_device``, ``ocr_mode``, ``docconf``, ``img_size``,
    ``job_timeout_sec``, ``max_retries``, ``log_level``), so the two tiers line
    up by construction rather than by a hand-maintained table.

    This layering was previously missing in two different ways, and both were
    real: the worker's value ignored the environment entirely, so
    ``COMPUTE_DEVICE=cpu`` was logged and then disregarded; and the settings page
    read the schema default, so it would have displayed ``auto`` for a service
    actually running on CPU.
    """
    definition = SCHEMA_BY_KEY[key]
    env_value = getattr(get_settings(), key, None)
    if env_value is None or env_value == "":
        return definition.default
    try:
        return coerce(key, env_value)
    except SettingValidationError:
        # A bad env value must not take the service down, but silence would hide
        # a deployment mistake behind a plausible default.
        log.warning("[SETTINGS] ignoring invalid %s=%r from the environment; "
                    "using %r", key.upper(), env_value, definition.default)
        return definition.default


def get_all(db: FileStore) -> dict[str, str]:
    """Current values, with environment-or-schema defaults for anything unset."""
    stored = db.all_settings()
    return {d.key: stored.get(d.key, effective_default(d.key)) for d in SETTINGS_SCHEMA}


def get_value(db: FileStore, key: str, fallback: Any = None) -> Any:
    """One typed value. The worker's accessor.

    ``fallback`` applies only to keys that are not in the schema at all; for
    known keys the environment layer above is authoritative, precisely so a
    caller passing the wrong fallback cannot desync the runtime from the
    settings page.
    """
    if key not in SCHEMA_BY_KEY:
        return fallback
    stored = db.all_settings().get(key)
    return typed(key, stored if stored is not None else effective_default(key))


def bulk_update(db: FileStore, values: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Validate and store. Returns ``(all values, keys needing a restart)``.

    Unknown keys are dropped silently (the whitelist); known keys with bad
    values raise, because a UI that reports "saved" while discarding the value
    is worse than an error.
    """
    accepted: dict[str, str] = {}
    restart_required: list[str] = []
    current = db.all_settings()

    for key, value in values.items():
        if key not in UI_KEYS:
            continue
        normalised = coerce(key, value)         # raises SettingValidationError
        accepted[key] = normalised
        definition = SCHEMA_BY_KEY[key]
        previous = current.get(key, definition.default)
        if definition.restart_required and normalised != previous:
            restart_required.append(key)

    if accepted:
        db.set_settings(accepted)
    return get_all(db), restart_required


def schema() -> list[dict[str, Any]]:
    return [d.as_dict() for d in SETTINGS_SCHEMA]


__all__ = ["get_all", "get_value", "effective_default", "bulk_update", "schema",
           "SettingValidationError"]
