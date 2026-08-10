"""Server-owned schema for runtime-tunable settings.

The server describes its own knobs — type, bounds, choices, help text, group —
and the UI renders itself from that. The alternative (a hand-written form) means
every new pipeline knob is a frontend change, and the defaults end up duplicated
on both sides where they drift.

Two properties matter beyond convenience:

* ``restart_required`` marks settings baked into ``Pipeline.__init__``. Changing
  ``ocr_mode`` in the UI cannot affect a pipeline that is already built, and
  silently pretending otherwise is worse than saying so.
* Values are stored as strings (the store is JSON, and SQL would use a
  key/value table). Coercion and validation happen here, in one place, on the
  way in — the reference project's whitelist accepts ``poll_interval=banana``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class SettingDef:
    key: str
    type: str                      # bool | int | float | choice | str
    default: str
    label: str
    description: str
    group: str = "General"
    min_value: float | None = None
    max_value: float | None = None
    choices: tuple[str, ...] | None = None
    restart_required: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["choices"] = list(self.choices) if self.choices else None
        return data


SETTINGS_SCHEMA: tuple[SettingDef, ...] = (
    SettingDef(
        "compute_device", "choice", "auto", "Compute device",
        "GPU is used only when onnxruntime reports a CUDA provider AND the "
        "pipeline actually builds on it. Applied at startup.",
        group="Recognition", choices=("auto", "cpu", "gpu"), restart_required=True,
    ),
    SettingDef(
        "ocr_mode", "choice", "accurate", "OCR engine",
        "'accurate' is MobileNetV4 (best quality); 'fast' is EdgeNext. "
        "Baked into the pipeline at construction.",
        group="Recognition", choices=("accurate", "fast"), restart_required=True,
    ),
    SettingDef(
        "docconf", "float", "0.5", "Document confidence threshold",
        "Minimum confidence for accepting a detected document type.",
        group="Recognition", min_value=0.0, max_value=1.0,
    ),
    SettingDef(
        "img_size", "int", "1500", "Processing image size",
        "Longest side the image is scaled to before inference. Only ever "
        "downscales — a smaller upload is not enlarged.",
        group="Recognition", min_value=640, max_value=2560,
    ),
    SettingDef(
        "job_timeout_sec", "int", "120", "Job timeout (seconds)",
        "Typical processing is well under one second; this is a wedge detector, "
        "not a performance limit.",
        group="Queue", min_value=10, max_value=600,
    ),
    SettingDef(
        "max_retries", "int", "2", "Max retries",
        "Applies to transient failures only. A corrupt image fails immediately "
        "and is never retried.",
        group="Queue", min_value=0, max_value=5,
    ),
    SettingDef(
        "log_level", "choice", "INFO", "Log level", "",
        group="Service", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    ),
)

SCHEMA_BY_KEY = {d.key: d for d in SETTINGS_SCHEMA}
#: The write whitelist, derived rather than duplicated.
UI_KEYS = frozenset(SCHEMA_BY_KEY)


class SettingValidationError(ValueError):
    pass


def coerce(key: str, value: Any) -> str:
    """Validate against the schema and normalise to the stored string form."""
    definition = SCHEMA_BY_KEY.get(key)
    if definition is None:
        raise SettingValidationError(f"unknown setting '{key}'")

    raw = str(value).strip()
    if definition.type == "bool":
        return "1" if raw.lower() in ("1", "true", "yes", "on") else "0"

    if definition.type in ("int", "float"):
        try:
            number = float(raw)
        except ValueError:
            raise SettingValidationError(f"{key} must be a number, got '{raw}'") from None
        if definition.min_value is not None and number < definition.min_value:
            raise SettingValidationError(f"{key} must be >= {definition.min_value}")
        if definition.max_value is not None and number > definition.max_value:
            raise SettingValidationError(f"{key} must be <= {definition.max_value}")
        return str(int(number)) if definition.type == "int" else str(number)

    if definition.type == "choice":
        if definition.choices and raw not in definition.choices:
            raise SettingValidationError(
                f"{key} must be one of {', '.join(definition.choices)}")
        return raw

    return raw


def typed(key: str, stored: str | None) -> Any:
    """Stored string -> the Python value the worker wants."""
    definition = SCHEMA_BY_KEY.get(key)
    if definition is None:
        return stored
    raw = stored if stored is not None else definition.default
    try:
        if definition.type == "bool":
            return str(raw).lower() in ("1", "true", "yes", "on")
        if definition.type == "int":
            return int(float(raw))
        if definition.type == "float":
            return float(raw)
    except (TypeError, ValueError):
        # A malformed stored value must not take the worker down; fall back to
        # the schema default and carry on.
        return typed(key, definition.default) if raw != definition.default else raw
    return str(raw)
