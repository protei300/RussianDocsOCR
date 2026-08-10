"""JSON logs to stdout, plus an in-memory ring buffer the UI can read.

The ring buffer exists so an operator can read recent logs from the browser
without shell access to the container — the same reasoning as the reference
project. It is capped and lossy by design; stdout remains the durable record.

Conventions used throughout the service (worth following in new code):

* ``log.exception(...)`` inside every ``except`` in a loop, so one bad document
  logs a traceback and the loop survives.
* Lazy ``%s`` formatting, never f-strings, in log calls.
* Bracketed subsystem prefixes — ``[RUNTIME]``, ``[WORKER]``, ``[STORE]``,
  ``[API]`` — so a shared stream stays greppable.
"""
from __future__ import annotations

import collections
import logging
import sys
from typing import Any

from pythonjsonlogger import json as jsonlogger

from service.core.config import get_settings

_BUFFER_SIZE = 5_000
_buffer: collections.deque = collections.deque(maxlen=_BUFFER_SIZE)


class _BufferHandler(logging.Handler):
    """Keeps the last N records in memory for ``GET /logs``."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _buffer.append({
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "exc": self.formatException(record.exc_info) if record.exc_info else None,
            })
        except Exception:
            pass  # a logging failure must never propagate into the caller


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)

    buffer_handler = _BufferHandler()
    buffer_handler.setLevel(logging.DEBUG)  # capture everything, regardless of stdout level

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stream)
    root.addHandler(buffer_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    for noisy in ("multipart", "python_multipart", "watchfiles", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_log_entries(n: int = 200, level: str | None = None,
                    search: str | None = None) -> list[dict[str, Any]]:
    """Most recent entries first, optionally filtered.

    ``level`` is a *minimum* severity, not an exact match — asking for warnings
    should show errors too.
    """
    order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    floor = order.index(level.upper()) if level and level.upper() in order else 0
    needle = search.lower() if search else None

    out: list[dict[str, Any]] = []
    for entry in reversed(_buffer):
        if order.index(entry["level"]) < floor if entry["level"] in order else False:
            continue
        if needle and needle not in entry["message"].lower():
            continue
        out.append(entry)
        if len(out) >= n:
            break
    return out
