"""Which processing mode is in force: documents stored, or recognised in memory only.

``PROCESSING_MODE=stored`` (the default) is the service as it always was.
``PROCESSING_MODE=memory`` is the mode for deployments that must not keep personal
data at all: the document store, the background worker and every document route
are switched off, and ``POST /recognize`` is the only way a document gets in —
decoded, recognised and answered inside the request, with nothing written.

**Resolution never fails, and when the value is wrong it fails toward privacy.**
That is the opposite of ``AUTH_MODE``, which falls back to the old behaviour, and
the difference is deliberate. Somebody who typed ``PROCESSING_MODE=memroy`` asked
for documents not to be stored; quietly storing them would break the one promise
the setting exists to make, and nothing in the interface would show it. So:

    unset, empty   -> stored   (backward compatible: the default is not a mistake)
    "stored"       -> stored
    "memory"       -> memory
    anything else  -> memory, with a reason that is logged and shown on /status

Like the auth mode, it is configuration only — never a runtime setting: a switch
on a web page would let anyone with admin rights turn storage back on for a
service that promised not to store.
"""
from __future__ import annotations

from service.core.config import get_settings

STORED = "stored"
MEMORY = "memory"
MODES = (STORED, MEMORY)


def resolve_processing_mode() -> tuple[str, str | None]:
    """``(mode, reason)`` — reason is set only when the configured value was not used."""
    raw = (get_settings().processing_mode or "").strip().lower()
    if not raw:
        return STORED, None
    if raw in MODES:
        return raw, None
    return MEMORY, (f"PROCESSING_MODE={raw!r} is not one of {', '.join(MODES)} — using "
                    f"'memory' (documents are not stored), the safe reading of a mistyped "
                    f"privacy setting")


def processing_mode() -> str:
    return resolve_processing_mode()[0]


def storage_enabled() -> bool:
    return processing_mode() == STORED
