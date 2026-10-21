"""The recognition journal: one line per ``POST /recognize`` call, and nothing about the document.

A line says **that** something was recognised, how it ended and how long it took:
id, time, outcome, HTTP status, milliseconds. No filename, no document type, no
size, no field, no caller, no client address — the journal is the one record of an
in-memory recognition, and it must be safe to show to anyone who can see the
service at all. The UI lists these lines but cannot open them: there is nothing
behind them to open.

Kept **in memory only**, capped, lost at restart. Writing it to the store would
be harmless content-wise, but the point of the mode it serves is that a recognition
leaves nothing on disk; a journal file would be the one exception, and an
exception is what an auditor asks about first.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from service.core.config import get_settings

#: Outcomes, as the UI shows them. ``recognised`` = a document type was found;
#: ``not_recognised`` = the image was read but no supported document was in it;
#: ``rejected`` = the request was refused before recognition (bad or oversized
#: image); ``busy`` = no pipeline became free in time; ``error`` = anything else.
OUTCOMES = ("recognised", "not_recognised", "rejected", "busy", "error")


@dataclass(frozen=True)
class RecognitionEvent:
    id: int
    outcome: str
    status_code: int
    processing_ms: int
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def public(self) -> dict:
        return {
            "id": self.id,
            "at": self.at.isoformat().replace("+00:00", "Z"),
            "outcome": self.outcome,
            "status_code": self.status_code,
            "processing_ms": self.processing_ms,
            "stored": False,          # the UI keys "no drill-down" off this
        }


_lock = threading.Lock()
_events: deque[RecognitionEvent] = deque(maxlen=5000)
_next_id = 1


def record(outcome: str, status_code: int, processing_ms: int) -> RecognitionEvent:
    global _events, _next_id
    cap = max(1, get_settings().recognition_journal_max)
    with _lock:
        if _events.maxlen != cap:
            _events = deque(_events, maxlen=cap)
        event = RecognitionEvent(id=_next_id, outcome=outcome, status_code=status_code,
                                 processing_ms=max(0, int(processing_ms)))
        _next_id += 1
        _events.append(event)
        return event


def recent(limit: int = 200, offset: int = 0) -> tuple[list[RecognitionEvent], int]:
    """Newest first. Returns ``(page, total)``."""
    with _lock:
        rows = list(_events)
    rows.reverse()
    return rows[offset:offset + max(0, limit)], len(rows)


def counts() -> dict[str, int]:
    with _lock:
        rows = list(_events)
    out = {o: 0 for o in OUTCOMES}
    for e in rows:
        out[e.outcome] = out.get(e.outcome, 0) + 1
    out["total"] = len(rows)
    return out


def reset_for_tests() -> None:
    global _events, _next_id
    with _lock:
        _events = deque(maxlen=max(1, get_settings().recognition_journal_max))
        _next_id = 1
