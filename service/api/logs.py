"""Recent log lines, so an operator can diagnose without shell access."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from service.api.deps import require_session
from service.core.logging import get_log_entries

router = APIRouter(tags=["logs"])


@router.get("/logs")
def read_logs(
    n: int = Query(200, ge=1, le=2000),
    level: str | None = Query(None),
    search: str | None = Query(None),
    _user=Depends(require_session),
) -> dict:
    entries = get_log_entries(n=n, level=level, search=search)
    return {"count": len(entries), "entries": entries}
