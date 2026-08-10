"""Runtime settings — the schema travels with the values, so the UI self-renders."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from service.api.deps import require_session
from service.core.database import DbSession, get_db
from service.repositories import settings as settings_repo
from service.core.settings_schema import SettingValidationError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    values: dict[str, Any]


@router.get("")
def get_settings_values(db: DbSession = Depends(get_db),
                        _user=Depends(require_session)) -> dict:
    return {"values": settings_repo.get_all(db), "schema": settings_repo.schema()}


@router.put("")
def update_settings(body: SettingsUpdate, db: DbSession = Depends(get_db),
                    _user=Depends(require_session)) -> dict:
    """Validate and store.

    ``restart_required`` names settings baked into ``Pipeline.__init__`` that
    changed — the UI must say so rather than implying they took effect.
    """
    try:
        values, restart_required = settings_repo.bulk_update(db, body.values)
    except SettingValidationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    if restart_required:
        log.info("[API] settings changed, restart required for: %s",
                 ", ".join(restart_required))
    return {"values": values, "schema": settings_repo.schema(),
            "restart_required": restart_required}
