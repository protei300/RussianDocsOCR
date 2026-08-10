"""API key management. Browser-session only — an integration cannot mint keys."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from service.api.deps import require_session
from service.core.database import DbSession, get_db
from service.repositories import api_keys as key_repo

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class KeyCreate(BaseModel):
    label: str = Field(default="", max_length=100)


@router.get("")
def list_keys(db: DbSession = Depends(get_db), _user=Depends(require_session)) -> dict:
    return {
        "items": key_repo.public_list(db),
        # Surfaced so the UI can warn rather than letting a restart quietly
        # delete keys someone pasted into a config somewhere.
        "note": "Keys created here live in ephemeral storage and are lost when "
                "the service restarts. The default key comes from the "
                "environment and always exists.",
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_key(body: KeyCreate, db: DbSession = Depends(get_db),
               _user=Depends(require_session)) -> dict:
    """Mint a key.

    ``key`` is the only time the plaintext exists outside the caller's hands —
    only its hash is stored, so it cannot be shown again.
    """
    record, plaintext = key_repo.create(db, body.label)
    log.info("[API] created API key id=%s label=%s", record.id, record.label)
    return {**record.public(), "key": plaintext,
            "warning": "Copy this key now — it will not be shown again."}


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_key(key_id: int, db: DbSession = Depends(get_db),
               _user=Depends(require_session)) -> Response:
    if key_id == key_repo.DEFAULT_KEY_ID:
        # Refused rather than silently undone by the next restart: the default
        # key is derived from the environment every boot, so "deleting" it
        # would only appear to work.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="The default key is defined by the environment and cannot be "
                   "deleted. Change DEFAULT_API_KEY and restart to rotate it.")
    if not key_repo.delete(db, key_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Key not found")
    log.info("[API] deleted API key id=%s", key_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
