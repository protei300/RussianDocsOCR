"""PIN sign-in for the browser UI."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from service.core.auth import create_access_token, verify_pin
from service.api.deps import SESSION_USER

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class PinRequest(BaseModel):
    pin: str = Field(min_length=1, max_length=32)


@router.get("/config")
def auth_config() -> dict:
    """What the login page needs before anyone has authenticated."""
    return {"pin_required": True}


@router.post("/pin-login")
def pin_login(body: PinRequest) -> dict:
    if not verify_pin(body.pin):
        # Logged without the attempted value — writing rejected PINs to disk
        # would be its own small credential leak.
        log.warning("[API] rejected PIN sign-in attempt")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Wrong PIN")
    token = create_access_token({"sub": "operator", **SESSION_USER})
    return {"access_token": token, "token_type": "bearer", "user": SESSION_USER}
