"""ASGI entry point.

Run it with::

    uvicorn service.main:app --host 0.0.0.0 --port 8002 --workers 1

``--workers 1`` is **mandatory**, not a default. The in-memory document index
and the ``Pipeline`` singleton are both per-process: a second worker means two
divergent indexes and another 215 MB of models. The lifespan refuses to start
if it detects otherwise.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from service import __version__, worker
from service.api import api_keys, auth, documents, logs, settings_api, status
from service.core.auth import resolve_default_key
from service.core.config import get_settings
from service.core.database import set_store
from service.core.seed import seed_if_empty
from service.core.storage_mode import build_store
from service.core.logging import setup_logging

log = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parents[1]
PREFIX = "/api/v1"


def _guard_single_worker() -> None:
    """Refuse to run multi-worker rather than corrupt data quietly.

    Someone will eventually try to "scale" this by raising the worker count.
    Two processes means two in-memory indexes that immediately disagree and two
    copies of the model set — failures that look like flaky storage rather than
    a config mistake, so it is worth failing loudly here.
    """
    concurrency = os.environ.get("WEB_CONCURRENCY")
    if concurrency and concurrency.strip() not in ("", "1"):
        log.critical("[BOOT] WEB_CONCURRENCY=%s — this service must run with exactly "
                     "one worker (in-memory index + pipeline singleton)", concurrency)
        raise SystemExit(1)


def _announce_default_key() -> None:
    """Make the bootstrap API key impossible to miss at startup.

    Deliberately printed as a banner as well as logged. The log goes out as
    single-line JSON, which is right for aggregation but easy to scroll past in
    a terminal — and a generated key that nobody notices is a key nobody can
    use, since it exists only in this process's memory.
    """
    key, generated = resolve_default_key()
    line = "─" * 74
    if generated:
        print(f"\n┌{line}┐", file=sys.stderr)
        print("│ DEFAULT_API_KEY is not set. A random key was generated for this run:"
              .ljust(75) + "│", file=sys.stderr)
        print(f"│   {key}".ljust(75) + "│", file=sys.stderr)
        print("│".ljust(75) + "│", file=sys.stderr)
        print("│ It changes every restart, so any integration using it will break."
              .ljust(75) + "│", file=sys.stderr)
        print("│ Set DEFAULT_API_KEY in the environment for a stable key:"
              .ljust(75) + "│", file=sys.stderr)
        print(f"│   DEFAULT_API_KEY={key}".ljust(75) + "│", file=sys.stderr)
        print(f"└{line}┘\n", file=sys.stderr)
        log.warning("[BOOT] DEFAULT_API_KEY not set — generated a temporary key for this "
                    "run (%s). Set DEFAULT_API_KEY for a stable key.", key)
        return

    # Configured, but is it any good? A key that is short, or obviously a
    # placeholder someone pasted from a README, is worth flagging just as
    # loudly — it protects the upload endpoint on a machine that may be
    # reachable from the network.
    weak_markers = ("test", "demo", "example", "change", "default", "secret", "password")
    lowered = key.lower()
    reasons = []
    if len(key) < 24:
        reasons.append(f"only {len(key)} characters")
    if any(marker in lowered for marker in weak_markers):
        reasons.append("looks like a placeholder")

    if reasons:
        print(f"\n┌{line}┐", file=sys.stderr)
        print("│ DEFAULT_API_KEY looks weak: ".ljust(75) + "│", file=sys.stderr)
        for reason in reasons:
            print(f"│   - {reason}".ljust(75) + "│", file=sys.stderr)
        print("│".ljust(75) + "│", file=sys.stderr)
        print("│ Change it to a long random value. Anyone holding this key can upload"
              .ljust(75) + "│", file=sys.stderr)
        print("│ documents and read every result.".ljust(75) + "│", file=sys.stderr)
        print(f"└{line}┘\n", file=sys.stderr)
        log.warning("[BOOT] DEFAULT_API_KEY looks weak (%s) — change it",
                    "; ".join(reasons))
    else:
        log.info("[BOOT] using DEFAULT_API_KEY from the environment (%s…)", key[:10])


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    _guard_single_worker()

    log.info("[BOOT] RussianDocs service %s starting (commit=%s, python=%s)",
             __version__, settings.git_commit, sys.version.split()[0])
    if settings.jwt_secret == "changeme-in-production":
        log.warning("[BOOT] JWT_SECRET is still the default — set it before deploying")

    _announce_default_key()

    # Chooses files-vs-database, applies migrations if a database is configured,
    # and prints the consequences. Raises rather than silently downgrading when
    # a configured database is unreachable.
    store, mode = build_store()
    set_store(store)
    app.state.storage_mode = mode

    # Only when the store is empty, so a database keeps whatever the operator
    # left there and a deleted sample stays deleted.
    if settings.seed_samples >= 0:
        seeded = seed_if_empty(store, limit=settings.seed_samples or None)
        if seeded:
            log.info("[BOOT] seeded %d sample document(s) from pre-computed "
                     "results — already visible in the log, no GPU time spent",
                     seeded)

    await worker.start_worker()
    try:
        yield
    finally:
        await worker.stop_worker()
        log.info("[BOOT] service stopped")


app = FastAPI(title="RussianDocs Recognition Service", version=__version__,
              lifespan=lifespan, docs_url="/api/docs", redoc_url=None)

_origins = [o.strip() for o in get_settings().cors_allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    # Dev default covers the Vite dev server; in production the SPA is served
    # from this same origin, so no CORS is involved at all.
    allow_origins=_origins or ["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(auth.router, prefix=PREFIX)
app.include_router(documents.router, prefix=PREFIX)
app.include_router(api_keys.router, prefix=PREFIX)
app.include_router(settings_api.router, prefix=PREFIX)
app.include_router(status.router, prefix=PREFIX)
app.include_router(logs.router, prefix=PREFIX)


@app.get("/health")
def health() -> dict:
    """Liveness only — deliberately does not require the models to be loaded.

    Model loading takes ~10 s and happens in the background; gating health on
    it would fight Docker's healthcheck during every startup. Readiness of the
    recognition runtime is reported by ``/api/v1/status``.
    """
    return {"status": "ok", "version": __version__}


# --- SPA -------------------------------------------------------------------
# Served by this same process, so the browser sees one origin. Not via
# `app.mount()`: a Mount matches before route handlers and would shadow the
# API. A catch-all route runs last, which is what we want.
_web_dir = next((d for d in (_BASE_DIR / "web" / "dist", _BASE_DIR / "web")
                 if d.is_dir()), None)

if _web_dir is not None:
    _web_root = _web_dir.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = (_web_root / full_path).resolve()
        # Containment check: without it, `GET /../../etc/passwd` would escape
        # the web directory. `resolve()` collapses the traversal so the prefix
        # comparison is meaningful.
        inside = candidate == _web_root or candidate.is_relative_to(_web_root)
        if inside and candidate.is_file():
            return FileResponse(candidate)
        index = _web_root / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {"detail": "Frontend not built. Run `npm run build` in web/."}
else:
    log.warning("[BOOT] no web/ directory — API only")
