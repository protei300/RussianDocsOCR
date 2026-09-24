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
from fastapi.responses import FileResponse, HTMLResponse

from service import __version__, worker
from service.api import api_keys, auth, documents, logs, settings_api, status, users
from service.core import auth as auth_core
from service.core.auth import resolve_default_key
from service.core.config import DEFAULT_ADMIN_PASSWORD, get_settings
from service.core.database import set_store
from service.core.seed import seed_if_empty
from service.repositories import users as user_repo
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


def _announce_auth_mode(store) -> None:
    """Say which authentication is in force, and seed the first account.

    Loud on purpose, and loudest when the configuration was not honoured. A
    downgrade from named accounts to a shared four-digit PIN is precisely the
    thing nobody notices from the outside: the service works, the login page
    looks plausible, and the operator believes the accounts they configured are
    in effect.
    """
    effective, downgrade_reason = auth_core.resolve_auth_mode()
    if downgrade_reason:
        log.warning("[AUTH] %s", downgrade_reason)

    if effective != auth_core.USERS_MODE:
        log.info("[AUTH] PIN sign-in; user accounts are disabled "
                 "(set AUTH_MODE=users to enable them)")
        return

    settings = get_settings()
    try:
        created = user_repo.seed_admin(store, username=settings.admin_username,
                                       password=settings.admin_password)
    except Exception:
        # Never fatal: a service that will not boot because it could not seed a
        # demo account is worse than one that boots with no accounts and says so.
        log.exception("[AUTH] could not seed the first administrator")
        return

    log.info("[AUTH] named accounts (AUTH_MODE=users)")
    if created is not None:
        # The password is printed only when it is the documented demo value. An
        # earlier version logged settings.admin_password unconditionally, which
        # wrote a real ADMIN_PASSWORD into every log collector the service feeds.
        shown = (repr(settings.admin_password)
                 if settings.admin_password == DEFAULT_ADMIN_PASSWORD
                 else "from ADMIN_PASSWORD (not logged)")
        log.warning("[AUTH] seeded the first administrator %r, password %s — it must "
                    "be changed at first sign-in, and it is re-created after every "
                    "restart because this store is temporary", created.username, shown)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    _guard_single_worker()

    log.info("[BOOT] RussianDocs service %s starting (commit=%s, python=%s)",
             __version__, settings.git_commit, sys.version.split()[0])
    if auth_core.jwt_secret_is_ephemeral():
        # Not a warning to ignore any more: the default is no longer *used*. A
        # random secret is generated per process instead, so a public default can
        # never sign a token — the cost is only that sessions end at restart.
        log.warning("[BOOT] JWT_SECRET is unset or still the published default — using a "
                    "random per-process secret instead; every session ends when the "
                    "service restarts. Set JWT_SECRET to keep sessions across restarts.")

    _announce_default_key()

    # Chooses files-vs-database, applies migrations if a database is configured,
    # and prints the consequences. Raises rather than silently downgrading when
    # a configured database is unreachable.
    store, mode = build_store()
    set_store(store)
    app.state.storage_mode = mode

    # Named accounts exist for the file store only, so the storage decision has
    # to reach the auth layer before the mode is resolved — otherwise a database
    # deployment would accept AUTH_MODE=users and fail at the first login.
    auth_core.configure_storage_backend(mode.backend)
    _announce_auth_mode(store)

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
app.include_router(users.router, prefix=PREFIX)
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
#
# Only ``web/dist`` is ever served. There used to be a fallback to ``web/``
# itself, and it was the defect: ``web/`` is the SOURCE tree, it is tracked, so
# it exists in every clone — which meant the "not built" message below could
# never be reached in the situation it names. What the visitor got instead was
# the source ``index.html``, whose `<script type="module" src="/src/main.ts">`
# the browser refuses to execute (Python serves `.ts` as
# `video/vnd.dlna.mpeg-tts`, and module scripts are type-checked strictly),
# while `/favicon.svg` and the two stylesheets live in ``web/public/`` and
# resolve to `index.html` instead. Net effect: a blank page, no diagnosis, and
# a working API nobody could tell was working.
#
# Do not restore the fallback as a kindness. The source tree is never a
# servable artifact: vite writes the build to ``dist`` (``web/vite.config.ts``,
# ``build.outDir``), the Dockerfile copies it to ``web/dist`` and asserts
# ``web/dist/index.html`` exists, and during development the SPA is served by
# vite itself on port 8000 with a proxy to this API. There is no configuration
# in which serving ``web/`` produces a working page.
_web_root = (_BASE_DIR / "web" / "dist").resolve()

_UNBUILT_HINT = "Frontend not built. Run `npm run build` in web/."
_UNBUILT_PAGE = f"""<!doctype html>
<meta charset="utf-8">
<title>RussianDocs — frontend not built</title>
<style>
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 42rem;
         margin: 4rem auto; padding: 0 1.5rem; }}
  code {{ background: #f4f4f5; padding: .15em .4em; border-radius: .25rem; }}
</style>
<h1>The web interface is not built</h1>
<p>The API is running and fully usable — this page is only about the UI.</p>
<p>Build it once:</p>
<pre><code>cd web
npm install
npm run build</code></pre>
<p>Then reload this page. The build is written to <code>web/dist/</code>, which
is deliberately not stored in the repository.</p>
<p>API docs: <a href="/docs">/docs</a> &middot; health: <a href="/health">/health</a></p>
"""

if not (_web_root / "index.html").is_file():
    log.warning("[BOOT] %s Looked in %s. The API works; the UI will answer "
                "with build instructions.", _UNBUILT_HINT, _web_root)

# **index.html must always be revalidated.** Its asset references carry
# content hashes, so the bundles under /assets/ can be cached forever — but the
# document naming them cannot. A browser that keeps a stale index.html runs an
# old application against a new server, which is how a rebuilt UI still shows
# the previous one's login screen. That is not hypothetical: it happened here,
# with the PIN keypad surviving the switch to named accounts. Applied to BOTH
# places index.html is returned: the fallback and a direct hit on the file.
_INDEX_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    # Checked per request, not at import: building while the server runs then
    # takes effect on the next reload instead of requiring a restart.
    index = _web_root / "index.html"
    if not index.is_file():
        return HTMLResponse(_UNBUILT_PAGE, status_code=503)
    candidate = (_web_root / full_path).resolve()
    # Containment check: without it, `GET /../../etc/passwd` would escape
    # the web directory. `resolve()` collapses the traversal so the prefix
    # comparison is meaningful.
    inside = candidate == _web_root or candidate.is_relative_to(_web_root)
    if inside and candidate.is_file():
        if candidate == index.resolve():
            return FileResponse(candidate, headers=_INDEX_HEADERS)
        return FileResponse(candidate)
    return FileResponse(index, headers=_INDEX_HEADERS)
