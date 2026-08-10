"""Pick a storage backend at startup and be loud about the consequences.

Two modes, and the difference matters enough that the service announces it in a
banner rather than a log line someone will scroll past:

**Temporary (no connection string).** JSON files under ``DATA_DIR``, wiped at
every start. Zero configuration, so the service runs immediately — but every
recognised document is lost on restart. Fine for a demo, wrong for anything else.

**Database.** Any SQLAlchemy-supported dialect; MS SQL Server and PostgreSQL are
the tested ones. Migrations are applied automatically if not already applied.

The rule that ties them together: **``DATA_DIR`` is only wiped in temporary
mode.** With a database the rows outlive the process, so wiping the images would
leave every stored document pointing at a file that no longer exists. That
directory therefore has to be persistent (a Docker volume) whenever a connection
string is configured, and this module enforces the wipe half of it.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from service.core.config import get_settings

log = logging.getLogger(__name__)

_LINE = "─" * 74


@dataclass
class StorageMode:
    backend: str                  # 'files' | 'sql'
    ephemeral: bool
    wipe_on_start: bool
    dialect: str | None = None
    migrated_from: str | None = None
    migrated_to: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ephemeral": self.ephemeral,
            "dialect": self.dialect,
            "schema_revision": self.migrated_to,
            "error": self.error,
        }


def _banner(lines: list[str]) -> None:
    print(f"\n┌{_LINE}┐", file=sys.stderr)
    for line in lines:
        print(f"│ {line}".ljust(75) + "│", file=sys.stderr)
    print(f"└{_LINE}┘\n", file=sys.stderr)


def build_store() -> tuple[Any, StorageMode]:
    """Construct the configured store. Returns ``(store, mode)``.

    Never silently downgrades: if a connection string is present but the
    database cannot be reached, this raises. Falling back to temporary storage
    would look like it worked and quietly discard the data the operator went to
    the trouble of configuring a database for.
    """
    settings = get_settings()
    url = settings.database_connectionstring.strip()
    data_dir = Path(settings.data_dir).resolve()

    if not url:
        from service.core.database import init_store
        _banner([
            "RUSSIANDOCS_DATABASE_CONNECTIONSTRING is not set.",
            "",
            "Running with TEMPORARY storage: documents and results are written to",
            f"  {data_dir}",
            "and that directory is ERASED on every service start. Nothing you",
            "recognise here survives a restart.",
            "",
            "Set a connection string to store data properly. Examples:",
            "  MS SQL Server",
            "    RUSSIANDOCS_DATABASE_CONNECTIONSTRING=mssql+pyodbc://user:pass@host:1433/",
            "      russiandocs?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes",
            "  PostgreSQL",
            "    RUSSIANDOCS_DATABASE_CONNECTIONSTRING=postgresql+psycopg://user:pass@host:5432/russiandocs",
            "",
            "The schema is created automatically on first connect.",
        ])
        log.warning("[STORE] no connection string — using TEMPORARY storage in %s "
                    "(erased on every start). Set "
                    "RUSSIANDOCS_DATABASE_CONNECTIONSTRING to persist data.", data_dir)
        store = init_store(data_dir, wipe=settings.data_wipe_on_start)
        return store, StorageMode(backend="files", ephemeral=True,
                                  wipe_on_start=settings.data_wipe_on_start)

    from service.core.db_sql import SqlStore
    from service.core.db_sql_migrate import upgrade_to_head

    # Artifacts are never wiped in this mode — see the module docstring.
    data_dir.mkdir(parents=True, exist_ok=True)
    store = SqlStore(url, data_dir)
    dialect = store.dialect()
    log.info("[STORE] connecting to %s database", dialect)

    before, after = upgrade_to_head(store.engine)
    if before != after:
        log.info("[STORE] applied migrations: %s -> %s", before or "(empty)", after)

    log.info("[STORE] using %s; artifacts in %s (NOT wiped — the rows outlive "
             "the process, so the images must too)", dialect, data_dir)
    return store, StorageMode(backend="sql", ephemeral=False, wipe_on_start=False,
                              dialect=dialect, migrated_from=before, migrated_to=after)
