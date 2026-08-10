"""Apply migrations from code, at service startup.

There is no ``alembic.ini``: the connection string comes from the service's own
configuration, and keeping a second copy of it in an ini file is exactly how the
two drift apart. Alembic is driven through its programmatic API instead, which
also means one fewer thing to remember when deploying — "run the migrations"
is not a separate step someone can skip.

Startup is where this belongs rather than image build time: the database may not
be reachable while the image is being built, and a container that is restarted
after a schema change must migrate itself.
"""
from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

log = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
VERSION_TABLE = "rd_alembic_version"


def _config(engine: Engine) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    # Escape any '%' so ConfigParser interpolation does not eat it — ODBC
    # connection strings can legitimately contain percent-encoded characters.
    config.set_main_option("sqlalchemy.url", str(engine.url).replace("%", "%%"))
    return config


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection, opts={"version_table": VERSION_TABLE})
        return context.get_current_revision()


def head_revision() -> str | None:
    """The newest revision on disk, regardless of what the database holds."""
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    return ScriptDirectory.from_config(config).get_current_head()


def upgrade_to_head(engine: Engine) -> tuple[str | None, str | None]:
    """Bring the schema up to date. Returns ``(revision_before, revision_after)``.

    Idempotent: when the database is already at head, Alembic does nothing and
    both values match, which is what the "apply only if not applied" behaviour
    amounts to.
    """
    before = current_revision(engine)
    target = head_revision()

    if before == target:
        log.info("[DB] schema already at %s — nothing to migrate", target)
        return before, target

    log.info("[DB] migrating schema %s -> %s", before or "(empty)", target)
    config = _config(engine)
    # Hand Alembic our own connection so the migration runs on the same engine
    # (and the same pool settings) as the application, rather than opening a
    # second one with different behaviour.
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")

    after = current_revision(engine)
    log.info("[DB] schema now at %s", after)
    return before, after
