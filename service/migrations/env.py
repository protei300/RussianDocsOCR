"""Alembic environment.

Built to be driven from code rather than the ``alembic`` CLI — the service
applies migrations itself at startup (``core.db_sql_migrate.upgrade_to_head``),
so there is no ``alembic.ini`` and no sqlalchemy.url to keep in sync. The URL
always comes from the connection string the service was configured with.
"""
from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from service.core.db_sql import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it — for review, not for use."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)
    if connectable is None:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        with connectable.connect() as connection:
            _run(connection)
        connectable.dispose()
    else:
        # Re-using the service's own engine connection, which is what the
        # programmatic path passes in.
        _run(connectable)


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # Keep the version table alongside the service's own tables rather than
        # in whatever the default schema happens to be — on a shared corporate
        # SQL Server that matters.
        version_table="rd_alembic_version",
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
