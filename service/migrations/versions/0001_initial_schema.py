"""Initial schema: documents, api keys, settings.

Revision ID: 0001
Revises:
Create Date: 2026-08-03

Deliberately dialect-neutral. Only generic SQLAlchemy types and ``op.*``
operations appear here — no raw SQL, no server defaults, no dialect kwargs — so
the same migration produces the right thing on MS SQL Server and PostgreSQL
alike. ``String(n)`` becomes ``NVARCHAR(n)`` or ``VARCHAR(n)``, ``Text``
becomes ``NVARCHAR(MAX)`` or ``TEXT``, and the dialect layer decides.

Timestamps are naive UTC on purpose: not every dialect has a portable
timezone-aware type, and mixing aware with naive values is a reliable source of
comparison bugs. The application converts at the boundary
(``db_sql._naive_utc`` / ``_aware_utc``).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rd_documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),

        sa.Column("doc_type", sa.String(64), nullable=True),
        sa.Column("doc_conf", sa.Float(), nullable=True),
        sa.Column("recognised", sa.Boolean(), nullable=False),
        sa.Column("field_count", sa.Integer(), nullable=False),
        sa.Column("quality_json", sa.Text(), nullable=True),

        sa.Column("device", sa.String(16), nullable=True),
        sa.Column("processing_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(40), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),

        sa.Column("original_ext", sa.String(16), nullable=False),
        sa.Column("original_w", sa.Integer(), nullable=True),
        sa.Column("original_h", sa.Integer(), nullable=True),
        sa.Column("canvas_w", sa.Integer(), nullable=True),
        sa.Column("canvas_h", sa.Integer(), nullable=True),
        sa.Column("has_canvas", sa.Boolean(), nullable=False),

        sa.Column("search_text", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),

        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    # The list page's default ordering.
    op.create_index("ix_rd_documents_created_at", "rd_documents", ["created_at"])
    # Status is both a filter and the status-page aggregate.
    op.create_index("ix_rd_documents_status", "rd_documents", ["status"])
    op.create_index("ix_rd_documents_doc_type", "rd_documents", ["doc_type"])
    # The worker's hot query — "oldest queued document" — runs on every drain
    # loop iteration. A composite index answers it from the index alone.
    op.create_index("ix_rd_documents_queue", "rd_documents", ["status", "created_at"])

    op.create_table(
        "rd_api_keys",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("prefix", sa.String(32), nullable=False),
        # Only the hash is stored: a leaked database must not hand over working
        # credentials.
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_rd_api_keys_key_hash", "rd_api_keys", ["key_hash"])

    op.create_table(
        "rd_settings",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("rd_settings")
    op.drop_index("ix_rd_api_keys_key_hash", table_name="rd_api_keys")
    op.drop_table("rd_api_keys")
    for name in ("ix_rd_documents_queue", "ix_rd_documents_doc_type",
                 "ix_rd_documents_status", "ix_rd_documents_created_at"):
        op.drop_index(name, table_name="rd_documents")
    op.drop_table("rd_documents")
