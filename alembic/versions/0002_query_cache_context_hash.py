"""add context_hash column to query_cache

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-27
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE query_cache ADD COLUMN IF NOT EXISTS context_hash TEXT"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE query_cache DROP COLUMN IF EXISTS context_hash"
    )
