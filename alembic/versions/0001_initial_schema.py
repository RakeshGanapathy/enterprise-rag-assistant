"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-27
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id            TEXT PRIMARY KEY,
            text          TEXT NOT NULL,
            metadata_json JSONB NOT NULL,
            embedding     vector(1536) NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            source        TEXT PRIMARY KEY,
            content_hash  TEXT NOT NULL,
            file_mtime    DOUBLE PRECISION,
            department    TEXT,
            access_level  TEXT,
            chunks_count  INTEGER DEFAULT 0,
            indexed_at    TIMESTAMPTZ,
            status        TEXT DEFAULT 'indexed'
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS ingest_jobs (
            job_id         TEXT PRIMARY KEY,
            source         TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'queued',
            chunks_indexed INTEGER DEFAULT 0,
            error          TEXT,
            created_at     TIMESTAMPTZ NOT NULL,
            updated_at     TIMESTAMPTZ NOT NULL
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT NOT NULL,
            turn_index      INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            owner_subject   TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (conversation_id, turn_index)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS conversations_id_idx
        ON conversations (conversation_id, turn_index DESC)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            cache_key     TEXT PRIMARY KEY,
            question      TEXT NOT NULL,
            answer        TEXT NOT NULL,
            sources       JSONB NOT NULL DEFAULT '[]',
            context_hash  TEXT,
            embedding     vector(1536),
            hit_count     INTEGER DEFAULT 0,
            created_at    TIMESTAMPTZ DEFAULT NOW(),
            expires_at    TIMESTAMPTZ NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS query_cache_expires_idx
        ON query_cache (expires_at)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS query_cache_embedding_idx
        ON query_cache USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id     TEXT PRIMARY KEY,
            question        TEXT NOT NULL,
            rating          TEXT NOT NULL,
            answer          TEXT,
            comment         TEXT,
            conversation_id TEXT,
            sources         JSONB NOT NULL DEFAULT '[]',
            failure_mode    TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS rate_limit_counters (
            user_id        TEXT NOT NULL,
            window_key     BIGINT NOT NULL,
            request_count  INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (user_id, window_key)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rate_limit_counters")
    op.execute("DROP TABLE IF EXISTS user_feedback")
    op.execute("DROP TABLE IF EXISTS query_cache")
    op.execute("DROP TABLE IF EXISTS conversations")
    op.execute("DROP TABLE IF EXISTS ingest_jobs")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS chunks")
