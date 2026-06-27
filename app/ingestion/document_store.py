"""
Persistent tracking of indexed documents and async ingest jobs.

Two tables live alongside the `chunks` table in the same PostgreSQL DB:

  documents    one row per source file
               tracks content_hash so re-ingest skips unchanged files
               and deletes stale chunks before indexing a new version

  ingest_jobs  one row per async ingest request
               lets clients poll /documents/status/{job_id}

Why content_hash instead of file mtime?
  mtime changes when a file is copied, rsync'd, or touched without content change.
  Hash only changes when the actual text content changes — no false re-indexing.
  We hash the extracted text (not raw bytes) so PDF metadata changes are ignored.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


# ── table management ──────────────────────────────────────────────────────────

def ensure_tables(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            source          TEXT PRIMARY KEY,
            content_hash    TEXT NOT NULL,
            file_mtime      DOUBLE PRECISION,   -- os mtime, fast pre-filter
            department      TEXT,
            access_level    TEXT,
            chunks_count    INTEGER DEFAULT 0,
            indexed_at      TIMESTAMPTZ,
            status          TEXT DEFAULT 'indexed'
        )
        """
    )
    # Add file_mtime to existing tables that were created before this column existed
    connection.execute(
        """
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_mtime DOUBLE PRECISION
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_jobs (
            job_id          TEXT PRIMARY KEY,
            source          TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'queued',
            chunks_indexed  INTEGER DEFAULT 0,
            error           TEXT,
            created_at      TIMESTAMPTZ NOT NULL,
            updated_at      TIMESTAMPTZ NOT NULL
        )
        """
    )


# ── document registry ─────────────────────────────────────────────────────────

def get_document(connection, source: str) -> dict | None:
    row = connection.execute(
        "SELECT source, content_hash, file_mtime, department, access_level, chunks_count, indexed_at, status "
        "FROM documents WHERE source = %s",
        (source,),
    ).fetchone()
    if not row:
        return None
    return {
        "source": row[0],
        "content_hash": row[1],
        "file_mtime": row[2],
        "department": row[3],
        "access_level": row[4],
        "chunks_count": row[5],
        "indexed_at": row[6].isoformat() if row[6] else None,
        "status": row[7],
    }


def upsert_document(
    connection,
    source: str,
    content_hash: str,
    chunks_count: int,
    department: str = "general",
    access_level: str = "internal",
    file_mtime: float | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO documents (source, content_hash, file_mtime, department, access_level, chunks_count, indexed_at, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'indexed')
        ON CONFLICT (source) DO UPDATE SET
            content_hash  = EXCLUDED.content_hash,
            file_mtime    = EXCLUDED.file_mtime,
            department    = EXCLUDED.department,
            access_level  = EXCLUDED.access_level,
            chunks_count  = EXCLUDED.chunks_count,
            indexed_at    = EXCLUDED.indexed_at,
            status        = 'indexed'
        """,
        (source, content_hash, file_mtime, department, access_level, chunks_count, _now()),
    )


def update_mtime(connection, source: str, file_mtime: float) -> None:
    """Update stored mtime without touching chunks — used when mtime changed but content is same."""
    connection.execute(
        "UPDATE documents SET file_mtime = %s WHERE source = %s",
        (file_mtime, source),
    )


def list_documents(connection) -> list[dict]:
    rows = connection.execute(
        "SELECT source, content_hash, department, access_level, chunks_count, indexed_at, status "
        "FROM documents ORDER BY indexed_at DESC"
    ).fetchall()
    return [
        {
            "source": r[0],
            "content_hash": r[1][:12] + "...",   # abbreviated for display
            "department": r[2],
            "access_level": r[3],
            "chunks_count": r[4],
            "indexed_at": r[5].isoformat() if r[5] else None,
            "status": r[6],
        }
        for r in rows
    ]


def delete_document(connection, source: str) -> None:
    connection.execute("DELETE FROM documents WHERE source = %s", (source,))


# ── job registry ──────────────────────────────────────────────────────────────

def create_job(connection, source: str) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    connection.execute(
        "INSERT INTO ingest_jobs (job_id, source, status, created_at, updated_at) "
        "VALUES (%s, %s, 'queued', %s, %s)",
        (job_id, source, now, now),
    )
    return job_id


def update_job(
    connection,
    job_id: str,
    status: str,
    chunks_indexed: int | None = None,
    error: str | None = None,
) -> None:
    connection.execute(
        """
        UPDATE ingest_jobs SET
            status         = %s,
            chunks_indexed = COALESCE(%s, chunks_indexed),
            error          = %s,
            updated_at     = %s
        WHERE job_id = %s
        """,
        (status, chunks_indexed, error, _now(), job_id),
    )


def get_job(connection, job_id: str) -> dict | None:
    row = connection.execute(
        "SELECT job_id, source, status, chunks_indexed, error, created_at, updated_at "
        "FROM ingest_jobs WHERE job_id = %s",
        (job_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "job_id": row[0],
        "source": row[1],
        "status": row[2],
        "chunks_indexed": row[3],
        "error": row[4],
        "created_at": row[5].isoformat() if row[5] else None,
        "updated_at": row[6].isoformat() if row[6] else None,
    }


# ── stuck job reaper ─────────────────────────────────────────────────────────

def reap_stuck_jobs(connection, stuck_after_minutes: int = 10) -> int:
    """
    Mark jobs stuck in 'processing' for longer than stuck_after_minutes as 'failed'.
    Called at startup to recover from worker crashes mid-ingest.
    """
    result = connection.execute(
        """
        UPDATE ingest_jobs
        SET status     = 'failed',
            error      = 'Job timed out — worker may have restarted mid-ingest',
            updated_at = %s
        WHERE status = 'processing'
          AND updated_at < NOW() - INTERVAL '%s minutes'
        """,
        (_now(), stuck_after_minutes),
    )
    return result.rowcount


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)
