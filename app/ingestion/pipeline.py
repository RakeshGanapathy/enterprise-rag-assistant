"""
Ingestion pipeline with document versioning and async job support.

Key behaviour:
  - Each file's text content is hashed (SHA-256).
  - If the hash matches what is already in the documents table, the file is
    SKIPPED — no re-embedding, no API call, no DB write.
  - If the hash is new or changed, ALL old chunks for that source are deleted
    first, then the new version is chunked, embedded, and indexed.
  - This means re-ingesting a folder is always safe and idempotent.

Async flow (for uploads and sync):
  - The caller creates a job record and passes the job_id here.
  - This module updates job status as it progresses.
  - The caller returns the job_id immediately; client polls /documents/status/{id}.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile

from app.access.rbac import infer_document_metadata
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_directory, load_document
from app.ingestion.models import IngestionResult
from app.retrieval.vector_store import index_chunks


# ── public sync API (used by /documents/ingest-samples) ──────────────────────

def ingest_directory(directory: str | Path) -> IngestionResult:
    """Ingest all files in a directory. Skips unchanged files."""
    from app.retrieval.vector_store import _connect_pgvector
    from app.ingestion.document_store import ensure_tables

    root = Path(directory)
    total_loaded = 0
    total_chunks_created = 0
    total_chunks_indexed = 0
    sources: list[str] = []

    with _connect_pgvector() as conn:
        ensure_tables(conn)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue

        result = _ingest_file(path, source_name=path.name)
        if result:
            total_loaded += result.documents_loaded
            total_chunks_created += result.chunks_created
            total_chunks_indexed += result.chunks_indexed
            sources.append(path.name)

    return IngestionResult(
        documents_loaded=total_loaded,
        chunks_created=total_chunks_created,
        chunks_indexed=total_chunks_indexed,
        sources=sorted(sources),
    )


def sync_directory(directory: str | Path) -> dict:
    """
    Scan a directory, detect changed or new files by content hash,
    and re-index only those that changed.
    Returns a summary of what was indexed vs skipped.
    """
    from app.retrieval.vector_store import _connect_pgvector
    from app.ingestion.document_store import ensure_tables

    root = Path(directory)
    indexed = []
    skipped = []

    with _connect_pgvector() as conn:
        ensure_tables(conn)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue

        result = _ingest_file(path, source_name=path.name)
        if result is None:
            skipped.append(path.name)
        else:
            indexed.append({
                "source": path.name,
                "chunks_indexed": result.chunks_indexed,
            })

    return {
        "indexed": indexed,
        "skipped": skipped,
        "total_files_scanned": len(indexed) + len(skipped),
    }


# ── public async API (used by /documents/upload) ─────────────────────────────

async def ingest_upload(file: UploadFile) -> IngestionResult:
    """Ingest an uploaded file synchronously. For async use, call run_ingest_job."""
    suffix = Path(file.filename or "").suffix
    content = await file.read()

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(content)

    try:
        source_name = file.filename or "uploaded_file"
        result = _ingest_file(tmp_path, source_name=source_name, raw_bytes=content)
        return result or IngestionResult(
            documents_loaded=0,
            chunks_created=0,
            chunks_indexed=0,
            sources=[source_name],
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def run_ingest_job(file_path: Path, source_name: str, job_id: str) -> None:
    """
    Background worker function — runs in a thread via FastAPI BackgroundTasks.
    Updates job status in the DB as it progresses.
    """
    from app.retrieval.vector_store import _connect_pgvector
    from app.ingestion.document_store import ensure_tables, update_job

    with _connect_pgvector() as conn:
        ensure_tables(conn)
        update_job(conn, job_id, status="processing")

    try:
        result = _ingest_file(file_path, source_name=source_name)
        chunks_indexed = result.chunks_indexed if result else 0

        with _connect_pgvector() as conn:
            update_job(conn, job_id, status="done", chunks_indexed=chunks_indexed)

    except Exception as exc:
        with _connect_pgvector() as conn:
            update_job(conn, job_id, status="failed", error=str(exc))
    finally:
        file_path.unlink(missing_ok=True)


# ── S3 ingestion ─────────────────────────────────────────────────────────────

def ingest_from_s3(
    bucket: str,
    key: str,
    event_type: str = "created",
    department: str | None = None,
    access_level: str | None = None,
    presigned_url: str | None = None,
    job_id: str | None = None,
) -> IngestionResult | None:
    """
    Download a file from S3 and ingest it with explicit RBAC metadata.

    department and access_level come from S3 object tags (set by the uploader).
    This is more reliable than inferring from filename.

    For deletions (event_type='deleted'), removes all chunks for the source.
    """
    from app.retrieval.vector_store import _connect_pgvector, _delete_chunks_by_source
    from app.ingestion.document_store import (
        delete_document, ensure_tables, update_job,
    )

    source_name = Path(key).name     # "hr/hr_policy_v2.pdf" -> "hr_policy_v2.pdf"

    # Handle deletion events — clean up chunks and document record
    if event_type == "deleted":
        with _connect_pgvector() as conn:
            ensure_tables(conn)
            _delete_chunks_by_source(conn, source_name)
            delete_document(conn, source_name)
            if job_id:
                update_job(conn, job_id, status="done", chunks_indexed=0)
        return IngestionResult(
            documents_loaded=0, chunks_created=0, chunks_indexed=0, sources=[source_name]
        )

    # Download file from S3
    tmp_path = _download_from_s3(bucket, key, presigned_url)
    try:
        # Use explicit tags if provided, else fall back to filename inference
        rbac_override = {}
        if department:
            rbac_override["department"] = department
        if access_level:
            rbac_override["access_level"] = access_level

        result = _ingest_file(
            tmp_path,
            source_name=source_name,
            rbac_override=rbac_override if rbac_override else None,
        )

        if job_id:
            with _connect_pgvector() as conn:
                ensure_tables(conn)
                update_job(
                    conn, job_id,
                    status="done",
                    chunks_indexed=result.chunks_indexed if result else 0,
                )
        return result

    except Exception as exc:
        if job_id:
            with _connect_pgvector() as conn:
                ensure_tables(conn)
                update_job(conn, job_id, status="failed", error=str(exc))
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


def _flush_cache_for_source(connection, source_name: str) -> None:
    """
    Wipe the entire answer cache when a document changes.
    We can't know which cached answers referenced this source,
    so a full flush is the safe choice. Entries will rebuild on next request.
    """
    try:
        from app.cache.query_cache import ensure_cache_table, flush_cache
        ensure_cache_table(connection)
        flush_cache(connection, expired_only=False)
    except Exception:
        pass   # cache flush failure never blocks ingestion


def _download_from_s3(bucket: str, key: str, presigned_url: str | None = None) -> Path:
    """
    Download an S3 object to a temp file and return the path.

    Two modes:
      presigned_url provided → simple HTTP GET (no AWS credentials needed in RAG API)
      no presigned_url       → boto3 direct download (RAG API needs IAM read access)

    Presigned URL is the preferred pattern: Lambda generates the URL using its own
    IAM role, passes it to the RAG API. The RAG API never needs S3 credentials.
    """
    suffix = Path(key).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)

    if presigned_url:
        import httpx
        resp = httpx.get(presigned_url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        tmp_path.write_bytes(resp.content)
        return tmp_path

    # Direct boto3 download
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for direct S3 download. "
            "Run: pip install boto3  OR pass a presigned_url from Lambda."
        ) from exc

    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(tmp_path))
    return tmp_path


# ── core versioned ingest logic ───────────────────────────────────────────────

def _ingest_file(
    path: Path,
    source_name: str,
    raw_bytes: bytes | None = None,
    rbac_override: dict | None = None,
) -> IngestionResult | None:
    """
    Ingest a single file using a two-tier change detection strategy.

    Tier 1 — mtime check (free, no file read):
      If stored mtime matches current mtime → skip entirely.

    Tier 2 — content hash (only when mtime differs):
      Read and hash the extracted text.
      If hash matches stored hash → file was touched but content is same
        (e.g. rsync, copy, git checkout). Update stored mtime only, skip re-index.
      If hash differs → real content change. Delete old chunks, re-index.

    Returns IngestionResult when chunks were indexed, None when skipped.
    """
    from app.retrieval.vector_store import _connect_pgvector, _delete_chunks_by_source
    from app.ingestion.document_store import (
        ensure_tables, get_document, update_mtime, upsert_document,
    )

    current_mtime = path.stat().st_mtime

    with _connect_pgvector() as conn:
        ensure_tables(conn)
        existing = get_document(conn, source_name)

    # Tier 1: mtime unchanged → content definitely unchanged → skip
    if existing and existing["file_mtime"] and existing["file_mtime"] == current_mtime:
        return None

    # Tier 2: mtime changed (or no record yet) → read file and check hash
    documents = load_document(path)
    if not documents:
        return None

    for doc in documents:
        doc.metadata["source"] = source_name

    full_text = "\n".join(doc.text for doc in documents)
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    if existing and existing["content_hash"] == content_hash:
        # mtime bumped but content is identical — update mtime only, no re-index
        with _connect_pgvector() as conn:
            update_mtime(conn, source_name, current_mtime)
        return None

    # Real content change — delete stale chunks then index new version.
    # Also flush the answer cache: answers that referenced this source are now stale.
    with _connect_pgvector() as conn:
        _delete_chunks_by_source(conn, source_name)
        _flush_cache_for_source(conn, source_name)

    chunks = chunk_documents(documents)
    chunks_indexed = index_chunks(chunks)

    # S3 tags take precedence over filename inference
    rbac_meta = infer_document_metadata(source_name)
    if rbac_override:
        rbac_meta.update(rbac_override)

    with _connect_pgvector() as conn:
        upsert_document(
            conn,
            source=source_name,
            content_hash=content_hash,
            chunks_count=chunks_indexed,
            department=rbac_meta["department"],
            access_level=rbac_meta["access_level"],
            file_mtime=current_mtime,
        )

    return IngestionResult(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_indexed=chunks_indexed,
        sources=[source_name],
    )
