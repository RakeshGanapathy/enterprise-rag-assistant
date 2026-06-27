import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, UploadFile
from pydantic import Field
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer

from app.auth.dependencies import require_auth
from app.auth.jwt import TokenClaims, claims_to_access_filter
from app.auth.rate_limit import require_rate_limit
from app.config import get_settings
from app.db import close_pool, init_pool
from app.ingestion.pipeline import ingest_directory, ingest_from_s3, run_ingest_job, sync_directory
from app.middleware import LangfuseTracingMiddleware
from app.retrieval.models import AskRequest, AskResponse, FeedbackRequest, S3IngestRequest, SearchRequest, SearchResponse
from app.retrieval.qa import answer_question, search_knowledge_base
from app.retrieval.streaming import stream_rag_answer
from app.retrieval.vector_store import list_stored_chunks
from app.tracing import is_tracing_enabled, trace_span

from app.correlation import CorrelationIdFilter, CorrelationIdMiddleware

_handler = logging.StreamHandler()
_handler.addFilter(CorrelationIdFilter())
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(correlation_id)s %(name)s %(levelname)s %(message)s"
    )
)
logging.basicConfig(level=logging.INFO, handlers=[_handler])

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ───────────────────────────────────────────────────────────────
    settings.assert_production_ready()

    logger.info("Initialising DB connection pool (min=%d max=%d)...",
                settings.db_pool_min_size, settings.db_pool_max_size)
    init_pool()
    logger.info("DB pool ready.")
    settings.assert_embedding_dimensions()

    # Reap jobs that were stuck in 'processing' from a previous crash
    from app.db import get_conn
    from app.ingestion.document_store import ensure_tables, reap_stuck_jobs
    with get_conn() as conn:
        ensure_tables(conn)
        reaped = reap_stuck_jobs(conn)
        if reaped:
            logger.warning("Reaped %d stuck ingest jobs from previous run", reaped)

    if settings.sync_on_startup and Path(settings.sample_docs_dir).exists():
        logger.info("Startup sync: scanning %s for changes...", settings.sample_docs_dir)
        result = sync_directory(settings.sample_docs_dir)
        logger.info(
            "Startup sync complete: %d indexed, %d skipped",
            len(result["indexed"]),
            len(result["skipped"]),
        )

    async def _scheduled_sync():
        while True:
            await asyncio.sleep(settings.sync_interval_seconds)
            try:
                result = sync_directory(settings.sample_docs_dir)
                if result["indexed"]:
                    logger.info(
                        "Scheduled sync: re-indexed %s",
                        [r["source"] for r in result["indexed"]],
                    )
            except Exception as exc:
                logger.error("Scheduled sync error: %s", exc)

    task = None
    if settings.sync_interval_seconds > 0 and Path(settings.sample_docs_dir).exists():
        task = asyncio.create_task(_scheduled_sync())
        logger.info("Scheduled sync every %ds", settings.sync_interval_seconds)

    yield  # server runs here

    # ── shutdown ──────────────────────────────────────────────────────────────
    if task:
        task.cancel()
    close_pool()
    logger.info("DB pool closed.")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(CorrelationIdMiddleware)
if is_tracing_enabled():
    app.add_middleware(LangfuseTracingMiddleware)


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Deep health check — verifies DB connectivity."""
    from app.db import get_conn
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        db_status = "ok"
    except Exception as exc:
        logger.error("Health check DB failure: %s", exc)
        db_status = f"error: {exc}"

    status = "ok" if db_status == "ok" else "degraded"
    return {
        "status": status,
        "app": settings.app_name,
        "env": settings.app_env,
        "db": db_status,
    }


# ── ingestion ─────────────────────────────────────────────────────────────────

@app.post("/documents/ingest-samples")
def ingest_samples(claims: TokenClaims = Depends(require_auth)) -> dict:
    """Ingest sample docs. Requires auth. Skips files whose content has not changed."""
    with trace_span(name="ingest_samples", metadata={"operation": "ingest_directory"}) as span:
        result = ingest_directory("data/sample_docs")
        span["output"] = {"chunks_indexed": result.chunks_indexed}
        return result.model_dump()


_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_ALLOWED_MIME_PREFIXES = {
    "text/",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml",
}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    claims: TokenClaims = Depends(require_auth),
) -> dict:
    """
    Upload a document for async ingestion. Requires auth.
    Returns a job_id immediately. Poll /documents/status/{job_id} for progress.
    """
    from app.ingestion.document_store import create_job, ensure_tables
    from app.retrieval.vector_store import _connect_pgvector

    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    content = await file.read()

    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    # MIME type check — server-side verification, not just suffix
    content_type = file.content_type or ""
    if not any(content_type.startswith(p) for p in _ALLOWED_MIME_PREFIXES):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(content)

    with _connect_pgvector() as conn:
        ensure_tables(conn)
        job_id = create_job(conn, source=file.filename)

    background_tasks.add_task(
        run_ingest_job,
        file_path=tmp_path,
        source_name=file.filename,
        job_id=job_id,
    )

    return {"job_id": job_id, "source": file.filename, "status": "queued"}


@app.get("/documents/status/{job_id}")
def job_status(job_id: str) -> dict:
    """Poll async ingest job status. status: queued | processing | done | failed."""
    from app.ingestion.document_store import ensure_tables, get_job
    from app.retrieval.vector_store import _connect_pgvector

    with _connect_pgvector() as conn:
        ensure_tables(conn)
        job = get_job(conn, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@app.post("/documents/ingest-s3")
async def ingest_s3(request: S3IngestRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Called by a Lambda function when a file is created, updated, or deleted in S3.
    department and access_level come from S3 object tags — explicit, not inferred.
    Returns a job_id immediately. Processing happens in background.
    """
    from app.ingestion.document_store import create_job, ensure_tables
    from app.retrieval.vector_store import _connect_pgvector

    source_name = Path(request.key).name

    with _connect_pgvector() as conn:
        ensure_tables(conn)
        job_id = create_job(conn, source=source_name)

    background_tasks.add_task(
        ingest_from_s3,
        bucket=request.bucket,
        key=request.key,
        event_type=request.event_type,
        department=request.department,
        access_level=request.access_level,
        presigned_url=request.presigned_url,
        job_id=job_id,
    )

    return {
        "job_id": job_id,
        "source": source_name,
        "event_type": request.event_type,
        "status": "queued",
    }


@app.post("/documents/sync")
def sync_documents(claims: TokenClaims = Depends(require_auth)) -> dict:
    """Scan docs folder for changes and re-index. Requires auth."""
    with trace_span(name="sync_documents", metadata={"operation": "sync"}) as span:
        result = sync_directory(settings.sample_docs_dir)
        span["output"] = {"indexed": len(result["indexed"]), "skipped": len(result["skipped"])}
        return result


@app.get("/documents")
def list_documents() -> list[dict]:
    """List all indexed documents with their content hash, department, access level, and chunk count."""
    from app.ingestion.document_store import ensure_tables, list_documents as _list
    from app.retrieval.vector_store import _connect_pgvector

    with _connect_pgvector() as conn:
        ensure_tables(conn)
        return _list(conn)


# ── retrieval ─────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    claims: TokenClaims = Depends(require_rate_limit),
) -> SearchResponse:
    """
    Search the knowledge base.
    Requires a valid JWT. Department filter and access level are read from the token —
    the user_role field in the request body is ignored.
    """
    access_filter = claims_to_access_filter(claims)
    with trace_span(
        name="search",
        input_data={
            "question": request.question,
            "top_k": request.top_k,
            "search_mode": request.search_mode,
            "subject": claims.subject,
            "domain": claims.domain,
        },
        metadata={"operation": "search"},
    ) as span:
        result = search_knowledge_base(
            request.question, request.top_k, request.search_mode,
            access_filter=access_filter,
        )
        span["output"] = {"results_count": len(result.results)}
        return result


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    claims: TokenClaims = Depends(require_rate_limit),
) -> AskResponse:
    """
    Answer a question using RAG.
    Requires a valid JWT. RBAC filters are derived from token claims, not the request body.
    """
    access_filter = claims_to_access_filter(claims)
    with trace_span(
        name="ask",
        input_data={
            "question": request.question,
            "top_k": request.top_k,
            "search_mode": request.search_mode,
            "subject": claims.subject,
            "domain": claims.domain,
        },
        metadata={"operation": "rag_query"},
    ) as span:
        result = answer_question(
            request.question, request.top_k, request.search_mode,
            access_filter=access_filter,
            conversation_id=request.conversation_id,
        )
        span["output"] = {
            "answer_length": len(result.answer),
            "sources_count": len(result.sources),
            "grounded": result.grounded,
        }
        return result


@app.post("/ask/stream")
async def ask_stream(
    request: AskRequest,
    claims: TokenClaims = Depends(require_rate_limit),
) -> StreamingResponse:
    """
    Streaming version of /ask. Returns Server-Sent Events (text/event-stream).
    Event types: step | token | done | error
    RBAC is enforced from the JWT — token domain and actions determine what chunks are visible.
    """
    access_filter = claims_to_access_filter(claims)
    return StreamingResponse(
        stream_rag_answer(
            question=request.question,
            top_k=request.top_k,
            search_mode=request.search_mode,
            access_filter=access_filter,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── feedback ──────────────────────────────────────────────────────────────────

@app.post("/feedback", status_code=201)
def submit_feedback(
    request: FeedbackRequest,
    claims: TokenClaims = Depends(require_auth),
) -> dict:
    """
    Submit thumbs-up or thumbs-down on an answer.

    Pass the sources array from the AskResponse — used to auto-classify
    whether a negative rating was caused by retrieval failure or generation failure.

    failure_mode is set automatically:
      retrieval  — max source score < 0.25 (wrong chunks retrieved)
      generation — chunks were relevant but LLM still produced a bad answer
    """
    from app.feedback.store import ensure_feedback_table, submit_feedback as _submit
    from app.retrieval.vector_store import _connect_pgvector

    with _connect_pgvector() as conn:
        ensure_feedback_table(conn)
        feedback_id = _submit(
            conn,
            question=request.question,
            rating=request.rating,
            answer=request.answer,
            comment=request.comment,
            conversation_id=request.conversation_id,
            sources=[s.model_dump() for s in request.sources],
        )
    return {"feedback_id": feedback_id, "rating": request.rating}


@app.get("/feedback/summary")
def feedback_summary(claims: TokenClaims = Depends(require_auth)) -> dict:
    """Aggregate feedback stats. Requires auth."""
    from app.feedback.store import ensure_feedback_table, get_summary
    from app.retrieval.vector_store import _connect_pgvector

    with _connect_pgvector() as conn:
        ensure_feedback_table(conn)
        return get_summary(conn)


@app.get("/feedback/triage")
def feedback_triage(
    limit: int = Field(default=50, ge=1, le=500),
    claims: TokenClaims = Depends(require_auth),
) -> list[dict]:
    """List recent negative feedback for engineer review. Requires auth."""
    from app.feedback.store import ensure_feedback_table, list_negative_feedback
    from app.retrieval.vector_store import _connect_pgvector

    with _connect_pgvector() as conn:
        ensure_feedback_table(conn)
        return list_negative_feedback(conn, limit=limit)


# ── conversation history ──────────────────────────────────────────────────────

@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    claims: TokenClaims = Depends(require_auth),
) -> list[dict]:
    """Return full turn history for a conversation. Only the owner can read it."""
    from app.conversation.store import ensure_conversations_table, get_history, get_owner
    from app.retrieval.vector_store import _connect_pgvector

    with _connect_pgvector() as conn:
        ensure_conversations_table(conn)
        owner = get_owner(conn, conversation_id)
        if owner is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if owner != claims.subject:
            raise HTTPException(status_code=403, detail="Access denied")
        return get_history(conn, conversation_id)


# ── cache management ─────────────────────────────────────────────────────────

@app.get("/cache/stats")
def cache_stats_endpoint(claims: TokenClaims = Depends(require_auth)) -> dict:
    """Cache hit counts, live vs expired entries. Requires auth."""
    from app.cache.query_cache import cache_stats, ensure_cache_table
    from app.retrieval.vector_store import _connect_pgvector
    with _connect_pgvector() as conn:
        ensure_cache_table(conn)
        return cache_stats(conn)


@app.delete("/cache")
def flush_cache_endpoint(
    expired_only: bool = True,
    claims: TokenClaims = Depends(require_auth),
) -> dict:
    """Flush cache entries. Requires auth."""
    from app.cache.query_cache import ensure_cache_table, flush_cache
    from app.retrieval.vector_store import _connect_pgvector
    with _connect_pgvector() as conn:
        ensure_cache_table(conn)
        deleted = flush_cache(conn, expired_only=expired_only)
    return {"deleted": deleted, "expired_only": expired_only}


# ── debug ─────────────────────────────────────────────────────────────────────

@app.get("/debug/chunks")
def debug_chunks(
    limit: int = Field(default=20, ge=1, le=100),
    claims: TokenClaims = Depends(require_auth),
) -> list[dict]:
    """Inspect stored chunks. Requires auth. Disabled in production."""
    if settings.app_env not in {"local", "development"}:
        raise HTTPException(status_code=404, detail="Not found")
    with trace_span(name="debug_chunks", input_data={"limit": limit}, metadata={"operation": "debug"}) as span:
        result = list_stored_chunks(limit)
        span["output"] = {"chunks_count": len(result)}
        return result
