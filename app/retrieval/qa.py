from app.retrieval.models import AccessFilter, AskResponse, SearchMode, SearchResponse, SearchResult, Source
from app.retrieval.query_router import classify_query
from app.retrieval.vector_store import access_filter_for_role, hybrid_search_chunks, search_chunks
from app.tracing import trace_span


def resolve_search_mode(question: str, search_mode: SearchMode) -> tuple[str, str]:
    if search_mode == "auto":
        return classify_query(question)
    return search_mode, f"caller_specified:{search_mode}"


def _retrieve(
    question: str,
    top_k: int,
    resolved_mode: str,
    access_filter: AccessFilter | None,
):
    if resolved_mode == "hybrid":
        return hybrid_search_chunks(question, top_k=top_k, access_filter=access_filter)
    return search_chunks(question, top_k=top_k, access_filter=access_filter)


def search_knowledge_base(
    question: str,
    top_k: int = 4,
    search_mode: SearchMode = "auto",
    user_role: str | None = None,
    access_filter: AccessFilter | None = None,
    preloaded_matches=None,
) -> SearchResponse:
    resolved_mode, router_reason = resolve_search_mode(question, search_mode)
    if access_filter is None:
        access_filter = access_filter_for_role(user_role or "employee")

    with trace_span(
        name="search_knowledge_base",
        input_data={"question": question, "top_k": top_k, "search_mode": resolved_mode},
        metadata={"operation": resolved_mode, "router_reason": router_reason},
    ) as span:
        matches = preloaded_matches if preloaded_matches is not None else _retrieve(question, top_k, resolved_mode, access_filter)
        results = [
            SearchResult(
                text=document.page_content,
                source=_source_from_metadata(document.metadata, score),
            )
            for document, score in matches
        ]
        response = SearchResponse(question=question, results=results, search_mode=resolved_mode)
        span["output"] = {"results_count": len(results), "router_reason": router_reason}
        return response


def answer_question(
    question: str,
    top_k: int = 4,
    search_mode: SearchMode = "auto",
    user_role: str | None = None,
    access_filter: AccessFilter | None = None,
    conversation_id: str | None = None,
) -> AskResponse:
    """
    Answer a question using the RAG pipeline.

    Cache behaviour:
      1. Embed the question.
      2. Check exact hash → then semantic similarity in query_cache.
      3. Cache hit → return immediately (no retrieval, no LLM call).
      4. Cache miss → run full pipeline → store result in cache.

    access_filter is NOT part of the cache key deliberately: the cache stores
    the best possible answer for the question content. RBAC controls which
    chunks are retrieved — if two users with different roles ask the same
    question, they may get different answers. Cache is therefore scoped per
    access_filter by including its fingerprint in the hash.
    """
    from app.graph.workflow import run_rag_workflow

    if access_filter is None:
        access_filter = access_filter_for_role(user_role or "employee")

    resolved_mode, _ = resolve_search_mode(question, search_mode)

    with trace_span(
        name="answer_question",
        input_data={"question": question, "top_k": top_k, "search_mode": search_mode},
        metadata={"operation": "rag_answer"},
    ) as span:

        # ── conversation history ──────────────────────────────────────────────
        conv_id, history = _load_history(conversation_id)

        # ── retrieve first — cache key depends on what was retrieved ──────────
        matches = _retrieve(question, top_k, resolved_mode, access_filter)
        chunk_ids = [
            f"{doc.metadata.get('source','?')}:{doc.metadata.get('chunk_index','?')}"
            for doc, _ in matches
        ]

        from app.cache.query_cache import context_hash as _ctx_hash
        ctx_hash = _ctx_hash(chunk_ids)

        # ── cache lookup (skip for follow-up turns) ───────────────────────────
        cached = None
        embedding = _embed_question(question)
        if not history:
            cached = _cache_get(question, embedding, ctx_hash)

        if cached is not None:
            cached.conversation_id = conv_id
            span["output"] = {"cache_hit": True, "answer_length": len(cached.answer)}
            return cached

        # ── LLM generation (retrieval already done above) ─────────────────────
        result = run_rag_workflow(
            question,
            top_k=top_k,
            search_mode=search_mode,
            access_filter=access_filter,
            conversation_history=history,
            preloaded_matches=matches,
        )

        # ── persist conversation + cache ──────────────────────────────────────
        _save_turn(conv_id, question, result.answer)
        result.conversation_id = conv_id

        if not history:
            _cache_put(question, embedding, result, resolved_mode, ctx_hash)

        span["output"] = {
            "cache_hit": False,
            "conversation_id": conv_id,
            "answer_length": len(result.answer),
            "sources_count": len(result.sources),
            "grounded": result.grounded,
        }
        return result


# ── conversation helpers ──────────────────────────────────────────────────────

def _load_history(conversation_id: str | None) -> tuple[str, list[dict]]:
    """
    Returns (conversation_id, history_window).
    Creates a new conversation_id if none provided (first turn).
    """
    from app.conversation.store import (
        ensure_conversations_table, get_recent_window, new_conversation_id,
    )
    from app.retrieval.vector_store import _connect_pgvector

    conv_id = conversation_id or new_conversation_id()
    try:
        with _connect_pgvector() as conn:
            ensure_conversations_table(conn)
            history = get_recent_window(conn, conv_id) if conversation_id else []
        return conv_id, history
    except Exception:
        return conv_id, []


def _save_turn(conversation_id: str, question: str, answer: str) -> None:
    from app.conversation.store import append_turn, ensure_conversations_table
    from app.retrieval.vector_store import _connect_pgvector

    try:
        with _connect_pgvector() as conn:
            ensure_conversations_table(conn)
            append_turn(conn, conversation_id, "user", question)
            append_turn(conn, conversation_id, "assistant", answer)
    except Exception:
        pass   # conversation persistence failure never breaks the response


# ── cache helpers ─────────────────────────────────────────────────────────────

def _embed_question(question: str) -> list[float]:
    from langchain_openai import OpenAIEmbeddings
    from app.config import get_settings
    settings = get_settings()
    embedder = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
    return embedder.embed_query(question)


def _cache_get(question: str, embedding: list[float], ctx_hash: str) -> AskResponse | None:
    try:
        from app.cache.query_cache import ensure_cache_table, get_cached_answer
        from app.retrieval.vector_store import _connect_pgvector

        with _connect_pgvector() as conn:
            ensure_cache_table(conn)
            hit = get_cached_answer(conn, question, embedding, ctx_hash)

        if hit is None:
            return None

        sources = [Source(**s) for s in hit.get("sources", [])]
        return AskResponse(
            answer=hit["answer"],
            sources=sources,
            rewritten_question=hit.get("rewritten_question"),
            grounded=hit.get("grounded", False),
            workflow_steps=hit.get("workflow_steps", ["served from cache"]),
        )
    except Exception:
        return None


def _cache_put(
    question: str,
    embedding: list[float],
    result: AskResponse,
    search_mode: str,
    ctx_hash: str,
) -> None:
    try:
        from app.cache.query_cache import ensure_cache_table, store_cached_answer
        from app.retrieval.vector_store import _connect_pgvector
        from app.config import get_settings

        settings = get_settings()
        with _connect_pgvector() as conn:
            ensure_cache_table(conn)
            store_cached_answer(
                conn,
                question=question,
                embedding=embedding,
                answer=result.model_dump(),
                search_mode=search_mode,
                ctx_hash=ctx_hash,
                ttl_hours=int(settings.cache_ttl_hours),
            )
    except Exception:
        pass


def _source_from_metadata(metadata: dict, score: float | None = None) -> Source:
    return Source(
        source=str(metadata.get("source", "unknown")),
        page=metadata.get("page"),
        chunk_index=metadata.get("chunk_index"),
        score=score,
    )
