"""
Reranker: second-stage precision layer after BM25 + semantic recall.

Why a two-stage approach?
  Stage 1 (BM25 + semantic): bi-encoders — fast, run independently, optimise RECALL.
    They embed query and document separately, so they miss fine-grained relevance.
  Stage 2 (cross-encoder reranker): sees query + document TOGETHER in one forward pass.
    Much more accurate at judging relevance, but too slow to run on full corpus.

Rule: always retrieve more than you need (reranker_top_n), then rerank down to top_k.
  e.g. retrieve 20 candidates → rerank → return top 4.

Backends:
  local  — sentence-transformers cross-encoder, runs on CPU, free, good for dev/prod
  cohere — Cohere Rerank API, managed, production-grade, costs per call
  none   — skip reranking (pass through as-is, useful for A/B comparison)
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.config import get_settings


def rerank(
    question: str,
    candidates: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Rerank a candidate list and return the top_k most relevant results.
    Input scores are RRF scores; output scores are reranker scores.
    Falls back to input order if the backend is unavailable.
    """
    if not candidates:
        return candidates

    settings = get_settings()
    backend = settings.reranker_backend

    if backend == "cohere":
        return _rerank_cohere(question, candidates, top_k)
    if backend == "local":
        return _rerank_local(question, candidates, top_k)

    # backend == "none" or unknown — return as-is, already sorted by RRF
    return candidates[:top_k]


# ── local cross-encoder ───────────────────────────────────────────────────────

_local_model_cache: dict[str, object] = {}


def _rerank_local(
    question: str,
    candidates: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for local reranking. "
            "Run: pip install sentence-transformers"
        ) from exc

    settings = get_settings()
    model_name = settings.reranker_model

    if model_name not in _local_model_cache:
        _local_model_cache[model_name] = CrossEncoder(model_name)
    model = _local_model_cache[model_name]

    texts = [doc.page_content for doc, _ in candidates]
    pairs = [[question, text] for text in texts]
    scores: list[float] = model.predict(pairs).tolist()

    scored = sorted(
        zip([doc for doc, _ in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return scored[:top_k]


# ── cohere rerank API ─────────────────────────────────────────────────────────

def _rerank_cohere(
    question: str,
    candidates: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    try:
        import cohere
    except ImportError as exc:
        raise ImportError(
            "cohere is required for Cohere reranking. "
            "Run: pip install cohere"
        ) from exc

    settings = get_settings()
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY is missing. Add it to .env.")

    client = cohere.Client(settings.cohere_api_key)
    texts = [doc.page_content for doc, _ in candidates]

    response = client.rerank(
        model="rerank-english-v3.0",
        query=question,
        documents=texts,
        top_n=top_k,
    )

    doc_list = [doc for doc, _ in candidates]
    return [
        (doc_list[result.index], result.relevance_score)
        for result in response.results
    ]
