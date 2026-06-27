"""
BM25 corpus cache — avoids a full-table scan on every hybrid search.

The cache holds the complete unfiltered chunk corpus. Access-filter restriction
happens in-process via _passes_access_filter, so one invalidation serves all users.

Invalidation is version-based: any document upsert or delete bumps _corpus_version.
The cache is rebuilt lazily on the next hybrid search after invalidation.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from app.retrieval.models import AccessFilter

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cached_docs: list[Document] | None = None
_cached_version: int = -1
_corpus_version: int = 0


def bump_corpus_version() -> None:
    """Call after any document upsert or delete to mark the cache stale."""
    global _corpus_version
    with _lock:
        _corpus_version += 1


def get_cached_corpus(access_filter: "AccessFilter | None" = None) -> list[Document]:
    """
    Return the BM25 corpus, using the in-memory cache when valid.

    If the cache is stale (version mismatch) or empty, reloads from the DB.
    Access filtering is applied in-process after loading.
    """
    global _cached_docs, _cached_version

    with _lock:
        current_version = _corpus_version
        if _cached_docs is not None and _cached_version == current_version:
            docs = _cached_docs
        else:
            docs = None

    if docs is None:
        docs = _reload_corpus()
        with _lock:
            _cached_docs = docs
            _cached_version = current_version
        logger.info("BM25 corpus cache refreshed: %d chunks", len(docs))

    if access_filter is None:
        return docs

    from app.retrieval.vector_store import _passes_access_filter
    return [d for d in docs if _passes_access_filter(d.metadata, access_filter)]


def _reload_corpus() -> list[Document]:
    """Load all chunks from the DB (no access filter — cached unfiltered)."""
    from app.config import get_settings
    settings = get_settings()

    if settings.vector_store_backend == "pgvector":
        from app.retrieval.vector_store import _connect_pgvector
        with _connect_pgvector() as conn:
            rows = conn.execute("SELECT text, metadata_json FROM chunks").fetchall()
        return [Document(page_content=text, metadata=meta) for text, meta in rows]

    import json
    from app.retrieval.vector_store import _connect_sqlite, _create_table_sqlite
    with _connect_sqlite() as conn:
        _create_table_sqlite(conn)
        rows = conn.execute("SELECT text, metadata_json FROM chunks").fetchall()
    return [Document(page_content=text, metadata=json.loads(meta)) for text, meta in rows]
