"""
Hybrid search: combines BM25 keyword search with dense vector search
using Reciprocal Rank Fusion (RRF) to merge ranked result lists.

Why RRF?
- It doesn't require score normalization across two different scoring systems.
- It's robust: a document that ranks well in both lists gets a strong combined score.
- Formula: RRF(d) = sum(1 / (k + rank_i(d))) for each ranked list i, k=60 by default.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    pass

RRF_K = 60  # constant that dampens the impact of very high ranks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_search(
    query: str,
    corpus_texts: list[str],
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Run BM25 over a plain list of strings.
    Returns (corpus_index, bm25_score) sorted descending, limited to top_k.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 is required for hybrid search. "
            "Run: pip install rank-bm25"
        ) from exc

    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    scores: list[float] = bm25.get_scores(tokenize(query))

    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return indexed[:top_k]


def reciprocal_rank_fusion(
    semantic_ranking: list[tuple[Document, float]],
    keyword_ranking: list[tuple[int, float]],
    corpus_docs: list[Document],
    top_k: int,
    k: int = RRF_K,
) -> list[tuple[Document, float]]:
    """
    Merge two ranked lists into one using RRF.

    semantic_ranking: ordered list of (Document, cosine_score) from vector search
    keyword_ranking:  ordered list of (corpus_index, bm25_score) from BM25
    corpus_docs:      full list of Documents in corpus order (for keyword indexing)
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Score from semantic ranking
    for rank, (doc, _score) in enumerate(semantic_ranking, start=1):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        doc_map[key] = doc

    # Score from BM25 ranking
    for rank, (corpus_idx, _bm25_score) in enumerate(keyword_ranking, start=1):
        doc = corpus_docs[corpus_idx]
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        doc_map[key] = doc

    sorted_keys = sorted(rrf_scores, key=lambda k_: rrf_scores[k_], reverse=True)
    return [(doc_map[key], rrf_scores[key]) for key in sorted_keys[:top_k]]
