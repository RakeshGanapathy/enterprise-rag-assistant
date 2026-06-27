"""
Unit tests for pure-Python modules — no DB, no OpenAI, no network.

Coverage targets:
  retrieval/hybrid_search.py   tokenize, bm25_search, reciprocal_rank_fusion
  retrieval/query_router.py    classify_query + all classification branches
  retrieval/models.py          Pydantic field validation
  access/rbac.py               get_role_policy, infer_document_metadata
  ingestion/models.py          IngestionResult
  correlation.py               CorrelationIdFilter, get_correlation_id
  retrieval/bm25_cache.py      bump_corpus_version version counter
"""
from __future__ import annotations

import logging

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError


# ── hybrid_search ─────────────────────────────────────────────────────────────

class TestTokenize:
    def test_lowercases(self):
        from app.retrieval.hybrid_search import tokenize
        assert tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        from app.retrieval.hybrid_search import tokenize
        assert tokenize("What's the PTO policy?") == ["what", "s", "the", "pto", "policy"]

    def test_empty_string(self):
        from app.retrieval.hybrid_search import tokenize
        assert tokenize("") == []


class TestBM25Search:
    def test_returns_top_k_results(self):
        from app.retrieval.hybrid_search import bm25_search
        corpus = ["python programming language", "java enterprise software", "python web frameworks"]
        results = bm25_search("python", corpus, top_k=2)
        assert len(results) == 2

    def test_relevant_doc_ranks_higher(self):
        from app.retrieval.hybrid_search import bm25_search
        corpus = ["PTO policy allows 20 days", "security incident response plan", "PTO and vacation benefits"]
        results = bm25_search("PTO policy", corpus, top_k=3)
        top_indices = [idx for idx, _ in results]
        # The two PTO-related docs (0, 2) should rank above the security doc (1)
        assert 1 not in top_indices[:2]

    def test_scores_are_floats(self):
        from app.retrieval.hybrid_search import bm25_search
        results = bm25_search("test", ["test document"], top_k=1)
        assert isinstance(results[0][1], float)


class TestReciprocalRankFusion:
    def test_merges_both_lists(self):
        from app.retrieval.hybrid_search import reciprocal_rank_fusion
        doc_a = Document(page_content="alpha", metadata={})
        doc_b = Document(page_content="beta", metadata={})
        doc_c = Document(page_content="gamma", metadata={})

        semantic = [(doc_a, 0.9), (doc_b, 0.7)]
        keyword = [(2, 0.8), (0, 0.5)]   # corpus indices for doc_c, doc_a
        corpus = [doc_a, doc_b, doc_c]

        results = reciprocal_rank_fusion(semantic, keyword, corpus, top_k=3)
        texts = [doc.page_content for doc, _ in results]
        # doc_a appears in both lists so should rank first
        assert texts[0] == "alpha"

    def test_top_k_respected(self):
        from app.retrieval.hybrid_search import reciprocal_rank_fusion
        docs = [Document(page_content=str(i), metadata={}) for i in range(5)]
        semantic = [(d, 0.9 - i * 0.1) for i, d in enumerate(docs)]
        keyword = [(i, 1.0) for i in range(5)]
        results = reciprocal_rank_fusion(semantic, keyword, docs, top_k=3)
        assert len(results) == 3

    def test_rrf_scores_are_positive(self):
        from app.retrieval.hybrid_search import reciprocal_rank_fusion
        doc = Document(page_content="only doc", metadata={})
        results = reciprocal_rank_fusion([(doc, 0.8)], [(0, 1.0)], [doc], top_k=1)
        assert results[0][1] > 0


# ── query_router ──────────────────────────────────────────────────────────────

class TestClassifyQuery:
    def test_quoted_phrase_is_hybrid(self):
        from app.retrieval.query_router import classify_query
        mode, reason = classify_query('"parental leave policy"')
        assert mode == "hybrid"
        assert "exact_phrase" in reason

    def test_policy_id_is_hybrid(self):
        from app.retrieval.query_router import classify_query
        mode, reason = classify_query("What does HR-042 say about overtime?")
        assert mode == "hybrid"
        assert "policy_id" in reason

    def test_acronym_is_hybrid(self):
        from app.retrieval.query_router import classify_query
        mode, reason = classify_query("SOC2 compliance requirements")
        assert mode == "hybrid"
        assert "acronym_or_code" in reason

    def test_version_tag_is_hybrid(self):
        from app.retrieval.query_router import classify_query
        mode, reason = classify_query("changes in v2.0 release")
        assert mode == "hybrid"
        assert "version_tag" in reason

    def test_short_query_is_hybrid(self):
        from app.retrieval.query_router import classify_query
        mode, reason = classify_query("what is overtime pay")
        assert mode == "hybrid"
        assert "short_query" in reason

    def test_long_conceptual_query_is_semantic(self):
        from app.retrieval.query_router import classify_query
        q = "What are the general guidelines that employees must follow when requesting time off from work?"
        mode, reason = classify_query(q)
        assert mode == "semantic"
        assert "long_conceptual" in reason

    def test_empty_string_returns_hybrid(self):
        from app.retrieval.query_router import classify_query
        mode, reason = classify_query("")
        assert mode == "hybrid"

    def test_garbage_input_returns_hybrid(self):
        from app.retrieval.query_router import classify_query
        mode, reason = classify_query("!!!!!!!!!!!")
        assert mode == "hybrid"

    def test_mid_length_returns_hybrid(self):
        from app.retrieval.query_router import classify_query
        # 7-14 words, no strong signals → default hybrid
        mode, reason = classify_query("what is the parental leave policy for new parents")
        assert mode == "hybrid"
        assert "default" in reason

    def test_never_raises(self):
        from app.retrieval.query_router import classify_query
        # Should not raise for any input type coerced to string
        mode, reason = classify_query("normal question")
        assert mode in {"hybrid", "semantic"}


# ── retrieval/models (Pydantic validation) ────────────────────────────────────

class TestAskRequest:
    def test_valid_question(self):
        from app.retrieval.models import AskRequest
        r = AskRequest(question="What is the PTO policy?")
        assert r.question == "What is the PTO policy?"

    def test_rejects_empty(self):
        from app.retrieval.models import AskRequest
        with pytest.raises(ValidationError):
            AskRequest(question="")

    def test_rejects_over_2000_chars(self):
        from app.retrieval.models import AskRequest
        with pytest.raises(ValidationError):
            AskRequest(question="x" * 2001)

    def test_accepts_exactly_2000_chars(self):
        from app.retrieval.models import AskRequest
        r = AskRequest(question="x" * 2000)
        assert len(r.question) == 2000

    def test_default_search_mode(self):
        from app.retrieval.models import AskRequest
        r = AskRequest(question="hello")
        assert r.search_mode == "auto"

    def test_default_top_k(self):
        from app.retrieval.models import AskRequest
        r = AskRequest(question="hello")
        assert r.top_k == 4

    def test_top_k_bounds(self):
        from app.retrieval.models import AskRequest
        with pytest.raises(ValidationError):
            AskRequest(question="hello", top_k=0)
        with pytest.raises(ValidationError):
            AskRequest(question="hello", top_k=11)


class TestSearchRequest:
    def test_valid(self):
        from app.retrieval.models import SearchRequest
        r = SearchRequest(question="security policy")
        assert r.question == "security policy"

    def test_rejects_empty(self):
        from app.retrieval.models import SearchRequest
        with pytest.raises(ValidationError):
            SearchRequest(question="")


# ── access/rbac ───────────────────────────────────────────────────────────────

class TestGetRolePolicy:
    def test_known_role(self):
        from app.access.rbac import get_role_policy
        policy = get_role_policy("hr_staff")
        assert "hr" in policy.departments
        assert policy.max_access_level == 1  # internal

    def test_admin_has_all_access(self):
        from app.access.rbac import get_role_policy
        policy = get_role_policy("admin")
        assert "all" in policy.departments
        assert policy.max_access_level == 3  # restricted

    def test_unknown_role_falls_back_to_anonymous(self):
        from app.access.rbac import get_role_policy
        policy = get_role_policy("nonexistent_role")
        assert policy.max_access_level == 0  # public only

    def test_employee_has_public_only(self):
        from app.access.rbac import get_role_policy
        policy = get_role_policy("employee")
        assert policy.max_access_level == 0


class TestInferDocumentMetadata:
    def test_hr_prefix(self):
        from app.access.rbac import infer_document_metadata
        meta = infer_document_metadata("hr_policy_v2.pdf")
        assert meta["department"] == "hr"
        assert meta["access_level"] == "confidential"

    def test_security_prefix(self):
        from app.access.rbac import infer_document_metadata
        meta = infer_document_metadata("security_incident_response.md")
        assert meta["department"] == "security"
        assert meta["access_level"] == "restricted"

    def test_product_prefix(self):
        from app.access.rbac import infer_document_metadata
        meta = infer_document_metadata("product_faq.txt")
        assert meta["department"] == "product"
        assert meta["access_level"] == "public"

    def test_unknown_file_falls_back(self):
        from app.access.rbac import infer_document_metadata
        meta = infer_document_metadata("random_document.pdf")
        assert meta["department"] == "general"
        assert meta["access_level"] == "internal"

    def test_case_insensitive(self):
        from app.access.rbac import infer_document_metadata
        meta = infer_document_metadata("HR_POLICY.pdf")
        assert meta["department"] == "hr"


# ── ingestion/models ──────────────────────────────────────────────────────────

class TestIngestionModels:
    def test_ingestion_result(self):
        from app.ingestion.models import IngestionResult
        r = IngestionResult(documents_loaded=2, chunks_created=10, chunks_indexed=10, sources=["a.txt", "b.txt"])
        assert r.documents_loaded == 2
        assert r.chunks_indexed == 10
        assert len(r.sources) == 2

    def test_ingestion_result_default_chunks_indexed(self):
        from app.ingestion.models import IngestionResult
        r = IngestionResult(documents_loaded=1, chunks_created=5, sources=["a.txt"])
        assert r.chunks_indexed == 0

    def test_text_chunk(self):
        from app.ingestion.models import TextChunk
        chunk = TextChunk(text="hello world", metadata={"source": "test.txt"})
        assert chunk.text == "hello world"
        assert chunk.metadata["source"] == "test.txt"

    def test_source_document(self):
        from app.ingestion.models import SourceDocument
        doc = SourceDocument(text="content", metadata={})
        assert doc.text == "content"


# ── correlation ───────────────────────────────────────────────────────────────

class TestCorrelationIdFilter:
    def test_default_correlation_id(self):
        from app.correlation import get_correlation_id
        assert get_correlation_id() == "-"

    def test_filter_injects_correlation_id(self):
        from app.correlation import CorrelationIdFilter
        log_filter = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="test", args=(), exc_info=None,
        )
        log_filter.filter(record)
        assert hasattr(record, "correlation_id")
        assert record.correlation_id == "-"


# ── bm25_cache version counter ────────────────────────────────────────────────

class TestBM25CacheVersionCounter:
    def test_bump_increments_version(self):
        from app.retrieval import bm25_cache
        before = bm25_cache._corpus_version
        bm25_cache.bump_corpus_version()
        assert bm25_cache._corpus_version == before + 1

    def test_multiple_bumps(self):
        from app.retrieval import bm25_cache
        start = bm25_cache._corpus_version
        bm25_cache.bump_corpus_version()
        bm25_cache.bump_corpus_version()
        bm25_cache.bump_corpus_version()
        assert bm25_cache._corpus_version == start + 3


# ── vector_store math ─────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        from app.retrieval.vector_store import _cosine_similarity
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        from app.retrieval.vector_store import _cosine_similarity
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_zero_vector_returns_zero(self):
        from app.retrieval.vector_store import _cosine_similarity
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_opposite_vectors(self):
        from app.retrieval.vector_store import _cosine_similarity
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-6
