"""
Two-tier semantic answer cache backed by pgvector.

Cache key = SHA-256 of (question + context_hash).
context_hash = SHA-256 of sorted retrieved chunk IDs.

Why context-based key instead of access-filter-based key:
  If HR staff and Admin both ask "What's the PTO policy?" and both retrieve
  the same hr_policy chunks, the answer is identical — one LLM call should
  serve both. Keying on the access filter would create two separate entries
  with duplicate answers.

  Once RBAC has run and chunks are retrieved, the identity of the caller is
  irrelevant — the answer depends only on the question + context, not who asked.

Tier 1 — exact context match (SHA-256 of question + chunk IDs):
  O(1) lookup. Catches the same question retrieving the same chunks.

Tier 2 — semantic match (cosine similarity on question embeddings):
  Catches paraphrases that retrieve the same chunks.
  Threshold: 0.92 — tight enough to avoid wrong answers for different topics.

Cache table: query_cache
  question_hash   TEXT  PRIMARY KEY   — SHA-256(question + context_hash)
  question_text   TEXT                — stored for debugging
  context_hash    TEXT                — SHA-256 of sorted chunk IDs
  question_vec    VECTOR(n)           — for semantic lookup
  answer_json     TEXT                — serialised AskResponse
  search_mode     TEXT
  created_at      TIMESTAMPTZ
  expires_at      TIMESTAMPTZ         — TTL-based invalidation
  hit_count       INTEGER
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from app.config import get_settings

SEMANTIC_THRESHOLD = 0.92       # cosine similarity floor for a cache hit


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def context_hash(chunk_ids: list[str]) -> str:
    """Stable hash of the retrieved chunk set — order-independent."""
    key = "|".join(sorted(chunk_ids))
    return hashlib.sha256(key.encode()).hexdigest()


def ensure_cache_table(connection) -> None:
    settings = get_settings()
    dims = settings.embedding_dimensions
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS query_cache (
            question_hash  TEXT PRIMARY KEY,
            question_text  TEXT NOT NULL,
            context_hash   TEXT,
            question_vec   VECTOR({dims}),
            answer_json    TEXT NOT NULL,
            search_mode    TEXT,
            created_at     TIMESTAMPTZ DEFAULT NOW(),
            expires_at     TIMESTAMPTZ,
            hit_count      INTEGER DEFAULT 0
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS query_cache_vec_idx
        ON query_cache
        USING hnsw (question_vec vector_cosine_ops)
    """)


def make_cache_key(question: str, ctx_hash: str) -> str:
    """Cache key = question + context hash. Same question + same chunks = same key."""
    return _hash(f"{question.strip()}|CTX:{ctx_hash}")


def get_cached_answer(
    connection, question: str, embedding: list[float], ctx_hash: str
) -> dict | None:
    """
    Check both tiers. Returns the cached AskResponse dict or None.
    Also increments hit_count so you can see which questions are most-repeated.
    """
    question_hash = make_cache_key(question, ctx_hash)

    # Tier 1: exact match
    row = connection.execute(
        """
        SELECT answer_json, expires_at FROM query_cache
        WHERE question_hash = %s
        """,
        (question_hash,),
    ).fetchone()

    if row:
        answer_json, expires_at = row
        if expires_at is None or expires_at > _now():
            connection.execute(
                "UPDATE query_cache SET hit_count = hit_count + 1 WHERE question_hash = %s",
                (question_hash,),
            )
            return json.loads(answer_json)

    # Tier 2: semantic match — find questions that retrieved the same context
    # Restrict to entries with the same context_hash so we only match questions
    # whose retrieved chunks were identical (same docs, same content).
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    row = connection.execute(
        """
        SELECT question_hash, answer_json, expires_at,
               1 - (question_vec <=> %s::vector) AS similarity
        FROM query_cache
        WHERE (expires_at IS NULL OR expires_at > NOW())
          AND context_hash = %s
        ORDER BY question_vec <=> %s::vector
        LIMIT 1
        """,
        (vec_str, ctx_hash, vec_str),
    ).fetchone()

    if row:
        cached_hash, answer_json, expires_at, similarity = row
        if similarity >= SEMANTIC_THRESHOLD:
            connection.execute(
                "UPDATE query_cache SET hit_count = hit_count + 1 WHERE question_hash = %s",
                (cached_hash,),
            )
            return json.loads(answer_json)

    return None


def store_cached_answer(
    connection,
    question: str,
    embedding: list[float],
    answer: dict,
    search_mode: str,
    ctx_hash: str,
    ttl_hours: int = 24,
) -> None:
    """Store a new cache entry. Upserts on question_hash so re-indexing is safe."""
    question_hash = make_cache_key(question, ctx_hash)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    expires_at = _now() + timedelta(hours=ttl_hours)

    connection.execute(
        """
        INSERT INTO query_cache
            (question_hash, question_text, context_hash, question_vec, answer_json, search_mode, expires_at)
        VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
        ON CONFLICT (question_hash) DO UPDATE SET
            answer_json  = EXCLUDED.answer_json,
            search_mode  = EXCLUDED.search_mode,
            expires_at   = EXCLUDED.expires_at,
            hit_count    = query_cache.hit_count + 1
        """,
        (question_hash, question.strip(), ctx_hash, vec_str, json.dumps(answer), search_mode, expires_at),
    )


def flush_cache(connection, expired_only: bool = True) -> int:
    """Delete cache entries. Returns count of rows deleted."""
    if expired_only:
        result = connection.execute(
            "DELETE FROM query_cache WHERE expires_at IS NOT NULL AND expires_at <= NOW()"
        )
    else:
        result = connection.execute("DELETE FROM query_cache")
    return result.rowcount


def cache_stats(connection) -> dict:
    row = connection.execute(
        """
        SELECT
            COUNT(*)                                        AS total_entries,
            SUM(hit_count)                                  AS total_hits,
            COUNT(*) FILTER (WHERE expires_at > NOW())      AS live_entries,
            COUNT(*) FILTER (WHERE expires_at <= NOW())     AS expired_entries
        FROM query_cache
        """
    ).fetchone()
    return {
        "total_entries": row[0],
        "total_hits": row[1] or 0,
        "live_entries": row[2],
        "expired_entries": row[3],
    }
