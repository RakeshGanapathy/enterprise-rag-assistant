"""
Per-user rate limiting using a fixed window counter in pgvector.

Window key = floor(unix_ts / window_seconds) * window_seconds
Each (subject, window_key) pair gets an atomic counter.
If the counter exceeds the limit the request is rejected with 429.

Why fixed window over sliding window:
  Sliding window is more accurate but requires storing every request timestamp.
  Fixed window has one edge case: a user can burst 2x the limit across a boundary
  (limit requests at end of window + limit at start of next). For an enterprise
  internal tool this is acceptable — we're preventing runaway usage, not enforcing
  strict per-second guarantees.

Table: rate_limit_counters
  subject       TEXT  — JWT sub claim (user identity)
  window_key    BIGINT — unix timestamp of window start
  request_count INTEGER
  PRIMARY KEY (subject, window_key)

Old windows are cleaned up inline on every 100th request (probabilistic cleanup).
"""
from __future__ import annotations

import math
import time

from fastapi import Depends, HTTPException, status

from app.auth.dependencies import require_auth
from app.auth.jwt import TokenClaims
from app.config import get_settings


def ensure_rate_limit_table(connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS rate_limit_counters (
            subject        TEXT NOT NULL,
            window_key     BIGINT NOT NULL,
            request_count  INTEGER DEFAULT 1,
            PRIMARY KEY (subject, window_key)
        )
    """)


def check_rate_limit(connection, subject: str) -> tuple[int, int]:
    """
    Atomically increment the counter for this subject + window.
    Returns (current_count, limit).
    Raises HTTPException 429 if limit exceeded.
    """
    settings = get_settings()
    limit = settings.rate_limit_requests
    window_seconds = settings.rate_limit_window_seconds

    now = int(time.time())
    window_key = math.floor(now / window_seconds) * window_seconds

    ensure_rate_limit_table(connection)

    row = connection.execute(
        """
        INSERT INTO rate_limit_counters (subject, window_key, request_count)
        VALUES (%s, %s, 1)
        ON CONFLICT (subject, window_key)
        DO UPDATE SET request_count = rate_limit_counters.request_count + 1
        RETURNING request_count
        """,
        (subject, window_key),
    ).fetchone()

    count = row[0]

    # Probabilistic cleanup — delete expired windows 1% of the time
    # Avoids the table growing unbounded without a background job
    if now % 100 == 0:
        connection.execute(
            "DELETE FROM rate_limit_counters WHERE window_key < %s",
            (window_key,),
        )

    if count > limit:
        retry_after = window_seconds - (now - window_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window_seconds}s. "
                   f"Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    return count, limit


def require_rate_limit(
    claims: TokenClaims = Depends(require_auth),
) -> TokenClaims:
    """
    FastAPI dependency — enforces rate limit then passes claims through.
    Use in place of require_auth on endpoints that need both auth + rate limiting.

    Usage:
        @app.post("/ask")
        def ask(request: AskRequest, claims: TokenClaims = Depends(require_rate_limit)):
            ...
    """
    from app.retrieval.vector_store import _connect_pgvector

    with _connect_pgvector() as conn:
        check_rate_limit(conn, claims.subject)

    return claims
