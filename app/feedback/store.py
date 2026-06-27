"""
User feedback store — thumbs up / thumbs down on RAG answers.

Schema:
  feedback_id     UUID PK
  conversation_id TEXT (optional — links to conversations table)
  question        TEXT
  answer          TEXT
  rating          TEXT  — "positive" | "negative"
  comment         TEXT  — optional free text from the user
  sources_json    TEXT  — JSON of sources returned with the answer
  failure_mode    TEXT  — auto-classified on negative ratings:
                            "retrieval"  — wrong or missing chunks
                            "generation" — chunks were fine, LLM failed
                            null         — positive rating or unclassifiable
  created_at      TIMESTAMPTZ

Failure mode triage:
  We look at the max source score in the answer.
  Low score  → retrieval pulled irrelevant chunks → retrieval failure
  High score → relevant chunks retrieved but answer was still wrong → generation failure

  Threshold is the same MIN_RELEVANCE_SCORE used in the grading node (0.25).
  Below threshold: retrieval failure.
  At or above threshold: generation failure (hallucination / off-topic).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

RETRIEVAL_SCORE_THRESHOLD = 0.25   # matches MIN_RELEVANCE_SCORE in nodes.py


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_feedback_table(connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id      TEXT PRIMARY KEY,
            conversation_id  TEXT,
            question         TEXT NOT NULL,
            answer           TEXT,
            rating           TEXT NOT NULL,
            comment          TEXT,
            sources_json     TEXT,
            failure_mode     TEXT,
            created_at       TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS feedback_rating_idx ON feedback (rating)
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS feedback_conv_idx ON feedback (conversation_id)
        WHERE conversation_id IS NOT NULL
    """)


def submit_feedback(
    connection,
    question: str,
    rating: str,                        # "positive" | "negative"
    answer: str | None = None,
    comment: str | None = None,
    conversation_id: str | None = None,
    sources: list[dict] | None = None,  # Source dicts from AskResponse
) -> str:
    feedback_id = str(uuid.uuid4())
    failure_mode = _classify_failure(rating, sources)

    connection.execute(
        """
        INSERT INTO feedback
            (feedback_id, conversation_id, question, answer, rating, comment,
             sources_json, failure_mode)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            feedback_id,
            conversation_id,
            question,
            answer,
            rating,
            comment,
            json.dumps(sources or []),
            failure_mode,
        ),
    )
    return feedback_id


def get_summary(connection) -> dict:
    """Aggregate stats — useful for a monitoring dashboard or weekly review."""
    total = connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    pos   = connection.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'positive'").fetchone()[0]
    neg   = connection.execute("SELECT COUNT(*) FROM feedback WHERE rating = 'negative'").fetchone()[0]

    # Negative breakdown by failure mode
    modes = connection.execute(
        """
        SELECT failure_mode, COUNT(*) FROM feedback
        WHERE rating = 'negative'
        GROUP BY failure_mode
        """
    ).fetchall()
    failure_breakdown = {row[0] or "unclassified": row[1] for row in modes}

    # Top 5 most-flagged questions
    flagged = connection.execute(
        """
        SELECT question, COUNT(*) AS cnt FROM feedback
        WHERE rating = 'negative'
        GROUP BY question
        ORDER BY cnt DESC
        LIMIT 5
        """
    ).fetchall()

    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "positive_rate": round(pos / total, 3) if total else None,
        "failure_breakdown": failure_breakdown,
        "top_flagged_questions": [
            {"question": row[0], "negative_count": row[1]} for row in flagged
        ],
    }


def list_negative_feedback(connection, limit: int = 50) -> list[dict]:
    """Return negative feedback entries for engineer triage."""
    rows = connection.execute(
        """
        SELECT feedback_id, conversation_id, question, answer,
               comment, sources_json, failure_mode, created_at
        FROM feedback
        WHERE rating = 'negative'
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "feedback_id": r[0],
            "conversation_id": r[1],
            "question": r[2],
            "answer": r[3],
            "comment": r[4],
            "sources": json.loads(r[5] or "[]"),
            "failure_mode": r[6],
            "created_at": r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]


def _classify_failure(rating: str, sources: list[dict] | None) -> str | None:
    """
    Auto-classify why a negative answer failed.

    retrieval  — max source score below threshold → wrong chunks were retrieved
    generation — chunks were relevant but the LLM still produced a bad answer
    """
    if rating != "negative":
        return None

    if not sources:
        return "retrieval"   # no chunks at all → retrieval failure

    scores = [s.get("score") for s in sources if s.get("score") is not None]
    if not scores:
        return None   # no score metadata — can't classify

    max_score = max(scores)
    return "retrieval" if max_score < RETRIEVAL_SCORE_THRESHOLD else "generation"
