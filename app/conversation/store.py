"""
Conversation history store backed by pgvector.

Each conversation is a list of turns:
  { "role": "user" | "assistant", "content": "..." }

Stored as JSONB in a conversations table. The client receives a
conversation_id on the first turn and sends it back on subsequent turns.

Max window: last N turns injected into the LLM context.
Old turns are kept in the DB for audit but not sent to the LLM.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone


MAX_HISTORY_TURNS = 6   # last 6 turns (3 user + 3 assistant) injected into LLM


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_conversations_table(connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id  TEXT NOT NULL,
            turn_index       INTEGER NOT NULL,
            role             TEXT NOT NULL,        -- 'user' | 'assistant'
            content          TEXT NOT NULL,
            created_at       TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (conversation_id, turn_index)
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS conversations_id_idx
        ON conversations (conversation_id, turn_index DESC)
    """)


def new_conversation_id() -> str:
    return str(uuid.uuid4())


def get_history(connection, conversation_id: str) -> list[dict]:
    """Return all turns for a conversation, oldest first."""
    rows = connection.execute(
        """
        SELECT role, content FROM conversations
        WHERE conversation_id = %s
        ORDER BY turn_index ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def append_turn(connection, conversation_id: str, role: str, content: str) -> None:
    """Append one turn. turn_index auto-increments per conversation."""
    connection.execute(
        """
        INSERT INTO conversations (conversation_id, turn_index, role, content)
        VALUES (
            %s,
            COALESCE(
                (SELECT MAX(turn_index) + 1 FROM conversations WHERE conversation_id = %s),
                0
            ),
            %s, %s
        )
        """,
        (conversation_id, conversation_id, role, content),
    )


def get_recent_window(connection, conversation_id: str) -> list[dict]:
    """Return the last MAX_HISTORY_TURNS turns — what actually gets sent to the LLM."""
    rows = connection.execute(
        """
        SELECT role, content FROM conversations
        WHERE conversation_id = %s
        ORDER BY turn_index DESC
        LIMIT %s
        """,
        (conversation_id, MAX_HISTORY_TURNS),
    ).fetchall()
    # Reverse so oldest is first (chronological order for the LLM)
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]
