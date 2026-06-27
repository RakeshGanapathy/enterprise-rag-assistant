from typing import TypedDict

from app.retrieval.models import AccessFilter, SearchMode, SearchResult


class RagState(TypedDict):
    question: str
    active_question: str        # may be rewritten; used for retrieval
    top_k: int
    search_mode: SearchMode
    access_filter: AccessFilter | None
    conversation_history: list[dict]    # [{"role": "user"|"assistant", "content": "..."}]
    preloaded_matches: list | None      # (Document, score) pairs already retrieved
    attempts: int
    results: list[SearchResult]
    answer: str
    sources: list[dict]
    needs_rewrite: bool
    grounded: bool
    workflow_steps: list[str]

