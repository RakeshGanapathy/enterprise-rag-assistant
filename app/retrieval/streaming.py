"""
Streaming RAG response via Server-Sent Events (SSE).

Why two-phase instead of graph.stream()?
  graph.stream() runs every node including generate_answer, which calls llm.invoke()
  (blocking, returns only when the full answer is ready). To stream tokens we need
  llm.astream(), which means we take control of the generation step ourselves.

  Phase 1: call node functions directly in sequence (retrieve -> grade -> rewrite?)
           Each node is sync and fast (<1s total). We yield step events between nodes
           so the client sees progress immediately.

  Phase 2: run the LLM with astream(). Yield a token event per chunk.

  Phase 3: grounding check on the completed answer. Yield done event.

SSE event types:
  {"type": "step",   "text": "retrieved 4 chunks via hybrid + reranked via local"}
  {"type": "token",  "text": "Employees"}
  {"type": "done",   "sources": [...], "grounded": true, "rewritten_question": null}
  {"type": "error",  "text": "error message"}
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.nodes import (
    MIN_RELEVANCE_SCORE,
    _answer_sources,
    _chat_model,
    _format_context,
    check_grounding,
    grade_context,
    retrieve_context,
    rewrite_query,
)
from app.graph.state import RagState


def _sse(event_type: str, **kwargs) -> str:
    return f"data: {json.dumps({'type': event_type, **kwargs})}\n\n"


def _new_steps(before: list[str], after: list[str]) -> list[str]:
    """Return steps added by the most recent node call."""
    return after[len(before):]


async def stream_rag_answer(
    question: str,
    top_k: int = 4,
    search_mode: str = "auto",
    access_filter=None,
) -> AsyncIterator[str]:
    """
    Async generator that yields SSE-formatted strings.
    Plug directly into FastAPI StreamingResponse.
    """
    try:
        yield _sse("step", text="started streaming RAG workflow")

        state: RagState = {
            "question": question,
            "active_question": question,
            "top_k": top_k,
            "search_mode": search_mode,
            "access_filter": access_filter,
            "attempts": 0,
            "results": [],
            "answer": "",
            "sources": [],
            "needs_rewrite": False,
            "grounded": False,
            "workflow_steps": ["started streaming RAG workflow"],
        }

        # ── Phase 1: retrieval + grading + optional rewrite ───────────────────
        prev_steps = list(state["workflow_steps"])
        state = retrieve_context(state)
        for step in _new_steps(prev_steps, state["workflow_steps"]):
            yield _sse("step", text=step)

        prev_steps = list(state["workflow_steps"])
        state = grade_context(state)
        for step in _new_steps(prev_steps, state["workflow_steps"]):
            yield _sse("step", text=step)

        if state["needs_rewrite"]:
            prev_steps = list(state["workflow_steps"])
            state = rewrite_query(state)
            for step in _new_steps(prev_steps, state["workflow_steps"]):
                yield _sse("step", text=step)

            prev_steps = list(state["workflow_steps"])
            state = retrieve_context(state)
            for step in _new_steps(prev_steps, state["workflow_steps"]):
                yield _sse("step", text=step)

        # ── Phase 2: streaming generation ─────────────────────────────────────
        if not state["results"]:
            fallback = "I do not know because no relevant context was found."
            yield _sse("token", text=fallback)
            yield _sse("done", sources=[], grounded=False, rewritten_question=None)
            return

        context = _format_context(state["results"])
        llm = _chat_model()
        messages = [
            SystemMessage(
                content=(
                    "You are an internal company knowledge assistant. Answer only "
                    "from the provided context. If the context does not contain the "
                    "answer, say you do not know. Include concise citations using "
                    "the source names from the context."
                )
            ),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
        ]

        full_answer = ""
        async for chunk in llm.astream(messages):
            token = chunk.content
            if token:
                full_answer += token
                yield _sse("token", text=token)

        yield _sse("step", text="generated answer from retrieved context")

        # ── Phase 3: grounding check ──────────────────────────────────────────
        sources = _answer_sources(full_answer, state["results"])
        state = {
            **state,
            "answer": full_answer,
            "sources": sources,
        }
        state = check_grounding(state)

        rewritten = (
            state["active_question"]
            if state["active_question"] != question
            else None
        )

        yield _sse(
            "done",
            sources=sources,
            grounded=state["grounded"],
            rewritten_question=rewritten,
            workflow_steps=state["workflow_steps"],
        )

    except Exception as exc:
        yield _sse("error", text=str(exc))
