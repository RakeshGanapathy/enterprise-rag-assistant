from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.graph.state import RagState
from app.retrieval.models import SearchResult, Source
from app.retrieval.qa import resolve_search_mode, search_knowledge_base


MIN_RELEVANCE_SCORE = 0.25


def retrieve_context(state: RagState) -> RagState:
    search_mode = state.get("search_mode", "auto")
    resolved_mode, router_reason = resolve_search_mode(state["active_question"], search_mode)

    # Use pre-loaded matches from qa.py on the first pass (avoids double retrieval).
    # On rewrite attempts (attempts > 0) the question changed, so re-retrieve.
    preloaded = state.get("preloaded_matches") if state.get("attempts", 0) == 0 else None

    search_response = search_knowledge_base(
        state["active_question"],
        top_k=state["top_k"],
        search_mode=resolved_mode,
        access_filter=state.get("access_filter"),
        preloaded_matches=preloaded,
    )

    from app.config import get_settings
    settings = get_settings()
    reranker_note = (
        f"reranked via {settings.reranker_backend}"
        if resolved_mode == "hybrid" and settings.reranker_backend != "none"
        else "no reranking"
    )

    return {
        **state,
        "results": search_response.results,
        "workflow_steps": [
            *state["workflow_steps"],
            f"router: {router_reason}",
            f"retrieved {len(search_response.results)} chunks via {resolved_mode} + {reranker_note}",
        ],
    }


def grade_context(state: RagState) -> RagState:
    best_score = _best_score(state["results"])
    needs_rewrite = best_score < MIN_RELEVANCE_SCORE and state["attempts"] == 0

    return {
        **state,
        "needs_rewrite": needs_rewrite,
        "workflow_steps": [
            *state["workflow_steps"],
            f"graded retrieval best_score={best_score:.3f}",
        ],
    }


def rewrite_query(state: RagState) -> RagState:
    settings = get_settings()
    llm = _chat_model()

    history = state.get("conversation_history", [])
    history_block = _format_history(history)

    system_prompt = (
        "You are a query rewriter for an enterprise document search system.\n"
        "Rewrite the user's question into a self-contained search query that can be "
        "understood without any conversation context.\n"
        "Resolve pronouns and references like 'it', 'that', 'what about X' using the "
        "conversation history below.\n"
        "Return only the rewritten query, nothing else."
    )
    if history_block:
        system_prompt += f"\n\nConversation so far:\n{history_block}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ])

    rewritten = str(response.content).strip() or state["question"]
    return {
        **state,
        "active_question": rewritten,
        "attempts": state["attempts"] + 1,
        "workflow_steps": [
            *state["workflow_steps"],
            f"rewrote query: '{rewritten}' (model={settings.openai_chat_model})",
        ],
    }


def generate_answer(state: RagState) -> RagState:
    if not state["results"]:
        return {
            **state,
            "answer": "I do not know because no relevant context was found.",
            "sources": [],
            "workflow_steps": [
                *state["workflow_steps"],
                "skipped generation because no context was found",
            ],
        }

    context = _format_context(state["results"])
    history = state.get("conversation_history", [])
    history_block = _format_history(history)

    system_content = (
        "You are an internal company knowledge assistant.\n"
        "Answer only from the provided context. If the context does not contain the "
        "answer, say you do not know.\n"
        "Include concise citations using the source names from the context.\n"
        "When a conversation history is provided, maintain continuity — refer back "
        "to prior answers when relevant, but never invent facts not in the context."
    )
    if history_block:
        system_content += f"\n\nConversation history:\n{history_block}"

    llm = _chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=system_content),
            HumanMessage(
                content=f"Question: {state['active_question']}\n\nContext:\n{context}"
            ),
        ]
    )

    return {
        **state,
        "answer": str(response.content),
        "sources": _answer_sources(str(response.content), state["results"]),
        "workflow_steps": [
            *state["workflow_steps"],
            "generated answer from retrieved context",
        ],
    }


def check_grounding(state: RagState) -> RagState:
    answer = state["answer"].lower()
    source_names = {
        str(source.get("source", "")).lower()
        for source in state["sources"]
        if source.get("source")
    }

    cites_source = any(source_name in answer for source_name in source_names)
    says_unknown = "do not know" in answer or "don't know" in answer
    grounded = bool(state["results"]) and (cites_source or says_unknown)

    return {
        **state,
        "grounded": grounded,
        "workflow_steps": [
            *state["workflow_steps"],
            f"grounding_check grounded={grounded}",
        ],
    }


def route_after_grade(state: RagState) -> str:
    if state["needs_rewrite"]:
        return "rewrite"
    return "generate"


def _chat_model() -> ChatOpenAI:
    settings = get_settings()
    kwargs = {
        "model": settings.openai_chat_model,
        "temperature": 0,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return ChatOpenAI(**kwargs)


def _best_score(results: list[SearchResult]) -> float:
    scores = [
        result.source.score
        for result in results
        if result.source.score is not None
    ]
    return max(scores, default=0.0)


def _format_context(results: list[SearchResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        source = result.source.source
        page = f", page {result.source.page}" if result.source.page else ""
        blocks.append(f"[{index}] Source: {source}{page}\n{result.text}")
    return "\n\n".join(blocks)


def _answer_sources(answer: str, results: list[SearchResult]) -> list[dict]:
    answer_lower = answer.lower()
    cited_sources = [
        result.source.model_dump()
        for result in results
        if result.source.source.lower() in answer_lower
    ]
    if cited_sources:
        return cited_sources

    confident_sources = [
        result.source.model_dump()
        for result in results
        if result.source.score is not None and result.source.score >= MIN_RELEVANCE_SCORE
    ]
    return confident_sources[:2]


def _format_history(history: list[dict]) -> str:
    """Format conversation turns into a readable block for the LLM prompt."""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def sources_from_state(state: RagState) -> list[Source]:
    return [Source(**source) for source in state["sources"]]
