from functools import lru_cache

from fastapi import HTTPException
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.graph.nodes import (
    check_grounding,
    generate_answer,
    grade_context,
    retrieve_context,
    rewrite_query,
    route_after_grade,
    sources_from_state,
)
from app.graph.state import RagState
from app.retrieval.models import AskResponse
from app.tracing import trace_span


def run_rag_workflow(
    question: str,
    top_k: int = 4,
    search_mode: str = "auto",
    access_filter=None,
    conversation_history: list[dict] | None = None,
    preloaded_matches=None,   # list[(Document, score)] already retrieved by qa.py
) -> AskResponse:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing. Add it to .env before asking questions.",
        )

    with trace_span(
        name="rag_workflow",
        input_data={"question": question, "top_k": top_k},
        metadata={
            "operation": "rag_workflow",
            "top_k": top_k,
        },
    ) as span:
        initial_state: RagState = {
            "question": question,
            "active_question": question,
            "top_k": top_k,
            "search_mode": search_mode,
            "access_filter": access_filter,
            "conversation_history": conversation_history or [],
            "preloaded_matches": preloaded_matches,
            "attempts": 0,
            "results": [],
            "answer": "",
            "sources": [],
            "needs_rewrite": False,
            "grounded": False,
            "workflow_steps": ["started LangGraph RAG workflow"],
        }

        final_state = get_rag_graph().invoke(initial_state)
        response = AskResponse(
            answer=final_state["answer"],
            sources=sources_from_state(final_state),
            rewritten_question=(
                final_state["active_question"]
                if final_state["active_question"] != final_state["question"]
                else None
            ),
            grounded=final_state["grounded"],
            workflow_steps=final_state["workflow_steps"],
        )
        
        span["output"] = {
            "answer_length": len(response.answer),
            "sources_count": len(response.sources),
            "grounded": response.grounded,
            "workflow_steps": len(response.workflow_steps),
        }
        
        return response


@lru_cache
def get_rag_graph():
    graph = StateGraph(RagState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("grade", grade_context)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("generate", generate_answer)
    graph.add_node("grounding_check", check_grounding)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "rewrite": "rewrite",
            "generate": "generate",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "grounding_check")
    graph.add_edge("grounding_check", END)

    return graph.compile()

