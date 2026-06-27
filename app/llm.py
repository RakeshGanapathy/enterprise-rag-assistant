"""
Resilient LLM and embedding clients with automatic retry.

All OpenAI calls go through these helpers so transient errors (429 rate limit,
503 overload, network blips) are retried with exponential backoff instead of
surfacing as a 500 to the caller.
"""

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


def _retry_decorator():
    settings = get_settings()
    return retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(settings.openai_retry_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=settings.openai_retry_min_wait,
            max=settings.openai_retry_max_wait,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def get_embeddings():
    """Return an OpenAIEmbeddings instance. Raises on missing API key."""
    from fastapi import HTTPException
    from langchain_openai import OpenAIEmbeddings

    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing. Add it to .env before using the API.",
        )
    kwargs = {
        "model": settings.openai_embedding_model,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


@_retry_decorator()
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with retry on transient OpenAI errors."""
    return get_embeddings().embed_documents(texts)


@_retry_decorator()
def embed_query(text: str) -> list[float]:
    """Embed a single query string with retry."""
    return get_embeddings().embed_query(text)


def get_llm():
    """Return a ChatOpenAI instance."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    kwargs = {
        "model": settings.openai_chat_model,
        "api_key": settings.openai_api_key,
        "streaming": True,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)
