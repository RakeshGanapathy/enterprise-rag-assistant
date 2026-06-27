"""
Langfuse tracing configuration and utilities for the RAG application.
Provides context managers, decorators, and LangChain integration for observability.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Any, Generator, Optional
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

# Lazy import to avoid issues if langfuse is not installed
_langfuse_client = None


def get_langfuse_client():
    """Get or initialize Langfuse client."""
    global _langfuse_client
    if _langfuse_client is None:
        try:
            from langfuse import Langfuse

            settings = get_settings()
            
            # Build kwargs, only including base_url if it's set
            kwargs = {
                "public_key": settings.langfuse_public_key,
                "secret_key": settings.langfuse_secret_key,
            }
            if settings.langfuse_base_url:
                kwargs["base_url"] = settings.langfuse_base_url
            
            _langfuse_client = Langfuse(**kwargs)
            logger.info(
                "Initialized Langfuse client: base_url=%s project_key=%s",
                settings.langfuse_base_url,
                settings.langfuse_public_key,
            )
        except ImportError:
            raise ImportError(
                "langfuse package is required. Install it with: pip install langfuse"
            )
    return _langfuse_client


def is_tracing_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    settings = get_settings()
    # Only requires public and secret keys; base_url is optional (defaults to cloud)
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@contextmanager
def trace_span(
    name: str,
    input_data: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> Generator[dict, None, None]:
    """
    Context manager for tracing a span in Langfuse.

    Usage:
        with trace_span("operation_name", {"question": "..."}) as span:
            result = do_something()
            span["output"] = result
    """
    if not is_tracing_enabled():
        # Tracing disabled - yield empty dict
        yield {}
        return

    client = get_langfuse_client()
    
    # Create an observation/span
    observation = client.start_observation(
        name=name,
        input=input_data,
        metadata=metadata,
    )
    observation_id = getattr(observation, "observation_id", None) or getattr(observation, "id", None)
    logger.info(
        "Langfuse observation started: name=%s observation_id=%s input=%s metadata=%s",
        name,
        observation_id,
        input_data,
        metadata,
    )
    
    try:
        span_dict = {
            "observation_id": observation_id,
            "output": None,
        }
        yield span_dict

        # Update observation with output before ending it
        if span_dict.get("output") is not None:
            observation.update(output=span_dict["output"])
        observation.end()
        logger.info(
            "Langfuse observation ended: name=%s observation_id=%s output=%s",
            name,
            observation_id,
            span_dict.get("output"),
        )
    except Exception:
        observation.update(level="ERROR")
        observation.end()
        logger.exception(
            "Langfuse observation failed: name=%s observation_id=%s",
            name,
            observation_id,
        )
        raise
    finally:
        # Always flush to ensure data is sent
        client.flush()
        logger.debug("Langfuse client flushed for observation_id=%s", observation_id)


def trace_function(name: Optional[str] = None, metadata: Optional[dict] = None):
    """
    Decorator for tracing a function execution.

    Usage:
        @trace_function("my_function")
        def my_function(arg1, arg2):
            return result
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)

            func_name = name or func.__name__
            client = get_langfuse_client()

            input_data = {
                "args": str(args)[:500],  # Limit string length
                "kwargs": str(kwargs)[:500],
            }

            observation = client.start_observation(
                name=func_name,
                input=input_data,
                metadata=metadata,
            )

            try:
                result = func(*args, **kwargs)
                observation.end(output=str(result)[:500])
                return result
            except Exception as e:
                observation.end(level="ERROR")
                raise
            finally:
                client.flush()

        return wrapper

    return decorator


def trace_async_function(name: Optional[str] = None, metadata: Optional[dict] = None):
    """
    Decorator for tracing async function execution.

    Usage:
        @trace_async_function("my_async_function")
        async def my_async_function(arg1, arg2):
            return result
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return await func(*args, **kwargs)

            func_name = name or func.__name__
            client = get_langfuse_client()

            input_data = {
                "args": str(args)[:500],
                "kwargs": str(kwargs)[:500],
            }

            observation = client.start_observation(
                name=func_name,
                input=input_data,
                metadata=metadata,
            )

            try:
                result = await func(*args, **kwargs)
                observation.end(output=str(result)[:500])
                return result
            except Exception as e:
                observation.end(level="ERROR")
                raise
            finally:
                client.flush()

        return wrapper

    return decorator


def get_langchain_callbacks():
    """
    Get LangChain callbacks for Langfuse integration.
    This enables automatic tracing of LangChain LLM calls.
    """
    if not is_tracing_enabled():
        return []
    
    try:
        from langfuse.integrations.langchain import LangfuseCallbackHandler

        settings = get_settings()
        kwargs = {
            "public_key": settings.langfuse_public_key,
            "secret_key": settings.langfuse_secret_key,
        }
        if settings.langfuse_base_url:
            kwargs["base_url"] = settings.langfuse_base_url
        
        return [LangfuseCallbackHandler(**kwargs)]
    except ImportError:
        return []
