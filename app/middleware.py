"""
FastAPI middleware for Langfuse tracing.
Automatically traces all HTTP requests and responses.
"""

from datetime import datetime
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.tracing import is_tracing_enabled, trace_span


class LangfuseTracingMiddleware(BaseHTTPMiddleware):
    """Middleware to trace FastAPI requests with Langfuse."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with tracing."""
        if not is_tracing_enabled():
            return await call_next(request)

        # Prepare request metadata
        request_data = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
        }

        metadata = {
            "http.method": request.method,
            "http.url": str(request.url),
            "timestamp": datetime.now().isoformat(),
        }

        with trace_span(
            name=f"{request.method} {request.url.path}",
            input_data=request_data,
            metadata=metadata,
        ) as span:
            try:
                response = await call_next(request)
                span["output"] = {
                    "status_code": response.status_code,
                }
                # If an observation id was created, attach it to the response headers
                obs_id = span.get("observation_id")
                if obs_id:
                    # Ensure header value is a string
                    response.headers["X-Langfuse-Observation-Id"] = str(obs_id)
                return response
            except Exception as e:
                span["output"] = {
                    "error": str(e),
                }
                obs_id = span.get("observation_id")
                if obs_id:
                    # When an exception occurs, try to include the observation id as well
                    # Note: raising the exception will still bubble up after this
                    # but adding the header can help callers correlate traces.
                    # Create a minimal Response if one isn't available yet.
                    # Starlette/FastAPI will convert exceptions to responses downstream,
                    # but we attempt to set the header on the response if present.
                    # Since we don't have a response object here, we can't set headers,
                    # so we simply re-raise after updating the observation.
                  raise
