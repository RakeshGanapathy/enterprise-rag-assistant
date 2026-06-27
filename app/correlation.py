"""
Correlation ID middleware.

Assigns a unique X-Request-ID to every request. If the caller supplies one
it is reused (useful for client-side tracing). The ID is stored in a
ContextVar so any logger in the same thread/asyncio task can include it.

Usage in log formatters: %(correlation_id)s via CorrelationIdFilter.
"""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")

HEADER = "X-Request-ID"


def get_correlation_id() -> str:
    return _correlation_id_var.get()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get(HEADER) or str(uuid.uuid4())
        token = _correlation_id_var.set(correlation_id)
        try:
            response = await call_next(request)
            response.headers[HEADER] = correlation_id
            return response
        finally:
            _correlation_id_var.reset(token)


class CorrelationIdFilter(logging.Filter):
    """Inject correlation_id into every LogRecord so formatters can use %(correlation_id)s."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()  # type: ignore[attr-defined]
        return True
