"""
Central database connection pool.

A single ConnectionPool is created at startup and shared across all requests.
Every caller uses `get_conn()` as a context manager — it borrows a connection
from the pool and returns it automatically on exit.

Why a pool?
  Without pooling, every request calls psycopg.connect() which opens a new TCP
  connection to Postgres (~5-20ms). Under concurrent load this exhausts the
  Postgres max_connections limit fast. The pool keeps a fixed number of
  connections open and reuses them across requests.
"""

from contextlib import contextmanager

from psycopg_pool import ConnectionPool

from app.config import get_settings

_pool: ConnectionPool | None = None


def init_pool() -> None:
    """Create the connection pool. Called once at application startup."""
    global _pool
    settings = get_settings()
    _pool = ConnectionPool(
        conninfo=settings.postgres_url,
        min_size=2,
        max_size=10,
        open=True,
    )
    # Register pgvector type on every new connection
    from pgvector.psycopg import register_vector
    with _pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)


def close_pool() -> None:
    """Close the pool on shutdown."""
    global _pool
    if _pool:
        _pool.close()
        _pool = None


@contextmanager
def get_conn():
    """Borrow a connection from the pool. Returns it on exit."""
    if _pool is None:
        raise RuntimeError("DB pool not initialised — call init_pool() at startup")
    with _pool.connection() as conn:
        yield conn
