from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_JWT_SECRET = "change-me-in-production"


class Settings(BaseSettings):
    app_name: str = "Enterprise RAG Knowledge Assistant"
    app_env: str = "local"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    vector_store_backend: str = "pgvector"
    vector_db_path: str = "data/vector_store.sqlite3"
    postgres_url: str = "postgresql://rag:rag@localhost:5433/rag"
    embedding_dimensions: int = 1536
    chunk_size: int = 900
    chunk_overlap: int = 150
    reranker_backend: str = "local"        # "local" | "cohere" | "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 20
    cohere_api_key: str = ""
    sync_on_startup: bool = True
    sync_interval_seconds: int = 300       # 0 = disable scheduled sync
    jwt_secret: str = _INSECURE_JWT_SECRET
    jwt_algorithm: str = "HS256"
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    cache_ttl_hours: int = 24
    cache_semantic_threshold: float = 0.92
    min_relevance_score: float = 0.25      # feedback triage threshold
    max_history_turns: int = 6             # conversation context window
    sample_docs_dir: str = "data/sample_docs"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: Optional[str] = None
    openai_retry_attempts: int = 3         # tenacity retry count for OpenAI calls
    openai_retry_min_wait: float = 1.0     # seconds
    openai_retry_max_wait: float = 10.0    # seconds
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def assert_production_ready(self) -> None:
        """Raise at startup if insecure defaults are present in non-local envs."""
        if self.app_env != "local":
            if self.jwt_secret == _INSECURE_JWT_SECRET:
                raise ValueError(
                    "JWT_SECRET is still the default insecure value. "
                    "Set a strong secret (≥32 chars) in your environment."
                )
            if len(self.jwt_secret) < 32:
                raise ValueError(
                    f"JWT_SECRET is too short ({len(self.jwt_secret)} chars). "
                    "Use at least 32 characters."
                )
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY must be set in non-local environments.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
