from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Application settings using Pydantic BaseSettings, with environment variable support 
# and caching for efficient access
class Settings(BaseSettings):
    app_name: str = "Enterprise RAG Knowledge Assistant"
    app_env: str = "local"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    vector_store_backend: str = "sqlite"
    vector_db_path: str = "data/vector_store.sqlite3"
    postgres_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_dimensions: int = 1536
    chunk_size: int = 900
    chunk_overlap: int = 150
    reranker_backend: str = "local"        # "local" | "cohere" | "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 20               # candidate pool fed into reranker
    cohere_api_key: str = ""
    sync_on_startup: bool = True           # scan docs folder when API starts
    sync_interval_seconds: int = 300       # 0 = disable scheduled sync
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    rate_limit_requests: int = 60          # max requests per window per user
    rate_limit_window_seconds: int = 60    # window size in seconds
    cache_ttl_hours: int = 24              # how long a cached answer stays valid
    cache_semantic_threshold: float = 0.92 # cosine similarity floor for a cache hit
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
