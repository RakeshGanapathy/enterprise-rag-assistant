import hashlib
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.access.rbac import ACCESS_LEVELS, get_role_policy
from app.config import get_settings
from app.ingestion.models import TextChunk
from app.llm import embed_query, embed_texts
from app.retrieval.models import AccessFilter
from app.tracing import trace_span


def get_embeddings():
    from app.llm import get_embeddings as _get
    return _get()

# Index a list of TextChunks by generating embeddings for their text content and storing them in the configured vector store backend (either pgvector or SQLite). The function returns the number of chunks that were indexed.
def index_chunks(chunks: list[TextChunk]) -> int:
    if not chunks:
        return 0

    with trace_span(
        name="index_chunks",
        input_data={"chunks_count": len(chunks)},
        metadata={"operation": "embedding_index"},
    ) as span:
        settings = get_settings()
        texts = [chunk.text for chunk in chunks]
        vectors = embed_texts(texts)

        if settings.vector_store_backend == "pgvector":
            _index_chunks_pgvector(chunks, vectors)
        else:
            _index_chunks_sqlite(chunks, vectors)

        span["output"] = {"chunks_indexed": len(chunks)}
        return len(chunks)

def access_filter_for_role(user_role: str) -> AccessFilter:
    policy = get_role_policy(user_role)
    return AccessFilter(
        departments=list(policy.departments),
        max_access_level=policy.max_access_level,
    )


def hybrid_search_chunks(
    question: str,
    top_k: int = 4,
    access_filter: AccessFilter | None = None,
) -> list[tuple[Document, float]]:
    """
    Three-stage enterprise retrieval pipeline:

      Stage 1a — BM25 keyword search       (recall: exact terms, codes, acronyms)
      Stage 1b — Dense vector search       (recall: semantic meaning, paraphrases)
      Stage 2  — RRF merge                 (combine both ranked lists)
      Stage 3  — Cross-encoder reranker    (precision: score query+chunk together)

    BM25 and semantic run in parallel over the same corpus for maximum recall.
    Reranker sees a wider candidate pool (reranker_top_n) then narrows to top_k.
    """
    from app.retrieval.hybrid_search import bm25_search, reciprocal_rank_fusion
    from app.retrieval.reranker import rerank

    settings = get_settings()
    candidate_k = max(settings.reranker_top_n, top_k * 4, 20)

    # Step 1: embed the query — must finish before the vector DB search can start
    query_vector = embed_query(question)

    # Step 2: semantic search and corpus load are INDEPENDENT — run both in parallel.
    #
    # Why parallel?
    #   semantic search  → pgvector HNSW index lookup  (network I/O, ~20-50ms)
    #   corpus load      → full table scan for BM25     (network I/O, ~20-80ms)
    # Neither result depends on the other, so waiting sequentially wastes time.
    # ThreadPoolExecutor lets both hit the database simultaneously.
    #
    # Two DB connections are fine: pgvector/psycopg3 connections are not shared
    # across threads; each future opens and closes its own connection.
    if settings.vector_store_backend == "pgvector":
        _search_fn = lambda: _search_chunks_pgvector(query_vector, candidate_k, access_filter)
        _corpus_fn = lambda: _load_all_docs_pgvector(access_filter)
    else:
        _search_fn = lambda: _search_chunks_sqlite(query_vector, candidate_k, access_filter)
        _corpus_fn = lambda: _load_all_docs_sqlite(access_filter)

    with ThreadPoolExecutor(max_workers=2) as pool:
        semantic_future = pool.submit(_search_fn)
        corpus_future = pool.submit(_corpus_fn)
        semantic_results = semantic_future.result()   # blocks until done
        corpus_docs = corpus_future.result()          # blocks until done

    # Step 3: BM25 on full corpus (CPU-bound, runs after corpus is loaded)
    corpus_texts = [doc.page_content for doc in corpus_docs]
    keyword_ranking = bm25_search(question, corpus_texts, top_k=candidate_k)

    # Step 4: RRF — merge both ranked lists into one candidate pool
    rrf_candidates = reciprocal_rank_fusion(
        semantic_ranking=semantic_results,
        keyword_ranking=keyword_ranking,
        corpus_docs=corpus_docs,
        top_k=candidate_k,
    )

    # Step 5: cross-encoder reranker — precision pass over the candidate pool
    return rerank(question, rrf_candidates, top_k=top_k)


# Search for relevant chunks in the vector store by comparing the query embedding with stored chunk embeddings using cosine similarity. The top_k most similar chunks are returned as a list of tuples containing the Document and its similarity score.
def search_chunks(
    question: str,
    top_k: int = 4,
    access_filter: AccessFilter | None = None,
) -> list[tuple[Document, float]]:
    with trace_span(
        name="search_chunks",
        input_data={"question": question, "top_k": top_k},
        metadata={"operation": "vector_search"},
    ) as span:
        settings = get_settings()
        query_vector = embed_query(question)

        if settings.vector_store_backend == "pgvector":
            results = _search_chunks_pgvector(query_vector, top_k, access_filter)
        else:
            results = _search_chunks_sqlite(query_vector, top_k, access_filter)
        # Note: for hybrid mode, callers should use hybrid_search_chunks() instead

        span["output"] = {
            "results_count": len(results),
            "top_score": results[0][1] if results else None,
        }
        return results


def _delete_chunks_by_source(connection, source: str) -> None:
    """Delete all chunks belonging to a source document. Called before re-indexing."""
    settings = get_settings()
    if settings.vector_store_backend == "pgvector":
        connection.execute(
            "DELETE FROM chunks WHERE metadata_json->>'source' = %s",
            (source,),
        )
    else:
        connection.execute(
            "DELETE FROM chunks WHERE json_extract(metadata_json, '$.source') = ?",
            (source,),
        )


def reset_vector_store() -> None:
    settings = get_settings()
    if settings.vector_store_backend == "pgvector":
        with _connect_pgvector() as connection:
            _create_table_pgvector(connection)
            connection.execute("DELETE FROM chunks")
    else:
        with _connect_sqlite() as connection:
            _create_table_sqlite(connection)
            connection.execute("DELETE FROM chunks")


def list_stored_chunks(limit: int = 20) -> list[dict]:
    settings = get_settings()
    if settings.vector_store_backend == "pgvector":
        with _connect_pgvector() as connection:
            _create_table_pgvector(connection)
            rows = connection.execute(
                """
                SELECT id, text, metadata_json, embedding::text
                FROM chunks
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": chunk_id,
                "text": text,
                "metadata_json": metadata,
                "embedding_backend": "pgvector",
                "embedding_preview": embedding_text[:120],
            }
            for chunk_id, text, metadata, embedding_text in rows
        ]

    with _connect_sqlite() as connection:
        _create_table_sqlite(connection)
        rows = connection.execute(
            """
            SELECT id, text, metadata_json, embedding_json
            FROM chunks
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    chunks = []
    for chunk_id, text, metadata_json, embedding_json in rows:
        embedding = json.loads(embedding_json)
        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata_json": json.loads(metadata_json),
                "embedding_dimensions": len(embedding),
                "embedding_preview": embedding[:8],
            }
        )

    return chunks

# Index the text chunks by embedding them and storing the text, metadata, and embeddings in a SQLite database. The function returns the number of chunks indexed. If there are no chunks to index, it returns 0 without performing any database operations. --- IGNORE ---
def _index_chunks_sqlite(chunks: list[TextChunk], vectors: list[list[float]]) -> None:
    with _connect_sqlite() as connection:
        _create_table_sqlite(connection)
        for chunk, vector in zip(chunks, vectors):
            chunk_id = _chunk_id(chunk)
            connection.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (id, text, metadata_json, embedding_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk.text,
                    json.dumps(chunk.metadata),
                    json.dumps(vector),
                ),
            )

# Search for relevant chunks based on a question by embedding the question and comparing it to the stored chunk embeddings using cosine similarity. The top K most similar chunks are returned as a list of tuples containing the Document and its similarity score. --- IGNORE ---
def _search_chunks_sqlite(
    query_vector: list[float],
    top_k: int,
    access_filter: AccessFilter | None = None,
) -> list[tuple[Document, float]]:
    with _connect_sqlite() as connection:
        _create_table_sqlite(connection)
        rows = connection.execute(
            "SELECT text, metadata_json, embedding_json FROM chunks"
        ).fetchall()

    scored_documents: list[tuple[Document, float]] = []
    for text, metadata_json, embedding_json in rows:
        metadata = json.loads(metadata_json)
        if access_filter and not _passes_access_filter(metadata, access_filter):
            continue
        vector = json.loads(embedding_json)
        score = _cosine_similarity(query_vector, vector)
        document = Document(page_content=text, metadata=metadata)
        scored_documents.append((document, score))

    scored_documents.sort(key=lambda item: item[1], reverse=True)
    return scored_documents[:top_k]

# Index the text chunks by embedding them and storing the text, metadata, and embeddings in a PostgreSQL database using the pgvector extension. The function returns the number of chunks indexed. If there are no chunks to index, it returns 0 without performing any database operations. --- IGNORE ---
def _index_chunks_pgvector(chunks: list[TextChunk], vectors: list[list[float]]) -> None:
    with _connect_pgvector() as connection:
        _create_table_pgvector(connection)
        for chunk, vector in zip(chunks, vectors):
            chunk_id = _chunk_id(chunk)
            connection.execute(
                """
                INSERT INTO chunks (id, text, metadata_json, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata_json = EXCLUDED.metadata_json,
                    embedding = EXCLUDED.embedding
                """,
                (
                    chunk_id,
                    chunk.text,
                    json.dumps(chunk.metadata),
                    vector,
                ),
            )

# Search for relevant chunks based on a question by embedding the question and comparing it to the stored chunk embeddings using cosine similarity. The top K most similar chunks are returned as a list of tuples containing the Document and its similarity score. --- IGNORE ---
def _search_chunks_pgvector(
    query_vector: list[float],
    top_k: int,
    access_filter: AccessFilter | None = None,
) -> list[tuple[Document, float]]:
    where_clause, params = _build_pgvector_where(access_filter)
    with _connect_pgvector() as connection:
        _create_table_pgvector(connection)
        rows = connection.execute(
            f"""
            SELECT
                text,
                metadata_json,
                1 - (embedding <=> %s::vector) AS score
            FROM chunks
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, *params, query_vector, top_k),
        ).fetchall()

    return [
        (Document(page_content=text, metadata=metadata), float(score))
        for text, metadata, score in rows
    ]


def _load_all_docs_sqlite(access_filter: AccessFilter | None = None) -> list[Document]:
    """Load every accessible chunk from SQLite, used by BM25."""
    with _connect_sqlite() as connection:
        _create_table_sqlite(connection)
        rows = connection.execute("SELECT text, metadata_json FROM chunks").fetchall()
    return [
        Document(page_content=text, metadata=json.loads(metadata_json))
        for text, metadata_json in rows
        if not access_filter or _passes_access_filter(json.loads(metadata_json), access_filter)
    ]


def _load_all_docs_pgvector(access_filter: AccessFilter | None = None) -> list[Document]:
    """Load every accessible chunk from pgvector, used by BM25."""
    where_clause, params = _build_pgvector_where(access_filter)
    with _connect_pgvector() as connection:
        _create_table_pgvector(connection)
        rows = connection.execute(
            f"SELECT text, metadata_json FROM chunks {where_clause}",
            params,
        ).fetchall()
    return [Document(page_content=text, metadata=metadata) for text, metadata in rows]


def _passes_access_filter(metadata: dict, f: AccessFilter) -> bool:
    """
    In-process filter for SQLite backend.
    A chunk passes when BOTH conditions hold:
      1. its department is in the allowed set, OR the role has 'all' access
      2. its access_level numeric value <= role's max_access_level
    """
    chunk_dept = metadata.get("department", "general")
    chunk_level_str = metadata.get("access_level", "internal")
    chunk_level = ACCESS_LEVELS.get(chunk_level_str, 1)

    dept_ok = "all" in f.departments or chunk_dept in f.departments
    level_ok = chunk_level <= f.max_access_level
    return dept_ok and level_ok


def _build_pgvector_where(f: AccessFilter | None) -> tuple[str, tuple]:
    """
    Build a SQL WHERE clause for pgvector access filtering.
    Runs inside the DB — the HNSW index scan only touches rows the user can see.

    Department check: 'all' in role departments grants cross-department access.
    Access level check: chunk's level must be <= role's max level.
    """
    if f is None:
        return "", ()

    if "all" in f.departments:
        # Dept unrestricted — only filter by access level
        return (
            "WHERE (metadata_json->>'access_level') IN %s",
            (_allowed_level_labels(f.max_access_level),),
        )

    return (
        "WHERE metadata_json->>'department' IN %s "
        "AND (metadata_json->>'access_level') IN %s",
        (
            tuple(f.departments),
            _allowed_level_labels(f.max_access_level),
        ),
    )


def _allowed_level_labels(max_level: int) -> tuple[str, ...]:
    """Return all access_level label strings with numeric value <= max_level."""
    return tuple(
        label for label, value in ACCESS_LEVELS.items() if value <= max_level
    )


# Connect to the SQLite database specified in the settings. If the database file or its parent directories do not exist, they will be created automatically. The function returns a connection object that can be used to execute SQL commands against the SQLite database.
def _connect_sqlite() -> sqlite3.Connection:
    settings = get_settings()
    db_path = Path(settings.vector_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)

# Create the chunks table if it doesn't exist. The table has columns for a unique ID, the chunk text, metadata as JSON, and the embedding vector as JSON. This allows us to store all necessary information for retrieval and similarity search in a single table.
def _create_table_sqlite(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        )
        """
    )


def _connect_pgvector():
    """Return a pooled connection context manager."""
    from app.db import get_conn
    return get_conn()


def _create_table_pgvector(connection) -> None:
    settings = get_settings()
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            metadata_json JSONB NOT NULL,
            embedding vector({settings.embedding_dimensions}) NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )

# Generate a unique ID for each chunk based on its source, chunk index, and text content. This helps to prevent duplicates in the vector store and allows for easy retrieval of the original chunk metadata when
def _chunk_id(chunk: TextChunk) -> str:
    source = str(chunk.metadata.get("source", "unknown"))
    chunk_index = str(chunk.metadata.get("chunk_index", "0"))
    raw = f"{source}:{chunk_index}:{chunk.text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# Cosine similarity function to compare query vectors with stored chunk vectors.
def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)
