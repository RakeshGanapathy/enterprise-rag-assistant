# Enterprise RAG Knowledge Assistant

A production-grade Retrieval-Augmented Generation system built with FastAPI, LangGraph, and pgvector. Built as a 4-day crash course progressively extended with enterprise features.

---

## Architecture

```
Client
  │
  ├── POST /ask/stream          Server-Sent Events (streaming tokens)
  ├── POST /ask                 Full RAG answer
  ├── POST /search              Retrieval only
  └── POST /feedback            Thumbs up / down
        │
        ▼
  JWT Auth + Rate Limiting      (60 req/min per user, domain + actions → AccessFilter)
        │
        ▼
  Query Router                  Deterministic NLP: semantic | hybrid
        │
        ├── Semantic path       Dense vector search (pgvector HNSW)
        └── Hybrid path         BM25 + Dense vector, parallel, merged with RRF
                │
                ▼
        Cross-Encoder Reranker  (local sentence-transformers or Cohere)
                │
                ▼
        Context-Hash Cache      pgvector — shared across roles when chunks are same
                │
                ├── HIT         Return cached answer (no LLM call)
                └── MISS        LangGraph RAG Workflow
                                  retrieve → grade → rewrite? → generate → ground
                                        │
                                        ▼
                                Conversation History   (6-turn window, pgvector)
                                        │
                                        ▼
                                Streaming SSE Response
```

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Workflow | LangGraph (stateful graph) |
| Vector DB | pgvector (HNSW index, JSONB metadata) |
| Sparse search | rank-bm25 (BM25Okapi) |
| Reranker | sentence-transformers cross-encoder / Cohere |
| LLM | OpenAI-compatible (gpt-4o-mini default) |
| Embeddings | text-embedding-3-small (1536 dims) |
| Auth | JWT (python-jose) — domain + actions claims |
| Tracing | Langfuse |
| Evaluation | RAGAS |

---

## Features

### Retrieval Pipeline (5 stages)

```
1. Query Router       — classifies query: keyword → hybrid, conceptual → semantic
2. BM25 Search        — sparse keyword retrieval (exact terms, codes, acronyms)
3. Dense Search       — pgvector cosine similarity (HNSW index)
4. RRF Merge          — Reciprocal Rank Fusion: 1/(60+rank_bm25) + 1/(60+rank_dense)
5. Cross-Encoder      — reranks candidate pool, returns top_k
```

BM25 and dense vector calls run in parallel via `ThreadPoolExecutor`.

### LangGraph Workflow

```
retrieve_context → grade_context → [needs rewrite?]
                                        │ yes
                                   rewrite_query → retrieve_context
                                        │ no
                                   generate_answer → check_grounding → END
```

### JWT Authentication

Token payload:
```json
{
  "sub": "rakesh@company.com",
  "domain": "hr",
  "actions": ["read:public", "read:internal", "read:confidential"],
  "exp": 1750000000
}
```

`domain` → department filter in pgvector WHERE clause
`actions` → max access level (highest level present in the list)

### RBAC

Two-dimension filtering runs inside pgvector before the HNSW scan:

```sql
WHERE metadata_json->>'department' = ANY(%s)
  AND (metadata_json->>'access_level_int')::int <= %s
```

Access levels: `public=0`, `internal=1`, `confidential=2`, `restricted=3`

### Semantic Answer Cache

Keyed on `SHA-256(question + context_hash)` where `context_hash = SHA-256(sorted chunk IDs)`.

Roles with different access filters sharing the same retrieved chunks share the same cache entry — one LLM call serves both.

Two tiers:
- Exact hash match → sub-millisecond
- Semantic similarity (pgvector cosine > 0.92) → ~100ms (embedding only, no LLM)

### Multi-turn Conversation

`conversation_id` returned on first turn, sent back on follow-ups. Last 6 turns injected into rewrite and generation prompts. Full history stored in pgvector for audit.

### Document Ingestion

**Supported formats:** `.txt`, `.md`, `.pdf`, `.docx`

PDF and Word tables are extracted as structured markdown chunks — never split mid-row.

**Change detection (two-tier):**
```
Tier 1 — mtime check  (free OS syscall, no file read)
  mtime unchanged → skip entirely

Tier 2 — content hash  (read file, SHA-256 of extracted text)
  hash unchanged → update mtime only, skip re-index
  hash changed   → delete old chunks atomically → re-index
```

**Event-driven ingestion (production):**
```
S3 upload → S3 Event → Lambda → POST /documents/ingest-s3
  Lambda reads object tags (department, access_level)
  Lambda generates presigned URL
  RAG API downloads via presigned URL, runs change detection
```

### Rate Limiting

Fixed window counter per JWT `sub`, stored in pgvector. Default: 60 req/min.
Returns `429 Too Many Requests` with `Retry-After` header.
Applied to `/search`, `/ask`, `/ask/stream` only.

### User Feedback

```
POST /feedback  →  { question, rating: "positive"|"negative", answer, sources, comment }
```

Auto-triage on negative ratings:
- `max(source.score) < 0.25` → `failure_mode: "retrieval"` (wrong chunks)
- `max(source.score) >= 0.25` → `failure_mode: "generation"` (hallucination)

### Evaluation (RAGAS)

```bash
python tests/run_evaluation.py           # compare semantic vs hybrid
python tests/run_evaluation.py --mode hybrid
```

Four metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`

---

## Local Setup

### Prerequisites

- Python 3.11+
- Docker (for pgvector)
- Git

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag.git
cd enterprise-rag
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start pgvector

```bash
docker-compose -f docker-compose.pgvector.yml up -d
```

Verify:
```bash
docker ps  # should show postgres container running on port 5433
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — minimum required:
```env
OPENAI_API_KEY=sk-...
POSTGRES_URL=postgresql://rag:rag@localhost:5433/rag
```

Full `.env.example` documents all options.

### 5. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

On startup the API:
- Runs `sync_directory("data/sample_docs")` to index any new/changed documents
- Starts a background sync task (every 5 minutes by default)

### 6. Ingest sample documents

```bash
curl -X POST http://localhost:8000/documents/ingest-samples
```

---

## API Reference

### Auth

All retrieval endpoints require `Authorization: Bearer <token>`.

Generate a test token:
```python
from jose import jwt
import time

token = jwt.encode(
    {
        "sub": "you@company.com",
        "domain": "hr",
        "actions": ["read:public", "read:internal", "read:confidential"],
        "exp": int(time.time()) + 3600,
    },
    "change-me-in-production",
    algorithm="HS256",
)
print(token)
```

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | Health check |
| POST | `/documents/ingest-samples` | No | Index sample docs folder |
| POST | `/documents/upload` | No | Upload a file (async job) |
| GET | `/documents/status/{job_id}` | No | Poll async job |
| POST | `/documents/sync` | No | Scan folder for changes |
| POST | `/documents/ingest-s3` | No | S3 event trigger (Lambda) |
| GET | `/documents` | No | List indexed documents |
| POST | `/search` | JWT + rate limit | Retrieval only |
| POST | `/ask` | JWT + rate limit | Full RAG answer |
| POST | `/ask/stream` | JWT + rate limit | Streaming SSE answer |
| POST | `/feedback` | JWT | Submit rating |
| GET | `/feedback/summary` | No | Aggregate stats |
| GET | `/feedback/triage` | No | Negative feedback list |
| GET | `/conversations/{id}` | No | Conversation history |
| GET | `/cache/stats` | No | Cache hit counts |
| DELETE | `/cache` | No | Flush cache |
| GET | `/debug/chunks` | No | Inspect stored chunks |

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the PTO policy?",
    "top_k": 4,
    "search_mode": "auto",
    "conversation_id": null
  }'
```

Response:
```json
{
  "answer": "Full-time employees receive 20 paid time off days...",
  "sources": [{"source": "hr_policy.md", "page": null, "score": 0.91}],
  "grounded": true,
  "workflow_steps": ["router: hybrid", "retrieved 4 chunks", "generated answer"],
  "conversation_id": "abc-123"
}
```

### POST /ask/stream (SSE)

```bash
curl -X POST http://localhost:8000/ask/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the PTO policy?"}' \
  --no-buffer
```

Event stream:
```
data: {"type": "step", "text": "retrieved 4 chunks via hybrid + reranked via local"}
data: {"type": "token", "text": "Full-time"}
data: {"type": "token", "text": " employees"}
data: {"type": "done", "sources": [...], "grounded": true}
```

### POST /feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the PTO policy?",
    "rating": "negative",
    "answer": "Employees get 15 days...",
    "comment": "Wrong — it is 20 days",
    "sources": [{"source": "hr_policy.md", "score": 0.85}]
  }'
```

---

## Testing

### Run RAGAS evaluation

```bash
python tests/run_evaluation.py
```

Compare semantic vs hybrid on 12 golden questions. Reports saved to `tests/eval_reports/`.

### Manual smoke test

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Ingest
curl -X POST http://localhost:8000/documents/ingest-samples

# 3. List documents
curl http://localhost:8000/documents

# 4. Generate token (Python)
python -c "
from jose import jwt; import time
print(jwt.encode({'sub':'test@co.com','domain':'hr','actions':['read:public','read:internal','read:confidential'],'exp':int(time.time())+3600},'change-me-in-production',algorithm='HS256'))
"

# 5. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many PTO days do employees get?"}'

# 6. Check cache stats
curl http://localhost:8000/cache/stats

# 7. Ask same question again (should be cache hit)
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many PTO days do employees get?"}'

# 8. Multi-turn follow-up
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What about sick leave?", "conversation_id": "CONV_ID_FROM_STEP_5"}'
```

### Pytest

```bash
pytest tests/ -v
```

---

## Production Ingestion: S3 + Lambda

```
Document author uploads file to S3 with tags: department=hr, access_level=confidential
        ↓
S3 Event Notification (ObjectCreated / ObjectRemoved)
        ↓
Lambda (lambda/s3_ingest_trigger.py)
  reads object tags, generates presigned URL
  calls POST /documents/ingest-s3
        ↓
RAG API downloads via presigned URL
  runs two-tier change detection
  if changed: delete old chunks → re-index
  returns { job_id }
```

Lambda needs `s3:GetObject` IAM permission. RAG API needs no S3 credentials.

---

## Authentication (JWT)

```
JWT_SECRET="your-secret-key-min-32-chars"
JWT_ALGORITHM="HS256"

action → access level:
  "read:public"        → 0
  "read:internal"      → 1
  "read:confidential"  → 2
  "read:restricted"    → 3
```

`domain=admin` or `domain=all` grants cross-department access.

---

## RBAC

```
Role            Domain          Max Level   Sees
hr_staff        hr              internal    hr docs (public + internal)
hr_manager      hr              confidential  all hr docs
security_engineer  security     restricted  security + cross-dept
admin           admin           restricted  everything
employee        hr, product     public      public docs only
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_BASE_URL` | — | Optional (OpenRouter etc) |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Generation model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `POSTGRES_URL` | `postgresql://rag:rag@localhost:5433/rag` | pgvector connection |
| `RERANKER_BACKEND` | `local` | `local` / `cohere` / `none` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local reranker |
| `RERANKER_TOP_N` | `20` | Candidate pool for reranker |
| `JWT_SECRET` | `change-me-in-production` | **Change this** |
| `JWT_ALGORITHM` | `HS256` | |
| `RATE_LIMIT_REQUESTS` | `60` | Per user per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Window size |
| `CACHE_TTL_HOURS` | `24` | Answer cache TTL |
| `CACHE_SEMANTIC_THRESHOLD` | `0.92` | Cosine similarity floor |
| `SYNC_ON_STARTUP` | `true` | Sync docs folder on API start |
| `SYNC_INTERVAL_SECONDS` | `300` | Background sync interval (0=off) |
| `LANGFUSE_PUBLIC_KEY` | — | Optional tracing |
| `LANGFUSE_SECRET_KEY` | — | Optional tracing |

---

## Project Structure

```
app/
├── main.py                  FastAPI app, all endpoints
├── config.py                Pydantic settings
├── middleware.py             Langfuse tracing middleware
├── access/
│   └── rbac.py              Role policies, department/level maps
├── auth/
│   ├── dependencies.py      require_auth / optional_auth FastAPI deps
│   ├── jwt.py               Token decode, claims → AccessFilter
│   └── rate_limit.py        Fixed-window counter, require_rate_limit dep
├── cache/
│   └── query_cache.py       Two-tier answer cache (exact + semantic)
├── conversation/
│   └── store.py             pgvector-backed conversation history
├── evaluation/
│   └── runner.py            RAGAS evaluation runner
├── feedback/
│   └── store.py             Feedback store + failure mode triage
├── graph/
│   ├── nodes.py             LangGraph nodes
│   ├── state.py             RagState TypedDict
│   └── workflow.py          Graph assembly + run_rag_workflow
├── ingestion/
│   ├── chunking.py          RecursiveCharacterTextSplitter (tables kept intact)
│   ├── document_store.py    documents + ingest_jobs tables
│   ├── loaders.py           txt / md / pdf (pdfplumber) / docx loaders
│   ├── models.py            IngestionResult, SourceDocument, TextChunk
│   └── pipeline.py          ingest_file, sync_directory, ingest_from_s3
└── retrieval/
    ├── hybrid_search.py     BM25Okapi + RRF
    ├── models.py            Pydantic request/response models
    ├── qa.py                answer_question, search_knowledge_base
    ├── query_router.py      Deterministic NLP classifier
    ├── reranker.py          CrossEncoder / Cohere / none
    ├── streaming.py         SSE async generator
    └── vector_store.py      pgvector CRUD + RBAC filtering

lambda/
└── s3_ingest_trigger.py     AWS Lambda — S3 event → RAG API

tests/
├── eval_dataset.json        12 golden Q&A pairs
├── eval_reports/            RAGAS JSON reports (gitignored)
└── run_evaluation.py        CLI evaluation script
```

