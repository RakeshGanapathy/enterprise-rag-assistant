# Enterprise RAG — Interview Preparation

Challenging questions an interviewer will ask about this system, with full answers.

---

## Retrieval Architecture

**Q: Why did you choose hybrid search over pure semantic search?**

Semantic search embeds both query and document into a shared vector space and retrieves by cosine similarity. It captures conceptual meaning well but struggles with exact matches — policy IDs like `HR-204`, version strings like `v2.3.1`, acronyms, and proper nouns often have embeddings far from their document mentions because they are rare or out-of-vocabulary tokens.

BM25 is a term frequency-inverse document frequency variant that excels exactly where semantic search fails: exact keyword matches, rare terms, codes. Running both in parallel and merging with RRF means we get recall from both retrieval modes. In benchmarks, hybrid typically outperforms either alone by 10–20% on mixed enterprise corpora.

---

**Q: Why RRF instead of score normalisation to merge BM25 and semantic results?**

BM25 scores are unbounded (depend on corpus statistics) and semantic scores are cosine similarities in [-1, 1]. Normalising them to the same scale requires knowing the min/max of the current result set, which is unstable across queries and introduces sensitivity to outliers.

RRF uses only rank positions: `score = 1/(60 + rank)`. It is parameter-free beyond the constant 60, which controls how much top positions are rewarded. It is robust to score scale differences and consistently outperforms normalisation approaches in the literature. The constant 60 was proposed in the original RRF paper and works well in practice.

---

**Q: What is the cross-encoder reranker doing that the bi-encoder didn't?**

The bi-encoder (embedding model) encodes the query and each document independently, then measures cosine similarity. This is fast — embeddings are precomputed — but the query and document never interact during encoding.

The cross-encoder takes the concatenated (query, document) pair and scores them jointly. This allows full attention across both, so the model can see exactly which terms in the document answer which part of the query. It is much more accurate but cannot precompute — you must run it at query time for every candidate.

We use the bi-encoder to retrieve a wide candidate pool (top-N), then the cross-encoder to precisely rerank that pool to top-K. This gives accuracy close to cross-encoder-only at a fraction of the cost.

---

**Q: Your BM25 builds a corpus from the full document set. What happens as the corpus grows to millions of chunks?**

BM25Okapi from rank-bm25 builds an in-memory index. At millions of chunks this becomes impractical — memory usage, indexing time on startup, and scoring latency all grow linearly.

At scale the right approach is:
1. Elasticsearch or OpenSearch with BM25 built-in — distributed, persistent, efficient at billions of documents
2. pgvector for dense search alongside
3. RRF merge in the application layer (same as current)

For the current system (thousands of enterprise documents) in-memory BM25 is fine. The migration path to Elasticsearch is clean because the interface is the same: `bm25_search(query) → [(doc, score)]`.

---

**Q: Why does the query router use deterministic NLP instead of an LLM call?**

An LLM routing call would add 300–800ms latency before every query and cost money on every request. For a classification task with 3 outputs (semantic / hybrid / keyword) a deterministic approach works reliably:

- Quoted phrases → exact search needed → hybrid
- Pattern matches: policy IDs `[A-Z]+-\d+`, version strings `v\d+.\d+`, all-caps codes → keyword signals → hybrid
- Very short queries (≤6 words) → probably a lookup → hybrid
- Long queries (≥15 words) → probably conceptual → semantic
- Default → hybrid (the safe enterprise choice)

This runs in microseconds with no external call and is fully auditable — you can see exactly why a query was routed. LLM routing adds latency, cost, and non-determinism for no measurable gain on this classification task.

---

## Access Control

**Q: Why does the RBAC filter run inside pgvector rather than post-retrieval in Python?**

Post-retrieval filtering retrieves top-K chunks, then removes unauthorised ones in Python. This means:
1. You might return fewer than top-K results to the user (some were filtered)
2. Unauthorised chunks were loaded into application memory
3. The vector similarity search scanned chunks the user was never allowed to see

Filtering inside the pgvector WHERE clause means the HNSW index only considers authorised chunks. Unauthorised chunks are never loaded, never scored, never returned. The user always gets exactly top-K authorised results. This is the correct security model — the vector DB enforces access, not the application.

---

**Q: How does the JWT domain + actions model map to RBAC? Why not just send a role?**

A role like `hr_manager` requires the RAG API to maintain a role-policy lookup table that must stay synchronised with the identity provider. When the HR team restructures, both the IdP and the RAG API config need updating.

`domain` + `actions` are issued directly by the identity provider (Okta, Auth0, Azure AD) as standard claims. The RAG API maps them mechanically:
- `domain` → department filter (no lookup needed)
- `actions` → max access level (take the highest level present in the list)

No role table to maintain. The IdP is the single source of truth. Adding a new department means tagging documents in S3 and issuing tokens with the new domain — no code change in the RAG API.

---

**Q: What is the security risk of the current rate limiter and how would you fix it in production?**

The rate limiter stores counters in pgvector with a fixed window per JWT `sub`. The risk is that `sub` comes from the JWT payload which we trust after signature verification — that part is fine.

The real risk is the fixed-window edge case: a user can make 60 requests at 23:59:59 and 60 more at 00:00:00 — 120 requests in 2 seconds. For an enterprise internal tool this is acceptable. To eliminate it, use a sliding window with a Redis sorted set (ZADD timestamps, ZREMRANGEBYSCORE for the window, ZCARD for count). Redis is the standard production solution because it supports atomic operations across distributed API instances; pgvector counters are per-instance and race under horizontal scaling.

---

## Caching

**Q: Why is the cache key based on retrieved chunk IDs rather than the access filter?**

Two users with different roles (HR staff and Admin) asking the same question may retrieve identical chunks — both have access to `hr_policy.md`. If the cache key includes the access filter fingerprint, they get two separate cache entries and two LLM calls, even though the answer is identical.

By keying on `SHA-256(sorted chunk IDs)`, the cache entry represents "this question + this exact context". Any user who retrieves the same chunks gets the same answer. RBAC already ran before retrieval — the chunk set is already the authorised view. The cache serves the answer for that chunk set to anyone who retrieves it.

---

**Q: What is the execution order now that retrieval must happen before the cache check?**

```
1. Retrieve chunks (always — RBAC runs here, ~200ms)
2. Hash chunk IDs → context_hash → cache lookup
3. Cache hit  → return immediately (no LLM call)
4. Cache miss → LLM generation (~2000ms) → store in cache
```

You always pay retrieval cost. The savings come from skipping the LLM call. Since LLM is 10× more expensive than retrieval, this is still a large win — and it is semantically correct, which AF-based caching was not.

---

**Q: Why is the cache bypassed on follow-up turns in a multi-turn conversation?**

"What about sick leave?" in isolation is a question about sick leave. In the context of a conversation that started with "What's the PTO policy?", it means "tell me about sick leave in the same way you just told me about PTO". The same question string has different semantics depending on prior context.

Caching follow-up turns risks returning a cached answer that was correct in a different conversation context. The cache is safe only for first-turn questions that are self-contained.

---

## Conversation

**Q: How does the system resolve "what about sick leave?" as a follow-up?**

The `rewrite_query` LangGraph node receives the current question and the last 6 conversation turns. The system prompt instructs it: "Rewrite the question into a self-contained search query by resolving pronouns and references using the conversation history."

Input: question="what about sick leave?" + history=["User: What's the PTO policy?", "Assistant: 20 days..."]
Output: "How many sick leave days do employees receive per year?"

The rewritten query is what hits the vector store — self-contained, unambiguous, retrieves the right chunk.

---

**Q: Why cap the context window at 6 turns rather than sending the full history?**

Full history injected into every prompt causes:
1. Token cost grows linearly with conversation length
2. LLM attention dilutes — distant turns are weighted poorly anyway
3. Older turns are often irrelevant to the current question

6 turns (3 user + 3 assistant) is the sweet spot for enterprise Q&A — captures enough context for pronoun resolution and follow-ups without bloating the prompt. Full history is kept in the DB for audit and could be summarised and re-injected for very long conversations (summarisation pattern), but that adds complexity not warranted for a knowledge assistant.

---

## Ingestion

**Q: Why two-tier change detection (mtime then content hash)?**

mtime is an OS syscall — it reads the file's last-modified timestamp from the filesystem without opening the file. It is essentially free. If mtime hasn't changed, the content definitely hasn't changed — skip.

mtime changes when files are copied, rsynced, or touched without changing content. Computing SHA-256 of the full extracted text catches this: if mtime changed but hash matches, update the stored mtime and skip re-indexing. Only when the hash changes do we delete old chunks and re-index.

For a folder of 100 documents where 1 changed: 99 files × mtime check (~1ms) + 1 file × hash + re-index. Without the mtime tier, all 100 files must be read and hashed on every sync.

---

**Q: Why delete all chunks for a source before re-indexing rather than updating in place?**

Documents are split into variable numbers of chunks. If a document changes from 8 chunks to 6 chunks, an in-place update would leave 2 stale chunks in the index. Partial updates also create a window where some old and some new chunks coexist — queries during this window return inconsistent results.

Atomic delete-then-insert means the document is fully absent briefly, then fully present in its new form. No mix of old and new chunks is ever queryable. The brief absence is acceptable for an internal knowledge assistant.

---

**Q: Why use S3 object tags for department and access_level instead of inferring from the filename?**

Filename inference is fragile: `hr_policy_v2_FINAL.pdf` → what department? Policy files get renamed, copied, version-suffixed. The mapping breaks.

S3 object tags are set explicitly by the uploader at upload time and travel with the object through any rename or copy. They are the upload-time declaration of metadata, not a post-hoc inference. The Lambda reads the tags and passes them to the RAG API — `department` and `access_level` are always explicit, never guessed.

---

## Evaluation and Feedback

**Q: What does RAGAS actually measure and why do you need it alongside a golden dataset?**

A golden dataset tells you the final answer was right or wrong. RAGAS tells you WHERE in the pipeline it went wrong:

- `context_precision` low → retrieval fetched irrelevant chunks. Fix: chunking, reranker weights, hybrid BM25 weight.
- `context_recall` low → relevant chunks exist but weren't retrieved. Fix: embedding model, chunk size, top-K.
- `faithfulness` low → chunks were fine but LLM added facts not in context. Fix: system prompt, temperature, grounding check.
- `answer_relevancy` low → answer doesn't address the question. Fix: query rewriting, generation prompt.

Without this breakdown you know the system is failing but not which component to fix.

---

**Q: How does feedback auto-triage into retrieval vs generation failure?**

The `sources` array in the AskResponse carries a `score` for each retrieved chunk (the cross-encoder score). On a negative rating:

- If `max(source.score) < 0.25`: the best chunk the system found scored below the relevance threshold. Wrong chunks were retrieved. → `failure_mode: retrieval`
- If `max(source.score) >= 0.25`: the system retrieved relevant chunks (high score) but still produced a bad answer. The LLM ignored or contradicted the context. → `failure_mode: generation`

This routes the feedback to the right team automatically. Retrieval failures go to the team tuning BM25 weights and chunking. Generation failures go to the team tuning prompts and the grounding check.

---

## Design Decisions

**Q: Why LangGraph instead of a simple sequential function call?**

A sequential function always runs retrieve → generate. LangGraph enables conditional branching: if retrieved context scores below threshold, rewrite the query and retrieve again before generating. This retry loop is stateful — the state carries the original question, the rewritten question, and the attempt count to prevent infinite loops.

LangGraph also makes the workflow inspectable — every node transition is visible in `workflow_steps` in the response. This is important for debugging production failures.

---

**Q: Why keep the evaluation as a CLI script rather than an API endpoint?**

Evaluation takes 3–5 minutes, makes many LLM calls for scoring, and uses an admin-level AccessFilter that bypasses RBAC. Exposing it as an endpoint means:
- Any user hitting it accidentally burns significant LLM budget
- An admin-scoped filter is reachable from the network
- The server is effectively blocked for minutes

Evaluation is a quality gate run by engineers in CI before merging, not a runtime feature. It has no business being on the public API surface.

---

**Q: You said you skipped HyDE. Under what conditions would you add it?**

HyDE (Hypothetical Document Embedding) generates a fake answer to the question, embeds the fake answer (not the question), and uses that embedding for retrieval. It helps when the semantic gap between question style and document style is large.

This system already has three layers that compensate for that gap: query rewriting normalises vague questions, BM25 handles exact-match precision, and the cross-encoder reranker corrects retrieval order. Adding HyDE would add one LLM call per non-cached query (~300–500ms) for marginal recall improvement.

I would add HyDE if:
1. RAGAS evaluation showed `context_recall` consistently below 0.7 despite tuning BM25 weights and chunk size
2. We moved to a domain-specific embedding model where the question-document gap is better characterised
3. Latency requirements allowed the extra LLM call

"We evaluated HyDE but found the reranker compensates for retrieval imprecision on our corpus" is the production answer.

---

**Q: What would you change first if this system had to handle 10,000 concurrent users?**

In order:
1. **Rate limiting to Redis** — pgvector counters race under horizontal scaling. Redis sorted sets with atomic ZADD/ZCARD are the standard solution.
2. **Embedding cache** — the embedding call in the cache lookup is the bottleneck. Cache question embeddings in Redis (exact string → vector) to avoid repeated API calls for the same question.
3. **Read replicas for pgvector** — retrieval queries are read-heavy. Route them to a read replica, writes (ingest, cache store, feedback) to primary.
4. **Async ingestion worker** — move `run_ingest_job` from FastAPI BackgroundTasks (runs in the same process) to a Celery worker or AWS SQS consumer so ingestion doesn't compete with query serving.
5. **Connection pooling** — replace single connections with PgBouncer or asyncpg pool.

The current architecture handles hundreds of concurrent users comfortably. The bottleneck beyond that is the embedding API rate limit and the pgvector connection pool.
