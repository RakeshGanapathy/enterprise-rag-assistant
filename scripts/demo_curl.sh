#!/usr/bin/env bash
# =============================================================================
# RAG API End-to-End Demo — curl commands
# Run from the project root: bash scripts/demo_curl.sh
#
# Prerequisites:
#   1. pgvector running:  docker start pgvector-rag  (or see README)
#   2. Server running:    uvicorn app.main:app --port 8000
#   3. TOKEN set below   (generate once, valid 8 hours)
# =============================================================================

BASE="http://localhost:8000"

# ── Generate token (run once, paste result into TOKEN below) ──────────────────
# python -c "
# import datetime, jose.jwt as j, sys; sys.path.insert(0,'.')
# from app.config import get_settings; s = get_settings()
# tok = j.encode({'sub':'demo@rag','role':'admin','domain':'all',
#   'exp': datetime.datetime.now(datetime.UTC)+datetime.timedelta(hours=8)},
#   s.jwt_secret, algorithm='HS256')
# print(tok)
# "

TOKEN="${RAG_TOKEN:-PASTE_TOKEN_HERE}"
AUTH="Authorization: Bearer $TOKEN"

sep() { echo; echo "──────────────────────────────────────────────"; echo "  $1"; echo "──────────────────────────────────────────────"; }

# ── 1. Health ─────────────────────────────────────────────────────────────────
sep "1. HEALTH CHECK"
curl -s "$BASE/health" | python -m json.tool

# ── 2. Upload documents ───────────────────────────────────────────────────────
sep "2. UPLOAD DOCUMENTS"
for f in data/sample_docs/hr_policy.md data/sample_docs/security_policy.md \
          data/sample_docs/product_faq.md data/sample_docs/incident_response.md; do
  echo "Uploading: $f"
  JOB=$(curl -s -X POST "$BASE/documents/upload" \
    -H "$AUTH" \
    -F "file=@$f")
  echo "$JOB" | python -m json.tool
  JOB_ID=$(echo "$JOB" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

  # Poll until done
  for i in $(seq 1 10); do
    sleep 2
    STATUS=$(curl -s "$BASE/documents/status/$JOB_ID")
    ST=$(echo "$STATUS" | python -c "import sys,json; print(json.load(sys.stdin).get('status','?'))")
    echo "  status: $ST"
    [ "$ST" = "done" ] || [ "$ST" = "failed" ] && break
  done
done

# ── 3. List documents ─────────────────────────────────────────────────────────
sep "3. LIST INDEXED DOCUMENTS"
curl -s "$BASE/documents" -H "$AUTH" | python -m json.tool

# ── 4. Hybrid search ──────────────────────────────────────────────────────────
sep "4. HYBRID SEARCH"
curl -s -X POST "$BASE/search" \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the PTO policy?"}' | python -m json.tool

# ── 5. /ask (first call — cache miss) ────────────────────────────────────────
sep "5. ASK (first call — will call LLM)"
QUESTION='{"question": "What is the PTO policy and how many vacation days do employees get?", "top_k": 4}'
curl -s -X POST "$BASE/ask" \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d "$QUESTION" | python -m json.tool

# ── 6. /ask (second call — cache hit) ────────────────────────────────────────
sep "6. ASK AGAIN (should show cached: true)"
curl -s -X POST "$BASE/ask" \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d "$QUESTION" | python -m json.tool

# ── 7. Streaming /ask/stream ──────────────────────────────────────────────────
sep "7. STREAMING ASK (SSE — watch tokens appear)"
curl -s --no-buffer -X POST "$BASE/ask/stream" \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the incident response procedure step by step"}' \
  --max-time 30
echo

# ── 8. Submit feedback ────────────────────────────────────────────────────────
sep "8. SUBMIT FEEDBACK (thumbs up)"
curl -s -X POST "$BASE/feedback" \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the PTO policy?","answer":"20 days PTO.","rating":1,"comment":"Very helpful!"}' \
  | python -m json.tool

# ── 9. Cache stats ────────────────────────────────────────────────────────────
sep "9. CACHE STATS"
curl -s "$BASE/cache/stats" -H "$AUTH" | python -m json.tool

# ── 10. Debug chunks ──────────────────────────────────────────────────────────
sep "10. DEBUG CHUNKS (first 5)"
curl -s "$BASE/debug/chunks?limit=5" -H "$AUTH" | python -m json.tool

# ── 11. Feedback triage ───────────────────────────────────────────────────────
sep "11. FEEDBACK TRIAGE"
curl -s "$BASE/feedback/triage?limit=10" -H "$AUTH" | python -m json.tool

sep "DEMO COMPLETE — open http://localhost:8000/docs for Swagger UI"
