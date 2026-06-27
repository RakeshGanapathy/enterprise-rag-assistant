# Langfuse Quick Start Guide

Get Langfuse tracing working in 5 minutes.

## Step 1: Get API Keys

### Option A: Use Langfuse Cloud (Recommended)

1. Sign up at https://cloud.langfuse.com
2. Create a new project
3. Go to Settings > API Keys
4. Copy your keys

### Option B: Run Locally with Docker

```bash
# Start Langfuse locally
docker-compose up -d

# Access at http://localhost:3000
# Default credentials will be shown
```

## Step 2: Configure Environment

Update `.env` with your credentials:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-1234567890abcdef
LANGFUSE_SECRET_KEY=sk-lf-abcdefghijklmnop
LANGFUSE_BASE_URL=http://localhost:3000  # Local or https://cloud.langfuse.com
```

## Step 3: Verify Installation

```bash
# Test that everything is configured
python -c "
from app.tracing import is_tracing_enabled, get_langfuse_client
if is_tracing_enabled():
    client = get_langfuse_client()
    print('✓ Langfuse is configured and ready!')
    print(f'✓ Connected to: {client.baseurl}')
else:
    print('✗ Langfuse credentials not found in .env')
"
```

## Step 4: Run Your App

```bash
# Start the FastAPI application
uvicorn app.main:app --reload
```

## Step 5: Make Requests and See Traces

### Search Example
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the HR policy?", "top_k": 3}'
```

### Ask Example
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What security measures are in place?", "top_k": 3}'
```

### View Traces
Open http://localhost:3000 (or https://cloud.langfuse.com) to see:
- Request traces
- Execution flow
- Performance metrics
- Any errors

## Troubleshooting

### Traces Not Appearing

1. **Check credentials:**
   ```bash
   python -c "from app.config import get_settings; s = get_settings(); print(f'Has keys: {bool(s.langfuse_public_key)}')"
   ```

2. **Test connection:**
   ```bash
   python test_langfuse_tracing.py
   ```

3. **Check Langfuse server:**
   - Local: http://localhost:3000 should be accessible
   - Cloud: Check your project is active

### "Extra inputs are not permitted" Error

This usually means invalid credentials. Check:
- No extra spaces in `.env` values
- Correct public and secret keys
- Base URL matches your Langfuse instance

## What Gets Traced

✓ All API endpoints (HTTP method, path, status)
✓ Search operations (question, results count)
✓ RAG workflow (retrieval, grading, generation)
✓ Vector embeddings (chunks indexed, similarity scores)
✓ Answer generation (answer length, sources, grounding)

## Next Steps

1. **View Traces**: Check Langfuse dashboard to see your traces
2. **Set Up Prompts**: Create managed prompts in Langfuse
3. **Add Evaluation**: See `LANGFUSE_ADVANCED.md` for evaluation setup
4. **Monitor Performance**: Create dashboards for key metrics
5. **Configure Alerts**: Get notified of issues

## Common Patterns

### Trace a Custom Function
```python
from app.tracing import trace_span

with trace_span("my_operation", {"input": "value"}) as span:
    result = do_something()
    span["output"] = result
```

### Disable Tracing Temporarily
```bash
# Remove credentials from .env or set empty values
LANGFUSE_PUBLIC_KEY=""
```

### High-Volume Operations
See `LANGFUSE_ADVANCED.md` for sampling patterns to avoid storage overload.

## Support

- **Langfuse Docs**: https://langfuse.com/docs
- **GitHub Issues**: https://github.com/langfuse/langfuse
- **Discord Community**: https://langfuse.com/discord

## Costs

- **Cloud**: Pay-as-you-go, check pricing at cloud.langfuse.com
- **Self-hosted**: Free, runs locally or on your infrastructure
