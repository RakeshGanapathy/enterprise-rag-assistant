# Langfuse Tracing Integration for RAG Application

This document describes the Langfuse tracing integration added to the RAG (Retrieval-Augmented Generation) application for comprehensive observability and monitoring of LLM operations.

## Overview

Langfuse tracing has been integrated into the RAG application to provide:

- **Request/Response Tracing**: All FastAPI endpoints are automatically traced
- **Vector Search Tracing**: Embedding and similarity search operations are tracked
- **Workflow Tracing**: The LangGraph RAG workflow execution is monitored
- **Performance Metrics**: Latency, status codes, and operation results are captured
- **Error Tracking**: Failures are logged with full context

## Configuration

### Environment Variables

Add these variables to your `.env` file:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_BASE_URL=http://localhost:3000  # or https://cloud.langfuse.com
```

Get your API keys from your Langfuse project under Settings > API Keys.

### Testing Locally

To test Langfuse locally, you can use the self-hosted version:

```bash
# Run Langfuse locally with Docker
docker-compose -f docker-compose.yml up
# Then access at http://localhost:3000
```

## Tracing Architecture

### 1. **FastAPI Middleware** (`app/middleware.py`)

The `LangfuseTracingMiddleware` automatically traces all HTTP requests:
- Captures request method, path, and query parameters
- Records response status code
- Tracks execution time
- Captures errors with context

**When it's active**: All endpoints are traced if `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set.

### 2. **Endpoint-Level Tracing** (`app/main.py`)

Each endpoint is wrapped with `trace_span()` context managers:
- `/documents/ingest-samples`: Tracks document ingestion
- `/documents/upload`: Tracks file upload and processing
- `/search`: Traces vector search operations
- `/ask`: Traces RAG query execution with full details
- `/debug/chunks`: Traces debug operations

### 3. **Workflow Tracing** (`app/graph/workflow.py`)

The RAG workflow (`run_rag_workflow`) is traced with:
- Input question and parameters
- Number of sources used
- Whether the answer is grounded
- Workflow step count

### 4. **Vector Store Tracing** (`app/retrieval/vector_store.py`)

- `index_chunks()`: Tracks embedding generation and storage
- `search_chunks()`: Tracks similarity search with top score

### 5. **QA Functions Tracing** (`app/retrieval/qa.py`)

- `search_knowledge_base()`: Traces search operations
- `answer_question()`: Traces RAG answer generation

## Usage

### Basic Usage

Once configured, tracing works automatically:

```python
from app.tracing import trace_span

# Trace a section of code
with trace_span("my_operation", {"input": "value"}) as span:
    result = do_something()
    span["output"] = result
```

### Trace Functions

Use the decorator for simple function tracing:

```python
from app.tracing import trace_function

@trace_function("operation_name")
def my_function(arg1, arg2):
    return result
```

### LangChain Integration

The tracing module provides LangChain callbacks for automatic LLM call tracing:

```python
from app.tracing import get_langchain_callbacks

callbacks = get_langchain_callbacks()
# Use with LangChain models
llm = OpenAI(callbacks=callbacks)
```

## Best Practices

### 1. **Structured Data**

Always provide meaningful input and output data:

```python
with trace_span("search", {"question": question, "top_k": top_k}) as span:
    results = search(question, top_k)
    span["output"] = {"count": len(results), "top_score": results[0].score}
```

### 2. **Metadata**

Add context with metadata for filtering and analysis:

```python
with trace_span(
    "operation",
    metadata={"operation": "type", "user_id": user_id}
) as span:
    ...
```

### 3. **Error Handling**

Errors are automatically captured, but provide context:

```python
with trace_span("operation") as span:
    try:
        result = risky_operation()
        span["output"] = result
    except Exception as e:
        # Error is auto-captured by the context manager
        raise
```

### 4. **Disabling Tracing**

Tracing is disabled if credentials are missing. Check the status:

```python
from app.tracing import is_tracing_enabled

if is_tracing_enabled():
    print("Tracing is active")
```

## Tracing Flow Example

Here's how a typical `/ask` request is traced:

```
HTTP Request (FastAPI Middleware)
  ├─ /ask endpoint (trace_span)
  │   ├─ answer_question (trace_span)
  │   │   ├─ run_rag_workflow (trace_span)
  │   │   │   └─ LangGraph nodes execution
  │   │   └─ (returns AskResponse)
  │   └─ (returns answer with sources)
  └─ HTTP Response
```

## Monitoring in Langfuse

After setting up Langfuse, you can:

1. **View Traces**: See complete request flows with timing
2. **Analyze Performance**: Identify slow operations
3. **Monitor Errors**: Track failures with full context
4. **Debug Issues**: Inspect input/output at each step
5. **Track Metrics**: Monitor success rates and latencies

## Troubleshooting

### Tracing not appearing

1. Check credentials are set in `.env`
2. Verify `is_tracing_enabled()` returns True
3. Check network connectivity to Langfuse server
4. Verify API keys in Langfuse dashboard

### Performance impact

- Tracing adds minimal overhead (~1-2ms per request)
- Disable for high-throughput scenarios if needed
- Use sampling for high-frequency operations

### Pydantic validation errors

If you see "Extra inputs are not permitted" errors:
- Ensure `LANGFUSE_SECRET_KEY` is correctly set
- Check there are no trailing spaces in credentials
- Use the exact format from the Langfuse dashboard

## Next Steps

1. **Set up prompts**: Create managed prompts in Langfuse
2. **Create evaluations**: Set up evaluation criteria for answers
3. **Configure scoring**: Add custom metrics and scores
4. **Set up alerts**: Get notified of issues
5. **Analyze performance**: Review trends and patterns

## References

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Python SDK](https://github.com/langfuse/langfuse-python)
- [Langfuse Skills](https://github.com/langfuse/skills)
