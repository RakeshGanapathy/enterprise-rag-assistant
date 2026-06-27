# Advanced Langfuse Integration Guide

This document provides advanced usage patterns and best practices for leveraging Langfuse features in the RAG application.

## Table of Contents

1. [Prompt Management](#prompt-management)
2. [Evaluations and Scoring](#evaluations-and-scoring)
3. [Datasets](#datasets)
4. [Custom Metrics](#custom-metrics)
5. [Advanced Tracing Patterns](#advanced-tracing-patterns)
6. [Performance Optimization](#performance-optimization)

## Prompt Management

### Using Langfuse Managed Prompts

Instead of hardcoding prompts, manage them in Langfuse for versioning and A/B testing:

```python
from langfuse import Langfuse
from app.config import get_settings

client = Langfuse(
    public_key=get_settings().langfuse_public_key,
    secret_key=get_settings().langfuse_secret_key,
    baseurl=get_settings().langfuse_base_url,
)

# Fetch prompt version from Langfuse
def get_qa_prompt():
    prompt = client.get_prompt("rag-qa-template", version=1)
    return prompt.prompt  # Returns the managed prompt text
```

### Using LangChain with Prompts

```python
from langfuse.integrations.langchain import LangfuseCallbackHandler
from langchain.prompts import PromptTemplate

# Create prompt template
template = """
Use the following context to answer the question.

Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Your LLM calls will be automatically traced
callbacks = [LangfuseCallbackHandler(...)]
```

## Evaluations and Scoring

### Define Custom Evaluations

```python
from app.tracing import get_langfuse_client

def evaluate_answer_quality(question: str, answer: str, sources: list) -> dict:
    """Evaluate RAG answer quality."""
    client = get_langfuse_client()
    
    span = client.span(name="evaluate_answer")
    
    # Your evaluation logic
    is_relevant = len(answer) > 50  # Simple heuristic
    has_sources = len(sources) > 0
    
    score = 1.0 if (is_relevant and has_sources) else 0.5
    
    # Log the score to Langfuse
    span.log(
        score=score,
        metadata={
            "relevant": is_relevant,
            "has_sources": has_sources,
        }
    )
    span.end()
    
    return {"score": score, "passed": score > 0.75}
```

### Integrate into Request Handler

```python
from app.main import app
from app.tracing import trace_span

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    with trace_span("ask_with_eval", input_data={"question": request.question}) as span:
        result = answer_question(request.question, request.top_k)
        
        # Evaluate the answer
        eval_result = evaluate_answer_quality(
            question=request.question,
            answer=result.answer,
            sources=result.sources
        )
        
        span["output"] = {
            "answer_length": len(result.answer),
            "evaluation_score": eval_result["score"],
        }
        
        return result
```

## Datasets

### Create Test Datasets

```python
def setup_test_dataset():
    """Create a dataset in Langfuse for testing."""
    from langfuse import Langfuse
    
    client = Langfuse(...)
    
    dataset = client.create_dataset(
        name="rag-test-questions",
        description="Questions for testing RAG system"
    )
    
    # Add examples
    examples = [
        {
            "question": "What is the HR policy?",
            "expected_answer": "The HR policy covers...",
        },
        {
            "question": "What security measures exist?",
            "expected_answer": "Security includes...",
        },
    ]
    
    for example in examples:
        dataset.add_example(
            input={"question": example["question"]},
            output={"answer": example["expected_answer"]},
        )
```

### Run Evaluations on Dataset

```python
def evaluate_on_dataset(dataset_name: str):
    """Run the RAG system on a dataset and evaluate."""
    from langfuse import Langfuse
    from app.retrieval.qa import answer_question
    
    client = Langfuse(...)
    dataset = client.get_dataset(dataset_name)
    
    results = []
    for example in dataset.examples:
        question = example.input["question"]
        
        # Run RAG
        result = answer_question(question)
        
        # Store result for analysis
        results.append({
            "question": question,
            "answer": result.answer,
            "grounded": result.grounded,
        })
    
    return results
```

## Custom Metrics

### Track Custom Metrics

```python
from app.tracing import trace_span

def process_with_metrics(question: str, top_k: int):
    """Track custom metrics during processing."""
    with trace_span("process_with_metrics") as span:
        start_time = time.time()
        
        # Your processing logic
        result = answer_question(question, top_k)
        
        elapsed = time.time() - start_time
        
        # Log custom metrics
        span["output"] = {
            "execution_time_ms": elapsed * 1000,
            "question_length": len(question),
            "answer_quality": 0.85,  # Your metric
            "retrieval_accuracy": 0.92,  # Your metric
        }
```

### Batch Metrics

```python
def batch_process(questions: list[str]):
    """Process multiple questions and aggregate metrics."""
    from app.tracing import get_langfuse_client
    
    client = get_langfuse_client()
    trace = client.trace(name="batch_process")
    
    metrics = {
        "total_questions": len(questions),
        "successful": 0,
        "failed": 0,
        "total_time_ms": 0,
    }
    
    for q in questions:
        try:
            start = time.time()
            result = answer_question(q)
            metrics["successful"] += 1
            metrics["total_time_ms"] += (time.time() - start) * 1000
        except Exception:
            metrics["failed"] += 1
    
    trace.end(output=metrics)
    return metrics
```

## Advanced Tracing Patterns

### Nested Traces for Complex Operations

```python
from app.tracing import trace_span

def complex_rag_operation(question: str):
    """Demonstrate nested tracing."""
    with trace_span("complex_operation", {"question": question}) as outer_span:
        
        # Retrieval phase
        with trace_span("retrieval_phase", {"question": question}) as ret_span:
            results = search_chunks(question)
            ret_span["output"] = {"chunks": len(results)}
        
        # Processing phase
        with trace_span("processing_phase") as proc_span:
            processed = process_chunks(results)
            proc_span["output"] = {"processed_chunks": len(processed)}
        
        # Generation phase
        with trace_span("generation_phase") as gen_span:
            answer = generate_answer(processed)
            gen_span["output"] = {"answer_length": len(answer)}
        
        outer_span["output"] = {
            "final_answer": answer,
            "pipeline_complete": True,
        }
        return answer
```

### Context-Aware Tracing

```python
from contextlib import contextmanager
from app.tracing import get_langfuse_client

@contextmanager
def trace_user_operation(user_id: str, operation_type: str):
    """Trace operations with user context."""
    client = get_langfuse_client()
    span = client.span(
        name=operation_type,
        metadata={
            "user_id": user_id,
            "operation": operation_type,
            "timestamp": datetime.now().isoformat(),
        }
    )
    try:
        yield span
    finally:
        span.end()

# Usage
with trace_user_operation("user123", "rag_query") as span:
    result = answer_question("What is...?")
    span.end(output={"answer": result.answer})
```

## Performance Optimization

### Sampling for High-Volume Operations

```python
import random
from app.tracing import trace_span, is_tracing_enabled

def search_with_sampling(question: str, sample_rate: float = 0.1):
    """Sample traces for high-frequency operations."""
    should_trace = random.random() < sample_rate
    
    if should_trace and is_tracing_enabled():
        with trace_span("search_sampled", {"question": question}) as span:
            result = search_chunks(question)
            span["output"] = {"count": len(result)}
    else:
        result = search_chunks(question)
    
    return result
```

### Batch Tracing

```python
def batch_search(questions: list[str]):
    """Trace batch operations efficiently."""
    from app.tracing import trace_span
    
    with trace_span("batch_search", {"count": len(questions)}) as span:
        results = []
        
        for question in questions:
            result = search_chunks(question)
            results.append(result)
        
        span["output"] = {
            "total_results": sum(len(r) for r in results),
            "questions_processed": len(questions),
        }
    
    return results
```

### Memory-Efficient Large Traces

```python
def process_large_dataset():
    """Process large data while keeping traces lean."""
    from app.tracing import get_langfuse_client
    
    client = get_langfuse_client()
    
    for batch in get_batches():
        span = client.span(name="process_batch")
        
        # Process but don't store full data in trace
        results = [process(item) for item in batch]
        
        # Store only aggregated metrics
        span.end(output={
            "processed_count": len(results),
            "success_rate": sum(1 for r in results if r.success) / len(results),
            # Don't store raw results - use references instead
        })
```

## Best Practices Summary

1. **Use Managed Prompts**: Version control your prompts in Langfuse
2. **Implement Evaluations**: Measure answer quality with custom metrics
3. **Create Datasets**: Build test sets for regression testing
4. **Track Custom Metrics**: Beyond built-in metrics, track domain-specific ones
5. **Nest Traces Meaningfully**: Create hierarchies that match your workflow
6. **Sample High-Volume Operations**: Avoid storage overload with sampling
7. **Leverage Metadata**: Use metadata for filtering and analysis
8. **Monitor in Production**: Set up dashboards and alerts

## Resources

- [Langfuse Python Docs](https://docs.langfuse.com/)
- [Langfuse API Reference](https://api.langfuse.com/)
- [Integration Examples](https://github.com/langfuse/langfuse-python)
