#!/usr/bin/env python
"""
Example script demonstrating Langfuse tracing with the RAG application.
Run this to test the integration and see traces in Langfuse.

Usage:
    python test_langfuse_tracing.py
"""

import asyncio
import json
import sys

from app.retrieval.models import AskRequest, SearchRequest
from app.retrieval.qa import answer_question, search_knowledge_base
from app.tracing import is_tracing_enabled


async def test_search():
    """Test vector search tracing."""
    print("=" * 60)
    print("Testing Search Function")
    print("=" * 60)
    
    if not is_tracing_enabled():
        print("⚠️  Tracing is disabled. Set LANGFUSE credentials in .env")
        return
    
    print("✓ Tracing is enabled")
    
    # Test search
    try:
        question = "What is the company's HR policy?"
        print(f"\nSearching for: {question}")
        
        result = search_knowledge_base(question, top_k=3)
        
        print(f"\n✓ Found {len(result.results)} results:")
        for i, item in enumerate(result.results, 1):
            print(f"  {i}. {item.source.source} (score: {item.source.score:.3f})")
            
        print(f"\n✓ Search trace sent to Langfuse")
    except Exception as e:
        print(f"✗ Error during search: {e}")
        return False
    
    return True


async def test_ask():
    """Test RAG query tracing."""
    print("\n" + "=" * 60)
    print("Testing RAG Ask Function")
    print("=" * 60)
    
    if not is_tracing_enabled():
        print("⚠️  Tracing is disabled. Set LANGFUSE credentials in .env")
        return
    
    print("✓ Tracing is enabled")
    
    # Test ask
    try:
        question = "What security measures are in place?"
        print(f"\nAsking: {question}")
        
        result = answer_question(question, top_k=3)
        
        print(f"\n✓ Answer generated:")
        print(f"  Length: {len(result.answer)} characters")
        print(f"  Sources: {len(result.sources)}")
        print(f"  Grounded: {result.grounded}")
        print(f"  Workflow steps: {len(result.workflow_steps)}")
        
        if result.rewritten_question:
            print(f"  Query rewritten: {result.rewritten_question}")
        
        print(f"\n  Answer preview: {result.answer[:100]}...")
        
        print(f"\n✓ RAG query trace sent to Langfuse")
    except Exception as e:
        print(f"✗ Error during ask: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def main():
    """Run all tests."""
    print("\n" + "🚀 " * 15)
    print("Langfuse Tracing Test Suite")
    print("🚀 " * 15 + "\n")
    
    # Check configuration
    print("Configuration Check:")
    print("-" * 60)
    print(f"Tracing enabled: {is_tracing_enabled()}")
    
    if not is_tracing_enabled():
        print("\n⚠️  SETUP REQUIRED:")
        print("Add the following to your .env file:")
        print("  LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("  LANGFUSE_SECRET_KEY=sk-lf-...")
        print("  LANGFUSE_BASE_URL=http://localhost:3000")
        return
    
    # Run tests
    tests = [
        ("Search Test", test_search),
        ("RAG Ask Test", test_ask),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Open Langfuse dashboard: LANGFUSE_BASE_URL")
    print("2. Look for 'search' and 'rag_workflow' traces")
    print("3. Inspect the trace details to see:")
    print("   - Input questions and parameters")
    print("   - Output metrics (results count, answer length)")
    print("   - Execution flow and timing")
    print("   - Any errors or performance issues")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
