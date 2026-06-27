import json
import time
from pathlib import Path

from app.ingestion.pipeline import ingest_directory
from app.retrieval.qa import answer_question, search_knowledge_base
from app.retrieval.vector_store import reset_vector_store


TEST_SET_PATH = Path("app/evaluation/test_questions.json")


def run_evaluation() -> dict:
    cases = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))

    reset_vector_store()
    ingest_result = ingest_directory("data/sample_docs")
    results = []
    start = time.perf_counter()

    for case in cases:
        question = case["question"]
        search_response = search_knowledge_base(question, top_k=4)
        answer_response = answer_question(question, top_k=4)

        retrieved_sources = [
            result.source.source for result in search_response.results
        ]
        answer_text = answer_response.answer.lower()
        expected_keywords = [
            keyword.lower() for keyword in case["expected_keywords"]
        ]

        source_hit = case["expected_source"] in retrieved_sources
        keyword_hit = all(keyword in answer_text for keyword in expected_keywords)
        citation_hit = any(
            source.source == case["expected_source"]
            for source in answer_response.sources
        )

        results.append(
            {
                "question": question,
                "expected_source": case["expected_source"],
                "retrieved_sources": retrieved_sources,
                "answer": answer_response.answer,
                "source_hit": source_hit,
                "keyword_hit": keyword_hit,
                "citation_hit": citation_hit,
                "grounded": answer_response.grounded,
            }
        )

    elapsed = time.perf_counter() - start
    total = len(results)

    return {
        "documents_loaded": ingest_result.documents_loaded,
        "chunks_indexed": ingest_result.chunks_indexed,
        "questions": total,
        "retrieval_hit_rate": _rate(results, "source_hit"),
        "answer_keyword_rate": _rate(results, "keyword_hit"),
        "citation_rate": _rate(results, "citation_hit"),
        "grounded_rate": _rate(results, "grounded"),
        "average_latency_seconds": round(elapsed / total, 2) if total else 0,
        "results": results,
    }


def _rate(results: list[dict], key: str) -> float:
    if not results:
        return 0.0
    passed = sum(1 for result in results if result[key])
    return round(passed / len(results), 2)


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
