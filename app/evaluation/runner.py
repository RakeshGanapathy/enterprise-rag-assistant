"""
RAGAS evaluation runner.

Runs each question in the eval dataset through the RAG pipeline,
collects (question, answer, contexts, ground_truth), then scores
with four RAGAS metrics.

Metrics:
  context_precision  — of chunks retrieved, what fraction were relevant to the question?
  context_recall     — did retrieved chunks cover all facts in the ground truth?
  faithfulness       — does the answer only state things supported by the context?
  answer_relevancy   — does the answer actually address what was asked?

All four are 0.0–1.0. Higher is better.

Usage:
  from app.evaluation.runner import run_evaluation
  results = run_evaluation(search_mode="hybrid", top_k=4)
  # returns EvalResult with per-question rows + aggregate scores
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DATASET_PATH = Path(__file__).parent.parent.parent / "tests" / "eval_dataset.json"


@dataclass
class QuestionResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    latency_ms: float
    # RAGAS scores filled in after batch scoring
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None


@dataclass
class EvalResult:
    search_mode: str
    top_k: int
    rows: list[QuestionResult] = field(default_factory=list)
    # Aggregate scores (mean across all questions)
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    mean_latency_ms: float | None = None

    def summary(self) -> dict:
        return {
            "search_mode": self.search_mode,
            "top_k": self.top_k,
            "n_questions": len(self.rows),
            "context_precision": round(self.context_precision or 0, 4),
            "context_recall": round(self.context_recall or 0, 4),
            "faithfulness": round(self.faithfulness or 0, 4),
            "answer_relevancy": round(self.answer_relevancy or 0, 4),
            "mean_latency_ms": round(self.mean_latency_ms or 0, 1),
        }


def run_evaluation(
    search_mode: Literal["semantic", "hybrid", "auto"] = "hybrid",
    top_k: int = 4,
    dataset_path: Path | None = None,
) -> EvalResult:
    """
    Run every question in the eval dataset through the RAG pipeline,
    then score with RAGAS. Returns an EvalResult.

    Uses an admin-level AccessFilter so all questions can reach all documents
    regardless of domain/access_level in the dataset. This gives a clean
    measurement of retrieval quality without RBAC interference.
    """
    from app.retrieval.models import AccessFilter
    from app.retrieval.qa import answer_question

    # Admin filter — eval needs to reach all docs
    admin_filter = AccessFilter(
        departments=["hr", "security", "support", "product", "finance", "all", "general"],
        max_access_level=3,
    )

    path = dataset_path or DATASET_PATH
    dataset = json.loads(path.read_text(encoding="utf-8"))

    result = EvalResult(search_mode=search_mode, top_k=top_k)

    print(f"\nRunning {len(dataset)} questions  [search_mode={search_mode}, top_k={top_k}]")
    print("-" * 60)

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        t0 = time.perf_counter()
        ask_result = answer_question(
            question,
            top_k=top_k,
            search_mode=search_mode,
            access_filter=admin_filter,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        contexts = [s.source for s in ask_result.sources]  # source filenames
        # Prefer full text if available — use the answer's source metadata
        # The actual chunk text isn't in AskResponse; we re-retrieve it below
        contexts_text = _fetch_context_texts(question, top_k, search_mode, admin_filter)

        row = QuestionResult(
            question=question,
            answer=ask_result.answer,
            contexts=contexts_text,
            ground_truth=ground_truth,
            latency_ms=latency_ms,
        )
        result.rows.append(row)
        print(f"  [{i:2d}/{len(dataset)}] {question[:60]}{'...' if len(question)>60 else ''}")
        print(f"         latency: {latency_ms:.0f}ms  |  answer: {ask_result.answer[:80]}...")

    print("\nScoring with RAGAS...")
    _score_with_ragas(result)

    result.mean_latency_ms = sum(r.latency_ms for r in result.rows) / len(result.rows)
    return result


def _fetch_context_texts(
    question: str, top_k: int, search_mode: str, access_filter
) -> list[str]:
    """Re-run retrieval to get chunk text (AskResponse only returns source names)."""
    from app.retrieval.qa import _retrieve, resolve_search_mode

    resolved_mode, _ = resolve_search_mode(question, search_mode)
    matches = _retrieve(question, top_k, resolved_mode, access_filter)
    return [doc.page_content for doc, _score in matches]


def _score_with_ragas(result: EvalResult) -> None:
    """Call RAGAS and write per-question + aggregate scores back into result."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from app.config import get_settings

    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )

    hf_dataset = Dataset.from_dict({
        "question":     [r.question for r in result.rows],
        "answer":       [r.answer for r in result.rows],
        "contexts":     [r.contexts for r in result.rows],
        "ground_truth": [r.ground_truth for r in result.rows],
    })

    scores = evaluate(
        hf_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    scores_df = scores.to_pandas()

    for i, row in enumerate(result.rows):
        row.context_precision = float(scores_df.iloc[i]["context_precision"])
        row.context_recall    = float(scores_df.iloc[i]["context_recall"])
        row.faithfulness      = float(scores_df.iloc[i]["faithfulness"])
        row.answer_relevancy  = float(scores_df.iloc[i]["answer_relevancy"])

    result.context_precision = float(scores_df["context_precision"].mean())
    result.context_recall    = float(scores_df["context_recall"].mean())
    result.faithfulness      = float(scores_df["faithfulness"].mean())
    result.answer_relevancy  = float(scores_df["answer_relevancy"].mean())
