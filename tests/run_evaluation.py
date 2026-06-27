"""
Evaluate the RAG pipeline and compare search modes.

Usage:
  python tests/run_evaluation.py                        # compare semantic vs hybrid
  python tests/run_evaluation.py --mode hybrid          # single mode
  python tests/run_evaluation.py --mode semantic        # single mode
  python tests/run_evaluation.py --top-k 6             # change top_k

Output: score table + per-question breakdown + saved JSON report.

Scores are 0.0–1.0. Higher is better.

  context_precision  — retrieved chunks that were actually useful
  context_recall     — facts from ground truth that were covered by retrieved chunks
  faithfulness       — answer grounded in context (no hallucination)
  answer_relevancy   — answer addresses the question asked
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make sure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.runner import EvalResult, run_evaluation


def print_summary_table(results: list[EvalResult]) -> None:
    col = 18
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "mean_latency_ms"]
    labels  = ["ctx_precision",     "ctx_recall",     "faithfulness", "answer_relev",     "latency(ms)"]

    header = f"{'mode':<12}" + "".join(f"{l:>{col}}" for l in labels)
    print("\n" + "=" * len(header))
    print("RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        s = r.summary()
        row = f"{s['search_mode']:<12}"
        for m in metrics:
            val = s[m]
            row += f"{val:>{col}.4f}" if isinstance(val, float) else f"{val:>{col}}"
        print(row)

    print("=" * len(header))


def print_per_question(result: EvalResult) -> None:
    print(f"\nPer-question breakdown  [{result.search_mode}]")
    print("-" * 80)
    for i, row in enumerate(result.rows, 1):
        print(f"\n[{i}] {row.question}")
        print(f"    ctx_prec={row.context_precision:.3f}  ctx_rec={row.context_recall:.3f}"
              f"  faithful={row.faithfulness:.3f}  ans_rel={row.answer_relevancy:.3f}"
              f"  {row.latency_ms:.0f}ms")
        print(f"    answer: {row.answer[:120]}{'...' if len(row.answer)>120 else ''}")


def save_report(results: list[EvalResult]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("tests/eval_reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "results": [
            {
                "summary": r.summary(),
                "rows": [
                    {
                        "question": row.question,
                        "answer": row.answer,
                        "ground_truth": row.ground_truth,
                        "context_precision": row.context_precision,
                        "context_recall": row.context_recall,
                        "faithfulness": row.faithfulness,
                        "answer_relevancy": row.answer_relevancy,
                        "latency_ms": row.latency_ms,
                    }
                    for row in r.rows
                ],
            }
            for r in results
        ],
    }

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS metrics")
    parser.add_argument("--mode", choices=["semantic", "hybrid", "auto", "compare"],
                        default="compare",
                        help="search mode to evaluate, or 'compare' to run both (default)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--no-save", action="store_true", help="skip saving JSON report")
    args = parser.parse_args()

    if args.mode == "compare":
        modes = ["semantic", "hybrid"]
    else:
        modes = [args.mode]

    results = []
    for mode in modes:
        r = run_evaluation(search_mode=mode, top_k=args.top_k)
        results.append(r)
        print_per_question(r)

    print_summary_table(results)

    if not args.no_save:
        report_path = save_report(results)
        print(f"\nReport saved: {report_path}")

    if len(results) == 2:
        _print_delta(results[0], results[1])


def _print_delta(baseline: EvalResult, improved: EvalResult) -> None:
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    print(f"\nDelta  ({improved.search_mode} vs {baseline.search_mode})")
    print("-" * 50)
    for m in metrics:
        b = getattr(baseline, m) or 0.0
        i = getattr(improved, m) or 0.0
        delta = i - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:<22}  {b:.4f}  ->  {i:.4f}  ({sign}{delta:.4f})")


if __name__ == "__main__":
    main()
