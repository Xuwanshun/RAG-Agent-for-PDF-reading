"""
Run the RAG evaluation pipeline against a JSONL question file.

Each line in the question file must be JSON with at minimum:
    {"question": "..."}

Optional fields that unlock additional metrics:
    "ground_truth":        str   → context_recall, answer_correctness
    "relevant_chunk_ids":  list  → hit_rate, mrr, precision@k, recall@k

Output
------
  eval/results/<run_name>/
    results.jsonl   — per-question scores + answer + retrieved sources
    summary.json    — aggregated mean scores across all questions
    report.txt      — human-readable table

Usage:
    python -m eval.run_eval \
        --questions eval/test_questions.jsonl \
        --run-name attention-paper \
        --doc-filter <document_id>   # optional: restrict to one document
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config import Settings
from document_process.clients import build_openai_client
from eval.metrics import score_sample
from rag.dispatch import answer_question

logger = logging.getLogger(__name__)


def _load_questions(path: Path) -> list[dict]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run(
    questions_path: Path,
    *,
    run_name: str = "default",
    doc_filter: list[str] | None = None,
    output_dir: Path = Path("eval/results"),
    settings: Settings | None = None,
) -> dict:
    resolved = settings or Settings()
    client = build_openai_client(resolved)
    questions = _load_questions(questions_path)

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    all_scores: dict[str, list[float]] = {}
    per_question_results = []

    with results_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(questions, start=1):
            question = item["question"]
            ground_truth = item.get("ground_truth")
            relevant_chunk_ids = item.get("relevant_chunk_ids")

            logger.info("[%d/%d] %s", i, len(questions), question[:80])

            try:
                response = answer_question(
                    question,
                    settings=resolved,
                    doc_filter=doc_filter,
                )
            except Exception as exc:
                logger.warning("Pipeline failed for question %d: %s", i, exc)
                continue

            contexts = [src.get("text", "") for src in response.sources if src.get("text")]
            # sources don't include text directly — pull from answer evidence
            # Use the source metadata to reconstruct context text
            contexts = _extract_context_texts(response, resolved)
            retrieved_ids = [src["chunk_id"] for src in response.sources]

            scores = score_sample(
                client,
                question=question,
                answer=response.answer,
                contexts=contexts,
                retrieved_ids=retrieved_ids,
                ground_truth=ground_truth,
                relevant_chunk_ids=relevant_chunk_ids,
            )

            for k, v in scores.items():
                all_scores.setdefault(k, []).append(v)

            record = {
                "question": question,
                "answer": response.answer,
                "sources": response.sources,
                "scores": scores,
                "ground_truth": ground_truth,
            }
            per_question_results.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            # Gentle rate-limit buffer between LLM scoring calls
            time.sleep(0.5)

    # Detect whether precision/recall@k are meaningful.
    # With only 1 labeled chunk per question, precision@k = hit_rate/K (always)
    # and recall@k = hit_rate (always). Flag this so the report can warn.
    max_labels = max(
        (len(item.get("relevant_chunk_ids") or []) for item in questions),
        default=0,
    )
    single_label_dataset = max_labels <= 1

    summary = {metric: round(_mean(vals), 4) for metric, vals in all_scores.items()}
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = _build_report(run_name, summary, per_question_results, single_label_dataset=single_label_dataset)
    report_path = run_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    logger.info("Results saved to %s", run_dir)
    return summary


def _extract_context_texts(response, settings: Settings) -> list[str]:
    """Load chunk text from processed artifacts for the retrieved sources."""
    texts = []
    seen_docs: dict[str, list] = {}

    for src in response.sources:
        doc_id = src.get("document_id")
        if not doc_id:
            continue
        if doc_id not in seen_docs:
            chunks_path = settings.processed_documents_dir / doc_id / "chunks.json"
            if chunks_path.exists():
                seen_docs[doc_id] = json.loads(chunks_path.read_text(encoding="utf-8"))
            else:
                seen_docs[doc_id] = []

        chunk_id = src.get("chunk_id")
        for chunk in seen_docs[doc_id]:
            if chunk.get("chunk_id") == chunk_id or chunk.get("id") == chunk_id:
                texts.append(chunk.get("text", ""))
                break

    return texts


def _build_report(
    run_name: str,
    summary: dict,
    results: list[dict],
    *,
    single_label_dataset: bool = False,
) -> str:
    lines = [
        f"RAG Evaluation Report — {run_name}",
        "=" * 50,
        "",
        "Aggregate Scores",
        "-" * 30,
    ]
    # Metrics that are redundant when each question has exactly 1 labeled chunk:
    # precision@k = hit_rate/K  (mathematically fixed)
    # recall@k    = hit_rate    (same signal)
    redundant = {"precision_at_k", "recall_at_k"} if single_label_dataset else set()

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_relevance": "Context Relevance",
        "context_recall": "Context Recall",
        "answer_correctness": "Answer Correctness",
        "hit_rate": "Hit Rate @ K",
        "mrr": "MRR",
        "precision_at_k": "Precision @ K",
        "recall_at_k": "Recall @ K",
    }
    for key, label in metric_labels.items():
        if key not in summary:
            continue
        bar = "█" * int(summary[key] * 20)
        if key in redundant:
            lines.append(
                f"  {label:<22} {summary[key]:.3f}  {bar}  ⚠ redundant (1 label/question = hit_rate/{len(results[0]['sources']) if results else 'K'})"
            )
        else:
            lines.append(f"  {label:<22} {summary[key]:.3f}  {bar}")

    lines += ["", "Per-Question Results", "-" * 30]
    for i, r in enumerate(results, start=1):
        scores_str = "  ".join(f"{k}={v:.2f}" for k, v in r["scores"].items())
        lines.append(f"Q{i:02d}: {r['question'][:60]}")
        lines.append(f"     Scores: {scores_str}")
        lines.append(f"     Answer: {r['answer'][:120]}...")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--questions", required=True, help="Path to JSONL question file")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--doc-filter", nargs="*", help="Restrict to specific document IDs")
    parser.add_argument("--output-dir", default="eval/results")
    args = parser.parse_args()

    summary = run(
        Path(args.questions),
        run_name=args.run_name,
        doc_filter=args.doc_filter,
        output_dir=Path(args.output_dir),
    )
    if any(v < 0.5 for v in summary.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
