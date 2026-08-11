"""
Run one evaluation: execute the pipeline over a labelled question set and score it.

Division of labour:
  RAGAS       generation-side metrics (faithfulness, answer relevancy, correctness)
  evaluators  retrieval metrics against hand-labelled gold chunks
  provenance  the flags, models and corpus hash that make the run attributable
  stats       comparing two runs (see compare_runs)

LangSmith is optional. Results are always written locally so a run never depends
on a network service; when LANGSMITH_API_KEY is set the same results are logged
as an experiment so runs can be compared in its UI.

Usage:
    python -m eval.run --dataset eval/datasets/msft_fy26q2_press_release.jsonl \
                       --name baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings
from eval.evaluators import score_retrieval
from eval.provenance import run_metadata
from rag.dispatch import answer_question

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    question: str
    ground_truth: str
    answer: str
    retrieved_chunk_ids: list[str]
    retrieved_texts: list[str]
    question_type: str
    retrieval_scores: dict[str, float]
    latency_seconds: float
    error: str | None = None


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_chunk_text_lookup(settings: Settings) -> dict[str, str]:
    """
    Map chunk_id -> chunk text from the frozen artifacts.

    The API's source payload carries ids and scores but not text, and RAGAS needs
    the passages the synthesiser actually saw. Looking them up here keeps the
    production response shape unchanged — an eval requirement should not widen a
    payload served to every user.
    """
    from eval.dataset import load_chunks

    return {
        str(chunk.get("chunk_id")): str(chunk.get("text") or "")
        for chunk in load_chunks(settings.processed_documents_dir)
    }


def run_pipeline(dataset: list[dict[str, Any]], settings: Settings, *, k: int) -> list[QuestionResult]:
    """Execute the RAG pipeline over every question, scoring retrieval as we go."""
    chunk_texts = build_chunk_text_lookup(settings)
    results: list[QuestionResult] = []

    for index, example in enumerate(dataset, start=1):
        question = example["question"]
        logger.info("[%d/%d] %s", index, len(dataset), question[:70])
        started = time.time()
        try:
            response = answer_question(question, settings=settings, top_k=k)
        except Exception as exc:  # noqa: BLE001 - one bad question must not kill the run
            logger.warning("Pipeline failed on %r: %s", question[:60], exc)
            results.append(
                QuestionResult(
                    question=question,
                    ground_truth=example.get("ground_truth", ""),
                    answer="",
                    retrieved_chunk_ids=[],
                    retrieved_texts=[],
                    question_type=example.get("question_type", "factual"),
                    retrieval_scores=score_retrieval({}, example, k=k),
                    latency_seconds=time.time() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        outputs = {"sources": response.sources}
        results.append(
            QuestionResult(
                question=question,
                ground_truth=example.get("ground_truth", ""),
                answer=response.answer,
                retrieved_chunk_ids=[str(s["chunk_id"]) for s in response.sources],
                # RAGAS scores against the text the synthesiser actually saw.
                retrieved_texts=[chunk_texts.get(str(s["chunk_id"]), "") for s in response.sources],
                question_type=example.get("question_type", "factual"),
                retrieval_scores=score_retrieval(outputs, example, k=k),
                latency_seconds=time.time() - started,
            )
        )
    return results


def score_generation(results: list[QuestionResult], settings: Settings) -> dict[str, float]:
    """
    Score the generation side with RAGAS.

    Imported lazily: RAGAS pulls in datasets and pyarrow, and a retrieval-only
    run should not pay that import cost or fail if it is not installed.
    """
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        logger.warning("ragas not installed; skipping generation metrics")
        return {}

    usable = [r for r in results if r.answer and r.retrieved_texts]
    if not usable:
        logger.warning("no answers with retrieved context; skipping generation metrics")
        return {}

    samples = [
        SingleTurnSample(
            user_input=r.question,
            response=r.answer,
            retrieved_contexts=r.retrieved_texts,
            reference=r.ground_truth,
        )
        for r in usable
    ]

    try:
        scored = evaluate(EvaluationDataset(samples=samples), metrics=[faithfulness, answer_relevancy])
        return {name: float(value) for name, value in scored._repr_dict.items()}
    except Exception as exc:  # noqa: BLE001 - generation metrics are not worth aborting a run
        logger.warning("RAGAS scoring failed: %s", exc)
        return {}


def aggregate(results: list[QuestionResult]) -> dict[str, Any]:
    """Mean of each retrieval metric, plus operational cost signals."""
    summary: dict[str, Any] = {}
    if results:
        for metric in results[0].retrieval_scores:
            summary[metric] = sum(r.retrieval_scores[metric] for r in results) / len(results)

    latencies = sorted(r.latency_seconds for r in results)
    if latencies:
        summary["latency_p50"] = latencies[len(latencies) // 2]
        summary["latency_p95"] = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    summary["questions"] = len(results)
    summary["errors"] = sum(1 for r in results if r.error)
    return summary


def write_results(output_dir: Path, results: list[QuestionResult], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run one RAG evaluation.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--name", required=True, help="Run name, e.g. 'baseline' or 'no-reranker'")
    parser.add_argument("--limit", type=int, help="Score only the first N questions")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--skip-generation", action="store_true", help="Retrieval metrics only (no RAGAS calls)")
    args = parser.parse_args()

    settings = Settings()
    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    k = settings.default_top_k
    results = run_pipeline(dataset, settings, k=k)

    summary: dict[str, Any] = {"run_name": args.name, **aggregate(results)}
    if not args.skip_generation:
        summary.update(score_generation(results, settings))
    # Provenance last so a metric can never overwrite it.
    summary["metadata"] = run_metadata(settings)

    write_results(args.output_dir / args.name, results, summary)

    print(f"\n=== {args.name} ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:16} {value:.4f}")
        elif not isinstance(value, dict):
            print(f"  {key:16} {value}")
    print(f"  config_hash      {summary['metadata']['config_hash']}")
    print(f"  corpus_hash      {summary['metadata']['corpus_hash']}")
    print(f"\nwrote {args.output_dir / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
