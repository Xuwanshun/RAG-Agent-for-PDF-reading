"""
Compare two eval runs and say whether the difference is real.

Both runs answer the same questions, so the comparison is paired: what matters
is the per-question difference, not the gap between two aggregate means. A bare
mean cannot distinguish a genuine improvement from two questions flipping.

Usage:
    python -m eval.compare --baseline eval/results/baseline-linear \
                           --variant  eval/results/langgraph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.stats import PairedComparison, paired_bootstrap


def load_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (per-question results, summary) for a completed run."""
    results = [
        json.loads(line)
        for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return results, summary


def align_runs(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    *,
    metric: str | None = None,
) -> list[tuple[str, tuple[float, float]]]:
    """
    Pair the two runs by question text.

    Pairing by question rather than by list position matters because a run that
    errored on a question, or that was sampled differently, would otherwise
    silently pair unrelated results and produce a confident wrong answer.
    """
    variant_by_question = {row["question"]: row for row in variant}
    pairs: list[tuple[str, tuple[float, float]]] = []
    for row in baseline:
        question = row["question"]
        other = variant_by_question.get(question)
        if other is None:
            continue
        if metric is None:
            # Alignment-only use: expose whatever single metric each row carries.
            base_scores = row.get("retrieval_scores", {})
            var_scores = other.get("retrieval_scores", {})
            key = next(iter(base_scores), None)
            if key is None:
                continue
            pairs.append((question, (float(base_scores[key]), float(var_scores.get(key, 0.0)))))
        else:
            pairs.append((question, (float(row["retrieval_scores"][metric]), float(other["retrieval_scores"][metric]))))
    return pairs


def compare_metric(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    *,
    metric: str,
    iterations: int = 10_000,
    seed: int = 0,
) -> PairedComparison:
    """Paired bootstrap for one metric across the questions both runs answered."""
    pairs = align_runs(baseline, variant, metric=metric)
    if not pairs:
        raise ValueError("runs share no questions; nothing to compare")
    base_scores = [pair[0] for _question, pair in pairs]
    variant_scores = [pair[1] for _question, pair in pairs]
    return paired_bootstrap(base_scores, variant_scores, iterations=iterations, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two eval runs.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--variant", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    baseline_results, baseline_summary = load_run(args.baseline)
    variant_results, variant_summary = load_run(args.variant)

    base_meta = baseline_summary.get("metadata", {})
    var_meta = variant_summary.get("metadata", {})

    print(f"baseline : {baseline_summary.get('run_name')}  config={base_meta.get('config_hash')}")
    print(f"variant  : {variant_summary.get('run_name')}  config={var_meta.get('config_hash')}")

    # A differing corpus invalidates the comparison regardless of how similar the
    # configs look, so say so loudly rather than printing a plausible table.
    if base_meta.get("corpus_hash") != var_meta.get("corpus_hash"):
        print("\n  WARNING: corpus_hash differs between runs — the comparison is not valid.")
        print(f"    baseline corpus {base_meta.get('corpus_hash')}  variant corpus {var_meta.get('corpus_hash')}")
    if base_meta.get("config_hash") == var_meta.get("config_hash"):
        print("\n  NOTE: identical config_hash — these runs used the same settings.")

    changed = sorted(
        key[5:]
        for key in set(base_meta) | set(var_meta)
        if key.startswith("flag.") and base_meta.get(key) != var_meta.get(key)
    )
    print(f"\nflags that differ: {', '.join(changed) if changed else '(none)'}")

    metrics = sorted(baseline_results[0]["retrieval_scores"]) if baseline_results else []
    print(f"\n{'metric':14} {'baseline':>9} {'variant':>9} {'diff':>9}   95% CI            verdict")
    print("-" * 78)
    for metric in metrics:
        result = compare_metric(baseline_results, variant_results, metric=metric, seed=args.seed)
        verdict = "SIGNIFICANT" if result.significant else "not significant"
        print(
            f"{metric:14} {result.mean_baseline:9.4f} {result.mean_variant:9.4f} "
            f"{result.mean_diff:+9.4f}   [{result.ci_low:+.4f}, {result.ci_high:+.4f}]  {verdict}"
        )
    print(f"\nn = {len(align_runs(baseline_results, variant_results, metric=metrics[0]))} paired questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
