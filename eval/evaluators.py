"""
Retrieval evaluators scored against hand-labelled gold chunks.

RAGAS covers the generation side and also offers context_precision /
context_recall — but those are LLM-judged: a model reads the retrieved passages
and opines on their relevance. These are set comparisons against labels a human
verified, so they answer a different and stricter question.

The distinction is not academic. The $81.3B-vs-$51.5B bug was a case where the
correct chunk *was* retrieved and the answer was still wrong. An LLM-judged
context metric cannot separate "we never retrieved it" from "we retrieved it and
misread it"; these metrics can, which is what makes failure attribution possible.
"""

from __future__ import annotations

from typing import Any

from eval.retrieval_metrics import hit_rate, mrr, ndcg_at_k, precision_at_1, recall_at_k


def extract_retrieved_chunk_ids(outputs: dict[str, Any]) -> list[str]:
    """
    Pull chunk ids out of a pipeline response, preserving rank order.

    Order is load-bearing: MRR and nDCG both score by position, so a reordering
    here silently changes the result rather than raising.
    """
    sources = (outputs or {}).get("sources") or []
    return [str(source["chunk_id"]) for source in sources if isinstance(source, dict) and source.get("chunk_id")]


def extract_gold_chunk_ids(reference_outputs: dict[str, Any]) -> list[str]:
    """Read the labelled gold chunks from a dataset example."""
    return list((reference_outputs or {}).get("relevant_chunk_ids") or [])


def score_retrieval(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
    *,
    k: int,
) -> dict[str, float]:
    """
    Score one question's retrieval against its gold labels.

    Reports both rank-insensitive metrics (hit_rate, recall@k) and rank-sensitive
    ones (MRR, nDCG@k, precision@1). Keeping both is deliberate: when they
    disagree, the gold chunk was found but ranked badly, which points at the
    reranker rather than at retrieval.

    Questions with no labels score 0.0 rather than raising, so one unlabelled
    example cannot abort a whole run.
    """
    retrieved = extract_retrieved_chunk_ids(outputs)
    gold = set(extract_gold_chunk_ids(reference_outputs))

    return {
        "hit_rate": hit_rate(retrieved, gold),
        "mrr": mrr(retrieved, gold),
        f"ndcg@{k}": ndcg_at_k(retrieved, gold, k=k),
        "precision@1": precision_at_1(retrieved, gold),
        f"recall@{k}": recall_at_k(retrieved, gold, k=k),
    }
