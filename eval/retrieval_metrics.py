"""
Retrieval metrics scored against hand-labelled relevant chunk ids.

These are the metrics that say whether the right passage was found at all, which
is upstream of every generation metric: an answer cannot be faithful to evidence
that never reached the synthesiser.

On precision@k, which this module deliberately does not provide:
the question set labels one relevant chunk per question, so with k=4 the best
achievable precision@k is 0.25. The historical runs reported ~0.21 and read it as
poor precision, when it was really recall/k — a rescaling of a metric already
reported, carrying no independent information. precision@1 and nDCG@k answer the
question precision@k was meant to answer: is the right chunk at the top?
"""

from __future__ import annotations

from math import log2


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1.0 if any labelled chunk appears anywhere in the retrieved list."""
    return 1.0 if any(chunk_id in relevant_ids for chunk_id in retrieved_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first labelled chunk."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def precision_at_1(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Whether the single top-ranked chunk is labelled relevant.

    Unlike precision@k this is not capped by the number of labels, so it stays
    interpretable with one gold chunk per question.
    """
    if not retrieved_ids:
        return 0.0
    return 1.0 if retrieved_ids[0] in relevant_ids else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    """Fraction of labelled chunks that appear within the top k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for chunk_id in relevant_ids if chunk_id in top_k)
    return found / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], *, k: int) -> float:
    """
    Normalised discounted cumulative gain over binary relevance.

    Gain is discounted by log2(rank + 1), so a gold chunk at rank 1 scores 1.0,
    at rank 2 scores 0.631, at rank 4 scores 0.431. That gradient is the point:
    only the top few chunks reach the synthesiser, so rank position matters and
    a flat "was it in the list" metric cannot see the difference.

    Normalised against the best achievable ordering for this question, so scores
    stay comparable across questions with different numbers of labels.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    dcg = sum(
        1.0 / log2(rank + 1) for rank, chunk_id in enumerate(retrieved_ids[:k], start=1) if chunk_id in relevant_ids
    )
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
