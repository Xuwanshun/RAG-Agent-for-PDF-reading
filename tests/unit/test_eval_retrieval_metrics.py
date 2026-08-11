"""
Tests for retrieval metrics.

The existing harness reported precision@k alongside recall@k without noting that
with exactly one labelled chunk per question and k=4, precision@k is capped at
0.25 — so the reported 0.21 was just recall/k and carried no independent signal.
nDCG@k replaces it: it rewards putting the gold chunk near the top, which is what
actually matters when only the top few chunks reach the synthesiser.
"""

from __future__ import annotations

from eval.retrieval_metrics import (
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_1,
    recall_at_k,
)

GOLD = {"doc:chunk:7"}


def _ranking(*positions_of_gold: int, length: int = 4) -> list[str]:
    """Build a retrieved list of `length` ids with gold at the given 1-based ranks."""
    ids = [f"doc:chunk:{i}" for i in range(100, 100 + length)]
    for rank in positions_of_gold:
        ids[rank - 1] = "doc:chunk:7"
    return ids


# ── nDCG@k ────────────────────────────────────────────────────────────────────


def test_ndcg_is_one_when_gold_is_ranked_first():
    assert ndcg_at_k(_ranking(1), GOLD, k=4) == 1.0


def test_ndcg_rewards_a_higher_rank():
    """The core property precision@k could not express."""
    assert ndcg_at_k(_ranking(1), GOLD, k=4) > ndcg_at_k(_ranking(2), GOLD, k=4)
    assert ndcg_at_k(_ranking(2), GOLD, k=4) > ndcg_at_k(_ranking(4), GOLD, k=4)


def test_ndcg_uses_the_standard_log2_discount():
    # gold at rank 2 -> 1/log2(3) = 0.63093
    assert round(ndcg_at_k(_ranking(2), GOLD, k=4), 5) == 0.63093


def test_ndcg_is_zero_when_gold_is_absent():
    assert ndcg_at_k(_ranking(length=4), GOLD, k=4) == 0.0


def test_ndcg_ignores_results_beyond_k():
    """Chunks past the cutoff never reach the synthesiser, so they must not score."""
    assert ndcg_at_k(_ranking(5, length=6), GOLD, k=4) == 0.0


def test_ndcg_handles_empty_results():
    assert ndcg_at_k([], GOLD, k=4) == 0.0


def test_ndcg_handles_no_labels():
    assert ndcg_at_k(_ranking(1), set(), k=4) == 0.0


def test_ndcg_with_several_gold_chunks_is_normalised_to_one():
    gold = {"doc:chunk:7", "doc:chunk:8"}
    ranking = ["doc:chunk:7", "doc:chunk:8", "doc:chunk:100", "doc:chunk:101"]

    assert ndcg_at_k(ranking, gold, k=4) == 1.0


# ── precision@1 ───────────────────────────────────────────────────────────────


def test_precision_at_1_is_one_when_the_top_hit_is_gold():
    assert precision_at_1(_ranking(1), GOLD) == 1.0


def test_precision_at_1_is_zero_when_the_top_hit_is_wrong():
    assert precision_at_1(_ranking(2), GOLD) == 0.0


def test_precision_at_1_handles_empty_results():
    assert precision_at_1([], GOLD) == 0.0


# ── the metrics carried over ──────────────────────────────────────────────────


def test_hit_rate_is_binary_over_the_whole_list():
    assert hit_rate(_ranking(4), GOLD) == 1.0
    assert hit_rate(_ranking(length=4), GOLD) == 0.0


def test_mrr_is_the_reciprocal_of_the_first_gold_rank():
    assert mrr(_ranking(1), GOLD) == 1.0
    assert mrr(_ranking(4), GOLD) == 0.25
    assert mrr(_ranking(length=4), GOLD) == 0.0


def test_recall_at_k_counts_only_gold_within_the_cutoff():
    gold = {"doc:chunk:7", "doc:chunk:8"}
    ranking = ["doc:chunk:7", "doc:chunk:100", "doc:chunk:101", "doc:chunk:102", "doc:chunk:8"]

    assert recall_at_k(ranking, gold, k=4) == 0.5
    assert recall_at_k(ranking, gold, k=5) == 1.0
