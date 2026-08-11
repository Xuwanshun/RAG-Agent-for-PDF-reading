"""
Tests for the adapters between pipeline output and retrieval metrics.

The metrics themselves are pure and tested separately. What breaks in practice is
the wiring: pulling chunk ids out of a QAResponse in the right order, and finding
the gold labels on the dataset example. A silent mismatch here reports 0.0 across
the board and looks like catastrophic retrieval rather than a broken adapter.
"""

from __future__ import annotations

from eval.evaluators import extract_gold_chunk_ids, extract_retrieved_chunk_ids, score_retrieval


def _outputs(*chunk_ids):
    return {"sources": [{"chunk_id": cid, "score": 0.9} for cid in chunk_ids]}


# ── extraction ────────────────────────────────────────────────────────────────


def test_retrieved_ids_are_extracted_in_rank_order():
    """Rank order is the whole point — nDCG and MRR both depend on it."""
    assert extract_retrieved_chunk_ids(_outputs("c:1", "c:2", "c:3")) == ["c:1", "c:2", "c:3"]


def test_missing_sources_yields_an_empty_list():
    assert extract_retrieved_chunk_ids({}) == []
    assert extract_retrieved_chunk_ids({"sources": []}) == []


def test_sources_without_a_chunk_id_are_skipped():
    outputs = {"sources": [{"score": 0.9}, {"chunk_id": "c:2"}]}

    assert extract_retrieved_chunk_ids(outputs) == ["c:2"]


def test_gold_ids_are_read_from_the_dataset_example():
    assert extract_gold_chunk_ids({"relevant_chunk_ids": ["c:7"]}) == ["c:7"]


def test_missing_gold_labels_yield_an_empty_list():
    assert extract_gold_chunk_ids({}) == []
    assert extract_gold_chunk_ids({"relevant_chunk_ids": None}) == []


# ── scoring ───────────────────────────────────────────────────────────────────


def test_score_retrieval_reports_every_metric():
    scores = score_retrieval(_outputs("c:7", "c:2"), {"relevant_chunk_ids": ["c:7"]}, k=4)

    assert set(scores) == {"hit_rate", "mrr", "ndcg@4", "precision@1", "recall@4"}


def test_a_perfect_retrieval_scores_one_across_the_board():
    scores = score_retrieval(_outputs("c:7", "c:2", "c:3"), {"relevant_chunk_ids": ["c:7"]}, k=4)

    assert scores["hit_rate"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["ndcg@4"] == 1.0
    assert scores["precision@1"] == 1.0
    assert scores["recall@4"] == 1.0


def test_a_gold_chunk_ranked_lower_costs_rank_sensitive_metrics_only():
    """recall and hit_rate cannot see rank; MRR and nDCG must."""
    scores = score_retrieval(_outputs("c:1", "c:2", "c:7"), {"relevant_chunk_ids": ["c:7"]}, k=4)

    assert scores["hit_rate"] == 1.0
    assert scores["recall@4"] == 1.0
    assert scores["precision@1"] == 0.0
    assert round(scores["mrr"], 4) == 0.3333
    assert scores["ndcg@4"] < 1.0


def test_a_complete_miss_scores_zero():
    scores = score_retrieval(_outputs("c:1", "c:2"), {"relevant_chunk_ids": ["c:7"]}, k=4)

    assert set(scores.values()) == {0.0}


def test_k_controls_the_cutoff():
    """A gold chunk beyond k never reaches the synthesiser, so it must not count."""
    outputs = _outputs("c:1", "c:2", "c:3", "c:4", "c:7")

    assert score_retrieval(outputs, {"relevant_chunk_ids": ["c:7"]}, k=4)["recall@4"] == 0.0
    assert score_retrieval(outputs, {"relevant_chunk_ids": ["c:7"]}, k=5)["recall@5"] == 1.0


def test_examples_without_gold_labels_score_zero_rather_than_crashing():
    """Some questions may be added without labels; the run must not die."""
    scores = score_retrieval(_outputs("c:1"), {}, k=4)

    assert scores["hit_rate"] == 0.0
