"""
Tests for comparing two eval runs.

A comparison is only valid when both runs answered the same questions against the
same corpus. Comparing runs whose corpus changed underneath them is how a
re-ingest gets mistaken for a retrieval improvement — which is a live risk here,
since the OCR fix changed chunk text without changing chunk ids.
"""

from __future__ import annotations

import pytest

from eval.compare import align_runs, compare_metric


def _result(question: str, **scores):
    return {"question": question, "retrieval_scores": scores}


# ── alignment ─────────────────────────────────────────────────────────────────


def test_runs_are_aligned_by_question_not_by_position():
    """Question order can differ between runs; pairing must survive that."""
    baseline = [_result("A?", hit_rate=1.0), _result("B?", hit_rate=0.0)]
    variant = [_result("B?", hit_rate=1.0), _result("A?", hit_rate=1.0)]

    pairs = align_runs(baseline, variant)

    assert dict(pairs)["A?"] == (1.0, 1.0)
    assert dict(pairs)["B?"] == (0.0, 1.0)


def test_questions_missing_from_one_run_are_excluded():
    """An errored question in one arm must not silently pair with another."""
    baseline = [_result("A?", hit_rate=1.0), _result("B?", hit_rate=1.0)]
    variant = [_result("A?", hit_rate=0.0)]

    pairs = align_runs(baseline, variant)

    assert [question for question, _ in pairs] == ["A?"]


def test_alignment_is_empty_when_runs_share_no_questions():
    pairs = align_runs([_result("A?", hit_rate=1.0)], [_result("Z?", hit_rate=1.0)])

    assert pairs == []


# ── comparison ────────────────────────────────────────────────────────────────


def test_a_clear_improvement_is_reported_as_significant():
    baseline = [_result(f"q{i}?", hit_rate=0.0) for i in range(60)]
    variant = [_result(f"q{i}?", hit_rate=1.0) for i in range(60)]

    result = compare_metric(baseline, variant, metric="hit_rate", seed=0)

    assert result.mean_diff == 1.0
    assert result.significant is True


def test_a_marginal_difference_is_not_reported_as_significant():
    """Two questions differing out of 30 must not read as a real effect."""
    baseline = [_result(f"q{i}?", hit_rate=1.0 if i < 24 else 0.0) for i in range(30)]
    variant = [_result(f"q{i}?", hit_rate=1.0 if i < 26 else 0.0) for i in range(30)]

    result = compare_metric(baseline, variant, metric="hit_rate", seed=0)

    assert result.significant is False


def test_comparing_an_absent_metric_raises_rather_than_scoring_zero():
    """A typo'd metric name must fail loudly, not silently report no difference."""
    baseline = [_result("A?", hit_rate=1.0)]
    variant = [_result("A?", hit_rate=1.0)]

    with pytest.raises(KeyError):
        compare_metric(baseline, variant, metric="ndcg@4", seed=0)


def test_comparison_needs_overlapping_questions():
    with pytest.raises(ValueError):
        compare_metric([_result("A?", hit_rate=1.0)], [_result("Z?", hit_rate=1.0)], metric="hit_rate", seed=0)
