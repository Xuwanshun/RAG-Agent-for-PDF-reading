"""
Tests for paired significance testing between two eval configurations.

Two configs are scored on the SAME questions, so the observations are paired and
the per-question difference is the statistic of interest. Comparing bare means
hides how little evidence a small question set carries: across six historical
runs on 30 questions, hit_rate moved 0.80 -> 0.8667, which is two questions.
"""

from __future__ import annotations

from eval.stats import paired_bootstrap


def test_identical_scores_produce_a_zero_difference(tmp_settings):
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    result = paired_bootstrap(scores, scores, seed=0)

    assert result.mean_diff == 0.0
    assert result.n == 6


def test_identical_scores_are_not_significant(tmp_settings):
    """A CI spanning zero means 'no detectable difference'."""
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    result = paired_bootstrap(scores, scores, seed=0)

    assert result.ci_low <= 0.0 <= result.ci_high
    assert result.significant is False


def test_a_two_question_swing_on_thirty_questions_is_not_significant(tmp_settings):
    """The exact situation in the historical runs: 0.80 -> 0.8667 on n=30."""
    baseline = [1.0] * 24 + [0.0] * 6
    variant = [1.0] * 26 + [0.0] * 4

    result = paired_bootstrap(baseline, variant, seed=0)

    assert result.significant is False, "a 2-question swing on n=30 must not read as a real effect"


def test_a_large_consistent_improvement_is_significant(tmp_settings):
    baseline = [0.0] * 100
    variant = [1.0] * 100

    result = paired_bootstrap(baseline, variant, seed=0)

    assert result.mean_diff == 1.0
    assert result.significant is True
    assert result.ci_low > 0.0


def test_direction_is_variant_minus_baseline(tmp_settings):
    """A regression must report a negative difference, not an absolute one."""
    baseline = [1.0] * 50
    variant = [0.0] * 50

    result = paired_bootstrap(baseline, variant, seed=0)

    assert result.mean_diff == -1.0
    assert result.ci_high < 0.0


def test_results_are_reproducible_for_a_fixed_seed(tmp_settings):
    baseline = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    variant = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    first = paired_bootstrap(baseline, variant, seed=42)
    second = paired_bootstrap(baseline, variant, seed=42)

    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_pairing_is_preserved_not_shuffled_independently(tmp_settings):
    """
    Paired resampling must draw whole (baseline, variant) pairs.

    Here every question improves by exactly 0.1, so every paired resample has a
    mean difference of exactly 0.1 and the CI collapses to a point. Sampling the
    two arms independently would produce a visibly wider interval.
    """
    baseline = [0.1, 0.4, 0.6, 0.9, 0.3, 0.7]
    variant = [value + 0.1 for value in baseline]

    result = paired_bootstrap(baseline, variant, seed=0)

    assert round(result.ci_low, 6) == round(result.ci_high, 6) == 0.1


def test_mismatched_lengths_are_rejected(tmp_settings):
    """Unequal arms mean the runs are not comparable question-for-question."""
    try:
        paired_bootstrap([1.0, 0.0], [1.0], seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched arms")


def test_empty_input_is_rejected(tmp_settings):
    try:
        paired_bootstrap([], [], seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty arms")
