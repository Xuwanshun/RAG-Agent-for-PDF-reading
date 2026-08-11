"""
Significance testing for eval comparisons.

Two configurations are always scored on the SAME question set, which makes the
observations paired: what matters is the per-question difference, not the gap
between two independent means. Paired resampling exploits that and gives a much
tighter, more honest interval than treating the arms as independent.

Why this exists: the historical runs reported hit_rate moving 0.80 -> 0.8667 on
30 questions and read it as an improvement. That is two questions. Without an
interval there is no way to tell a real effect from resampling noise, and every
ablation conclusion inherits that ambiguity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class PairedComparison:
    """Outcome of comparing a variant against a baseline on the same questions."""

    mean_baseline: float
    mean_variant: float
    # Positive means the variant scored higher. Always variant - baseline, so the
    # sign of a regression is preserved rather than reported as a magnitude.
    mean_diff: float
    ci_low: float
    ci_high: float
    confidence: float
    n: int

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def describe(self) -> str:
        """One-line summary suitable for a report table."""
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.mean_diff:+.4f} "
            f"[{int(self.confidence * 100)}% CI: {self.ci_low:+.4f}, {self.ci_high:+.4f}] "
            f"n={self.n} ({verdict})"
        )


def paired_bootstrap(
    baseline: list[float],
    variant: list[float],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """
    Bootstrap a confidence interval for the mean per-question difference.

    Each resample draws whole ``(baseline, variant)`` pairs with replacement, so
    question difficulty cancels out: a question both configs fail contributes a
    zero difference rather than inflating the variance of both arms.

    ``seed`` is explicit so a reported interval can be reproduced exactly.
    """
    if len(baseline) != len(variant):
        raise ValueError(f"paired comparison needs equal-length arms, got {len(baseline)} and {len(variant)}")
    if not baseline:
        raise ValueError("paired comparison needs at least one question")

    differences = [v - b for b, v in zip(baseline, variant, strict=True)]
    n = len(differences)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        # Resample pair indices, not the two arms independently.
        resampled = [differences[rng.randrange(n)] for _ in range(n)]
        means.append(fmean(resampled))
    means.sort()

    tail = (1.0 - confidence) / 2.0
    low_index = int(tail * iterations)
    high_index = min(iterations - 1, int((1.0 - tail) * iterations))

    return PairedComparison(
        mean_baseline=fmean(baseline),
        mean_variant=fmean(variant),
        mean_diff=fmean(differences),
        ci_low=means[low_index],
        ci_high=means[high_index],
        confidence=confidence,
        n=n,
    )
