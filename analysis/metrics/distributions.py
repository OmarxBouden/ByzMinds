"""Distribution representation + KL divergence helpers.

Stage A's action space is binary at the vote phase (approve / reject).
``BinaryDistribution`` carries the observed probability of approve and
the sample size used to estimate it; KL divergence has the closed
form

    D_KL(p || q) = p log(p/q) + (1-p) log((1-p)/(1-q))

Zero-probability outcomes get Laplace smoothing with α = 0.5 (Jeffreys
prior) applied symmetrically:

    p_smoothed = (n·p + α) / (n + 2α)

This keeps every probability strictly inside (0, 1) so KL is always
finite. The smoothing alpha is documented in PAPER_NOTES.md.

Multi-class support (channel content, capability requests) is
deferred to Stage B+ — the brief locks Stage A to the binary form
because the headline metric is computed at the vote phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["LAPLACE_ALPHA", "BinaryDistribution", "binary_kl"]

# Laplace / Jeffreys smoothing alpha for zero-probability outcomes,
# per Step 5 brief §Risks ("KL divergence undefined on zero-prob").
LAPLACE_ALPHA = 0.5


@dataclass(frozen=True)
class BinaryDistribution:
    """Observed P(approve) plus the sample size that produced it.

    The sample size matters for the smoothing: a higher n shrinks the
    Laplace correction toward the raw estimate, while a low n leaves
    enough head-room that KL stays finite even on a one-trial sample
    that landed exactly at 0 or 1.

    ``label`` is a free-form tag (e.g. "pi_ind θ=1.0 reviewer_01" or
    "mu_act run17 vote phase") that flows into log messages and
    DataFrames; it has no semantic effect on the math.
    """

    p_approve: float
    n: int
    label: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_approve <= 1.0:
            raise ValueError(
                f"BinaryDistribution.p_approve must be in [0, 1], got {self.p_approve}"
            )
        if self.n < 0:
            raise ValueError(f"BinaryDistribution.n must be >= 0, got {self.n}")

    @property
    def smoothed(self) -> tuple[float, float]:
        """Returns (P(approve), P(reject)) with Laplace smoothing.

        For n=0 the smoothed estimate is (0.5, 0.5) — pure prior, no
        observed information — which yields KL = 0 against any other
        n=0 distribution and a finite, sample-size-aware divergence
        against any non-trivial estimate.
        """
        denom = self.n + 2 * LAPLACE_ALPHA
        p = (self.p_approve * self.n + LAPLACE_ALPHA) / denom
        return p, 1.0 - p


def binary_kl(p: BinaryDistribution, q: BinaryDistribution) -> float:
    """KL divergence D_KL(p || q) over the binary {approve, reject}
    space, with Laplace smoothing on both arms.

    Always finite, always non-negative, zero iff the smoothed
    distributions agree exactly.
    """
    p_a, p_r = p.smoothed
    q_a, q_r = q.smoothed
    return p_a * math.log(p_a / q_a) + p_r * math.log(p_r / q_r)
