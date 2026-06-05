"""Unit tests for analysis/metrics/distributions.py."""

from __future__ import annotations

import math

import pytest

from analysis.metrics.distributions import (
    LAPLACE_ALPHA,
    BinaryDistribution,
    binary_kl,
)


def test_validators_reject_out_of_range():
    with pytest.raises(ValueError):
        BinaryDistribution(p_approve=-0.1, n=10)
    with pytest.raises(ValueError):
        BinaryDistribution(p_approve=1.1, n=10)
    with pytest.raises(ValueError):
        BinaryDistribution(p_approve=0.5, n=-1)


def test_kl_self_is_zero():
    d = BinaryDistribution(p_approve=0.7, n=30)
    assert binary_kl(d, d) == 0.0


def test_kl_extremes_are_finite_under_smoothing():
    """KL(0 || 1) is +∞ without smoothing; we use Laplace so it
    stays finite. The exact value depends on the smoothing alpha
    and the sample size; we just assert finiteness here."""
    p = BinaryDistribution(p_approve=0.0, n=30)
    q = BinaryDistribution(p_approve=1.0, n=30)
    a = binary_kl(p, q)
    b = binary_kl(q, p)
    assert math.isfinite(a)
    assert math.isfinite(b)
    assert a > 0
    assert b > 0


def test_kl_nonnegative_and_increases_with_separation():
    base = BinaryDistribution(p_approve=0.5, n=30)
    near = BinaryDistribution(p_approve=0.55, n=30)
    far = BinaryDistribution(p_approve=0.9, n=30)
    d_near = binary_kl(base, near)
    d_far = binary_kl(base, far)
    assert 0.0 <= d_near < d_far


def test_smoothing_keeps_probability_strictly_in_open_unit_interval():
    p = BinaryDistribution(p_approve=0.0, n=10)
    p_a, p_r = p.smoothed
    assert 0.0 < p_a < 1.0
    assert 0.0 < p_r < 1.0


def test_smoothing_with_zero_sample_is_uniform_prior():
    """n=0 → smoothed (0.5, 0.5). KL between two such distributions is 0."""
    a = BinaryDistribution(p_approve=0.0, n=0)
    b = BinaryDistribution(p_approve=1.0, n=0)
    p_a, p_r = a.smoothed
    assert abs(p_a - 0.5) < 1e-12
    assert binary_kl(a, b) == 0.0


def test_smoothing_alpha_locked_to_half():
    """The brief locks α=0.5 (Jeffreys). Surface it as a sanity check
    so an accidental change shows up in the test diff."""
    assert LAPLACE_ALPHA == 0.5


def test_kl_symmetric_under_label_swap():
    """KL is not symmetric in general, but swapping approve↔reject
    on both arms must preserve KL."""
    p = BinaryDistribution(p_approve=0.3, n=30)
    q = BinaryDistribution(p_approve=0.6, n=30)
    p_swap = BinaryDistribution(p_approve=1 - p.p_approve, n=p.n)
    q_swap = BinaryDistribution(p_approve=1 - q.p_approve, n=q.n)
    assert binary_kl(p, q) == pytest.approx(binary_kl(p_swap, q_swap), rel=1e-12)
