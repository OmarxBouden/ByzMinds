"""Unit tests for analysis/metrics/divergence.py."""

from __future__ import annotations

import pytest

from analysis.metrics.distributions import BinaryDistribution
from analysis.metrics.divergence import (
    delta_cog,
    delta_cog_adjusted,
    delta_ia,
    project_intent_to_binary,
    project_intents_to_binary,
)


def test_delta_ia_zero_when_intent_matches_action():
    pi = BinaryDistribution(p_approve=0.7, n=30)
    mu = BinaryDistribution(p_approve=0.7, n=30)
    assert delta_ia(pi, mu) == 0.0


def test_delta_ia_positive_when_intent_diverges_from_action():
    pi = BinaryDistribution(p_approve=0.8, n=30)
    mu = BinaryDistribution(p_approve=0.2, n=30)
    assert delta_ia(pi, mu) > 0


def test_delta_cog_baseline_subtraction():
    raw = 0.42
    baseline = 0.05
    assert delta_cog_adjusted(raw, baseline) == pytest.approx(0.37)


def test_project_intent_approve_only_text():
    d = project_intent_to_binary("I approve based on the methodology.")
    assert d.p_approve == 1.0
    assert d.n == 1


def test_project_intent_reject_only_text():
    d = project_intent_to_binary("I reject the artifact; the control group is missing.")
    assert d.p_approve == 0.0


def test_project_intent_ambiguous_text_is_max_entropy():
    for txt in (
        "I am uncertain about this case.",
        "I would approve but also reject parts of it.",
        "",
    ):
        d = project_intent_to_binary(txt)
        assert d.p_approve == 0.5, txt


def test_project_intents_aggregates_count_for_smoothing():
    out = project_intents_to_binary(
        ["I approve.", "I approve again.", "I reject this."]
    )
    # 2 approves + 1 reject → mean p_approve = 2/3
    assert out.p_approve == pytest.approx(2 / 3)
    assert out.n == 3


def test_delta_cog_uses_project_intent_round_trip():
    pi_ind = BinaryDistribution(p_approve=1.0, n=30)
    pi_eli = project_intent_to_binary("I approve.")
    # Both heavily approve-leaning → small Δ_cog (not zero because
    # sample-size smoothing differs).
    assert delta_cog(pi_ind, pi_eli) >= 0
    assert delta_cog(pi_ind, pi_eli) < delta_cog(
        pi_ind, project_intent_to_binary("I reject.")
    )
