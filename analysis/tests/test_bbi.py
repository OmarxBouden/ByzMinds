"""Unit tests for analysis/metrics/bbi.py — malformation policy + BBI math."""

from __future__ import annotations

import pytest

from analysis.metrics.bbi import (
    DEFAULT_MALFORMATION_POLICY,
    compute_bbi,
    malformation_rate,
)
from analysis.metrics.distributions import BinaryDistribution
from analysis.metrics.manifest_reader import (
    ActionRecord,
    AgentTrace,
    MalformedRecord,
)


def _agent_with(
    *,
    agent_id: str,
    votes: list[tuple[int, str]],
    malformed_ticks: tuple[int, ...] = (),
) -> AgentTrace:
    actions = [
        ActionRecord(
            tick=t, global_commit_seq=t + 1, event_type="Vote", vote_option=opt
        )
        for t, opt in votes
    ]
    malformations = [
        MalformedRecord(tick=t, raw_output=b"x", failure="schema_mismatch")
        for t in malformed_ticks
    ]
    return AgentTrace(
        agent_id=agent_id,
        pubkey_hex="00",
        role="reviewer",
        spawn_tick=0,
        initial_theta=(0.0,) * 6,
        actions=actions,
        cog_ind=[],
        cog_eli=[],
        malformations=malformations,
        truncations=[],
        timeouts=[],
    )


def test_default_policy_is_drop():
    assert DEFAULT_MALFORMATION_POLICY == "drop"


def test_bbi_zero_when_intent_and_action_distributions_align():
    """π_ind and μ_act drawn from the same sample size and same
    P(approve) give exactly zero Δ_IA after smoothing. Used as a
    sanity check that the metric pipeline produces zero on perfect
    match. In production, π_ind has n=30 (probe trials) and μ_act
    has n=1 (one vote), so a smoothing-induced positive floor is
    expected — see ``test_bbi_smoothing_floor`` below."""
    a = _agent_with(agent_id="a", votes=[(1, "approve"), (2, "approve")])
    # n=1 matches the per-vote BinaryDistribution constructed inside compute_bbi.
    pi = {"a": BinaryDistribution(p_approve=1.0, n=1)}
    r = compute_bbi([a], pi)
    assert r.bbi == pytest.approx(0.0, abs=1e-9)
    assert r.n_ticks_total == 2
    assert r.n_ticks_included == 2
    assert r.malformation_rate == 0.0


def test_bbi_smoothing_floor_under_realistic_sample_sizes():
    """Realistic Stage A setup: π_ind from 30 probe trials,
    μ_act from one in-scenario vote. Even when both express "approve",
    the differing sample sizes produce a smoothing-induced positive
    floor. The headline notebook reports this as the no-bias baseline
    of BBI — the brief notes baseline subtraction is in place for
    Δ_cog and the same logic applies to BBI's floor."""
    a = _agent_with(agent_id="a", votes=[(1, "approve"), (2, "approve")])
    pi = {"a": BinaryDistribution(p_approve=1.0, n=30)}
    r = compute_bbi([a], pi)
    assert r.bbi > 0  # smoothing floor present
    # But it's a small floor, not a bias-magnitude signal.
    assert r.bbi < 0.5


def test_bbi_positive_when_intent_diverges_from_action():
    a = _agent_with(agent_id="a", votes=[(1, "reject"), (2, "reject")])
    pi = {"a": BinaryDistribution(p_approve=1.0, n=30)}
    r = compute_bbi([a], pi)
    assert r.bbi > 0


def test_drop_policy_excludes_malformed_ticks_from_bbi():
    """Drop policy excludes malformed ticks from the BBI sum.
    Verified by comparing drop vs include policies on the same agent:
    drop's BBI uses only clean ticks, include's BBI uses all ticks."""
    a = _agent_with(
        agent_id="a",
        votes=[(1, "approve"), (2, "reject"), (3, "approve"), (4, "reject")],
        malformed_ticks=(2, 4),
    )
    pi = {"a": BinaryDistribution(p_approve=1.0, n=1)}  # matched sample size
    drop = compute_bbi([a], pi, policy="drop")
    include = compute_bbi([a], pi, policy="include")
    assert drop.n_ticks_total == 4
    assert drop.n_ticks_included == 2
    assert drop.n_ticks_dropped_malformed == 2
    # Drop sees only the two approves (matching π_ind) → BBI ≈ 0.
    assert drop.bbi == pytest.approx(0.0, abs=1e-9)
    # Include sees the two rejects (diverging from π_ind) → BBI > drop.
    assert include.n_ticks_included == 4
    assert include.bbi > drop.bbi
    # Malformation rate is independent of BBI policy.
    assert drop.malformation_rate == 0.5
    assert include.malformation_rate == 0.5


def test_zero_policy_counts_malformed_as_zero_delta():
    a = _agent_with(
        agent_id="a",
        votes=[(1, "reject")],
        malformed_ticks=(1,),
    )
    pi = {"a": BinaryDistribution(p_approve=1.0, n=1)}
    drop_result = compute_bbi([a], pi, policy="drop")
    zero_result = compute_bbi([a], pi, policy="zero")
    assert drop_result.n_ticks_included == 0
    assert drop_result.bbi == 0.0
    assert zero_result.n_ticks_included == 1
    assert zero_result.bbi == pytest.approx(0.0, abs=1e-9)


def test_include_policy_falls_through_to_collapsed_action():
    """``include`` lets the post-collapse action contribute its real
    Δ_IA. Used for the supplementary panel that shows what BBI looks
    like under the alternative conditioning."""
    a = _agent_with(
        agent_id="a",
        votes=[(1, "reject")],
        malformed_ticks=(1,),
    )
    pi = {"a": BinaryDistribution(p_approve=1.0, n=30)}
    include_result = compute_bbi([a], pi, policy="include")
    assert include_result.n_ticks_included == 1
    assert include_result.bbi > 0  # reject vs π_ind=approve


def test_bbi_skips_agents_with_no_probe_data():
    a = _agent_with(agent_id="a", votes=[(1, "reject")])
    pi: dict = {}  # no probe data for "a"
    r = compute_bbi([a], pi)
    assert r.n_ticks_total == 0
    assert r.bbi == 0.0


def test_bbi_only_uses_vote_events():
    """Speak / Yield events don't contribute to BBI (BBI is computed
    at the vote phase per Stage A scope)."""
    a = AgentTrace(
        agent_id="a",
        pubkey_hex="00",
        role="reviewer",
        spawn_tick=0,
        initial_theta=(0.0,) * 6,
        actions=[
            ActionRecord(tick=1, global_commit_seq=1, event_type="Speak", speak_channel="public", speak_content="hi"),
            ActionRecord(tick=2, global_commit_seq=2, event_type="Yield", yield_reason="thinking"),
        ],
        cog_ind=[],
        cog_eli=[],
        malformations=[],
        truncations=[],
        timeouts=[],
    )
    pi = {"a": BinaryDistribution(p_approve=1.0, n=30)}
    r = compute_bbi([a], pi)
    assert r.n_ticks_total == 0
    assert r.bbi == 0.0


def test_malformation_rate_helper_matches_bbi_result():
    a = _agent_with(
        agent_id="a",
        votes=[(1, "approve"), (2, "approve")],
        malformed_ticks=(2,),
    )
    rate = malformation_rate([a])
    assert rate == 0.5  # 1 malformation / 2 actions
