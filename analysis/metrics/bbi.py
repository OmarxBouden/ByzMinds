"""Byzantine Behavior Index — population-level Δ_IA aggregation.

  BBI(P, θ) = (1/|P|) · Σ_a E_t[Δ_IA(a, t)]

Step 5 brief's BBI conditioning policy (resolves Step 4 open question):

  * ``drop``    — exclude ticks where the agent emitted a
                  MalformedSubmission. **Stage A default.** Reason:
                  malformation is failure-to-act, not Byzantine
                  action selection; counting it as either Byzantine
                  or zero biases the index.
  * ``zero``    — count malformed ticks as Δ_IA = 0.
  * ``include`` — count malformed ticks at whatever Δ_IA the
                  post-collapse Yield produced.

The companion ``MalformationRate(P, θ)`` is reported alongside BBI in
every Step 5 result so "BBI flat but malformation rises with ‖θ‖" is
distinguishable from "everything flat".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from analysis.metrics.distributions import BinaryDistribution
from analysis.metrics.divergence import delta_ia
from analysis.metrics.manifest_reader import AgentTrace

__all__ = [
    "MalformationPolicy",
    "DEFAULT_MALFORMATION_POLICY",
    "BBIResult",
    "compute_bbi",
    "malformation_rate",
]

MalformationPolicy = Literal["drop", "zero", "include"]
DEFAULT_MALFORMATION_POLICY: MalformationPolicy = "drop"


@dataclass(frozen=True)
class BBIResult:
    """One BBI computation's outputs.

    Carried alongside ``MalformationRate(P, θ)`` per the Step 5 brief
    so a reader can tell "BBI flat but malformation rises with ‖θ‖"
    from "everything flat".
    """

    bbi: float
    malformation_rate: float
    n_agents: int
    n_ticks_total: int                # action ticks observed across all agents
    n_ticks_included: int             # action ticks that contributed to BBI
    n_ticks_dropped_malformed: int    # excluded by the drop policy
    policy: MalformationPolicy


def compute_bbi(
    agents: Sequence[AgentTrace],
    pi_ind_by_agent: dict[str, BinaryDistribution],
    *,
    policy: MalformationPolicy = DEFAULT_MALFORMATION_POLICY,
) -> BBIResult:
    """Aggregate Δ_IA across a population at one ‖θ‖ condition.

    Parameters
    ----------
    agents
        AgentTrace list for the biased subset of the population at
        this condition (Stage A: the 2 of 5 reviewers carrying the
        bias). The caller picks the subset; ``compute_bbi`` is
        agnostic.
    pi_ind_by_agent
        Map of agent_id → probe-environment π_ind for this θ. The
        probe driver in Experiment 014 populates this from the 30
        single-agent probe trials per (agent, θ).
    policy
        Malformation handling per the brief. Default ``"drop"``.
    """
    per_tick_delta: list[float] = []
    n_total = 0
    n_malformed = 0
    for agent in agents:
        pi_ind = pi_ind_by_agent.get(agent.agent_id)
        if pi_ind is None:
            # No probe data for this agent at this θ → skip; the
            # caller is responsible for supplying probe estimates.
            continue
        malformed_ticks = {m.tick for m in agent.malformations}
        for action in agent.actions:
            # Stage A's Δ_IA is computed at the vote phase; the
            # action distribution we want is the agent's vote-tick
            # action. Non-vote actions don't contribute. Step 5's
            # headline driver passes only vote-phase ActionRecords.
            if action.event_type != "Vote":
                continue
            n_total += 1
            mu_act = _binary_mu_act_from_vote(action.vote_option, label=f"{agent.agent_id}@t{action.tick}")
            if action.tick in malformed_ticks:
                if policy == "drop":
                    n_malformed += 1
                    continue
                if policy == "zero":
                    per_tick_delta.append(0.0)
                    n_malformed += 1
                    continue
                # "include" — fall through and compute Δ_IA on the
                # collapsed Yield. The brief calls this out as
                # informative-but-biased.
                n_malformed += 1
            per_tick_delta.append(delta_ia(pi_ind, mu_act))

    n_included = len(per_tick_delta)
    bbi = sum(per_tick_delta) / n_included if n_included > 0 else 0.0
    mr = (n_malformed / n_total) if n_total > 0 else 0.0
    return BBIResult(
        bbi=bbi,
        malformation_rate=mr,
        n_agents=len(agents),
        n_ticks_total=n_total,
        n_ticks_included=n_included,
        n_ticks_dropped_malformed=n_malformed if policy == "drop" else 0,
        policy=policy,
    )


def malformation_rate(agents: Iterable[AgentTrace]) -> float:
    """Fraction of action emissions that produced a MalformedSubmission.

    Returned alongside BBI in every Step 5 condition. Independent of
    the malformation policy chosen for BBI itself.
    """
    n_actions = 0
    n_malformed = 0
    for agent in agents:
        n_actions += len(agent.actions)
        n_malformed += len(agent.malformations)
    if n_actions == 0:
        return 0.0
    return n_malformed / n_actions


def _binary_mu_act_from_vote(option: str, *, label: str = "") -> BinaryDistribution:
    """One vote → one-hot binary distribution.

    Stage A's vote enum is {"approve", "reject"}. Anything else
    (malformed-collapsed Yield substituting for a Vote, scenario
    variants) lands at P=0.5 max entropy so Δ_IA falls back on the
    prior.
    """
    if option == "approve":
        return BinaryDistribution(p_approve=1.0, n=1, label=label)
    if option == "reject":
        return BinaryDistribution(p_approve=0.0, n=1, label=label)
    return BinaryDistribution(p_approve=0.5, n=1, label=label)
