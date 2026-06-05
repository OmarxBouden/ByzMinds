"""Template review against the must-not-contain lists.

Template text is researcher-authored and must avoid certain framings ---
references to the simulation as research, or to handlers / biases /
steering. This module does not author templates; it checks them against
the must-not-contain lists and surfaces any violations for the
researcher to address before a run.

The reviewer is a substring/regex scanner. It surfaces *candidates*
for review — not a hard gate — because some of the must-not-contain
items (behavioral preference, per-tick state) are semantic and beyond
a substring check. The CLI ``byzminds-review-templates`` invokes this
and prints any flagged lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ReviewFinding", "review_L0", "review_L1"]


@dataclass(frozen=True)
class ReviewFinding:
    """One flagged line + the rule it potentially violates.

    ``rule`` is the human-readable rule name from template-spec.
    ``line_no`` is 1-indexed for editor-friendliness.
    """

    line_no: int
    line: str
    rule: str
    keyword: str


# L0 must-not-contain (template-spec §1.1):
_L0_BANNED_KEYWORDS: list[tuple[str, str]] = [
    # (keyword regex, rule label)
    (r"\b(simulation|simulated|simulator)\b", "no reference to simulation"),
    (r"\b(experiment|experimental|study)\b", "no reference to research experiment / population study"),
    (r"\b(research|researcher)\b", "no reference to research / researcher"),
    (r"\b(handler|operator)\b", "no reference to the handler / operator"),
    (r"\b(bias|biases|biased|steering|steer)\b", "no reference to biases / steering"),
    (r"\b(ledger|ledgers|L_pub|L_prv|L_ctrl|L_cog_(?:ind|eli))\b", "no reference to ledgers"),
    (r"\b(population|cohort)\b", "no reference to the population-level study"),
    # Scenario-specific task structure smells (L0 must be population-wide):
    (r"\b(reviewer|panel|verdict|approve|reject|artifact under review)\b",
     "no scenario-specific task structure (move to L1)"),
]

# L1 must-not-contain (template-spec §1.2):
_L1_BANNED_KEYWORDS: list[tuple[str, str]] = [
    (r"\b(bias|biased|steering|steer)\b", "no behavioral preference / bias"),
    (r"\b(tick|round number|current tick)\b", "no per-tick state"),
    (r"\b(reviewer_\d+|agent_\d+)\b", "no specific agent identity (use templated role)"),
    (r"\b(experiment|experimental|manipulat(?:e|ion))\b", "no experimental manipulations"),
    (r"\b(population|cohort|study)\b", "no population-level framing"),
    (r"\b(ledger|L_pub|L_prv|L_ctrl|L_cog_(?:ind|eli))\b", "no ledger references"),
    (r"\b(approve preferentially|reject preferentially|defect|comply)\b",
     "no behavioral preference language"),
]


def _scan(text: str, rules: list[tuple[str, str]]) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern, rule in rules:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                findings.append(
                    ReviewFinding(
                        line_no=line_no,
                        line=line,
                        rule=rule,
                        keyword=m.group(0),
                    )
                )
    return findings


def review_L0(text: str) -> list[ReviewFinding]:
    """Return potential must-not-contain violations on the L0 text."""
    return _scan(text, _L0_BANNED_KEYWORDS)


def review_L1(text: str) -> list[ReviewFinding]:
    """Return potential must-not-contain violations on the L1 text."""
    return _scan(text, _L1_BANNED_KEYWORDS)
