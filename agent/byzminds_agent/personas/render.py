"""Render a dial persona at a given strength into an L1 system message.

Templates live at ``agent/data/personas/{dial}_{strength}.txt`` (researcher-
authored; reproduced in paper appendix A.5). Strength ``none`` returns the
empty string (no induced disposition — the dose-response baseline).
"""

from __future__ import annotations

from pathlib import Path

from byzminds_agent import DIALS

STRENGTHS = ["none", "mild", "moderate", "strong"]
PERSONA_DIR = Path(__file__).resolve().parents[2] / "data" / "personas"


def render_persona(dial: str, strength: str) -> str:
    """Return the L1 persona text for (dial, strength). ``none`` -> ''."""
    if dial not in DIALS:
        raise ValueError(f"unknown dial {dial!r}; expected one of {DIALS}")
    if strength not in STRENGTHS:
        raise ValueError(f"unknown strength {strength!r}; expected one of {STRENGTHS}")
    if strength == "none":
        return ""
    path = PERSONA_DIR / f"{dial}_{strength}.txt"
    if not path.exists():
        raise SystemExit(
            f"persona template missing: {path}. Author it (researcher-owned, "
            "appendix A.5) or run M3 only on dials whose templates exist."
        )
    return path.read_text().strip()


def available_dials() -> list[str]:
    """Dials that have at least the 'strong' template authored."""
    return [d for d in DIALS if (PERSONA_DIR / f"{d}_strong.txt").exists()]
