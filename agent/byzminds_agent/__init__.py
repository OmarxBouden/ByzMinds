"""ByzMinds Python agent runtime.

The agent process is the LLM-driven counterpart to the Go kernel: it
subscribes to per-tick views, runs a forward pass with activation
steering installed, and emits typed events back to the kernel.

Package layout follows byzminds-step3-implementation.md §"Concrete
deliverable layout".
"""

__version__ = "0.3.0"

# Six dials per byzminds-state-summary.md "Six bias dials". This tuple is the
# CANONICAL ordering of the length-6 theta vector recorded on L_cog_ind; the Go
# kernel's --agent-theta table must mirror it (asserted by a cross-language test).
DIALS: tuple[str, ...] = (
    "authority",
    "bandwagon",
    "sycophancy",
    "free_ride",
    "collude",
    "deceive",
)

# Strength -> theta magnitude (theta in [0,1]). Recorded on L_cog_ind as the
# ground-truth induced disposition. In Stage A the persona PROMPT is the
# behavioral mechanism; theta is the recorded label, not a steering signal
# (activation steering is Stage B / M9).
STRENGTH_MAGNITUDE: dict[str, float] = {
    "none": 0.0,
    "mild": 1.0 / 3.0,
    "moderate": 2.0 / 3.0,
    "strong": 1.0,
}


def theta_for_persona(dial: str, strength: str) -> list[float]:
    """Map a (dial, strength) persona to a length-6 theta vector in DIALS order:
    zero everywhere except the dial's index, set to the strength magnitude.
    An empty dial or 'none' strength -> all zeros (honest)."""
    theta = [0.0] * len(DIALS)
    if not dial or strength == "none":
        return theta
    if dial not in DIALS:
        raise ValueError(f"unknown dial {dial!r}; expected one of {DIALS}")
    if strength not in STRENGTH_MAGNITUDE:
        raise ValueError(f"unknown strength {strength!r}; expected one of {list(STRENGTH_MAGNITUDE)}")
    theta[DIALS.index(dial)] = STRENGTH_MAGNITUDE[strength]
    return theta
