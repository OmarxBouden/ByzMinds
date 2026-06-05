"""Prompted-persona bias injection (M3).

Personas are the Stage-A controllable bias mechanism (replacing activation
steering, which is deferred to Stage B). A persona is an L1-layer system
message describing the agent's disposition at one of four strength levels;
:func:`render_persona` loads it. See ``agent/data/personas/`` for templates.
"""

from byzminds_agent.personas.render import (  # noqa: F401
    STRENGTHS,
    render_persona,
)
