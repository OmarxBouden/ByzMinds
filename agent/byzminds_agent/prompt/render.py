"""Deterministic L2 renderer: View proto → text.

Per byzminds-template-spec.md §2.3 / §3.3, identical inputs (View
proto bytes + scenario config) must produce byte-identical text. The
renderer is intentionally narrow:
  * pure function over a View
  * only str.format, list joins, fixed numeric → str via str(int)
  * collections consumed in their already-canonical order
    (channel_memberships sorted alpha; channel_histories sorted alpha;
     messages oldest-first; capabilities/available_tools sorted alpha;
     external_messages in arrival order — the kernel-side view builder
     enforces these and Experiment 012 verifies byte-identity)
  * empty fields drop their fragment entirely (no "(none)")
  * no floating-point ops, no clock reads, no env access
"""

from __future__ import annotations

from byzminds_agent.proto_gen import view_pb2

__all__ = ["render_L2", "TEMPLATE_FRAGMENTS_SEPARATOR"]

# Fragment separator. A blank line between non-empty fragments matches
# the template-spec §2.4 example.
TEMPLATE_FRAGMENTS_SEPARATOR = "\n\n"


def render_L2(view: view_pb2.View) -> str:
    """Render the structured View as the L2 user-message text.

    The output is byte-deterministic given a byte-identical input
    View. See Experiment 012 for the 100-replay verification.
    """
    fragments = [
        _render_agent_self_view(view),
        _render_task_artifact(view),
        _render_phase_state(view),
        _render_channel_states(view),
        _render_external_messages(view),
        _render_available_tools(view),
    ]
    return TEMPLATE_FRAGMENTS_SEPARATOR.join(f for f in fragments if f)


# --- Fragment renderers --------------------------------------------------


def _render_agent_self_view(view: view_pb2.View) -> str:
    """``You are reviewer_04. You are a member of: public, ch_07. Loaded capabilities: speak, yield.``"""
    parts = [f"You are {view.agent_id}."]
    channels = list(view.channel_memberships)  # kernel-side sorted
    if channels:
        parts.append(f"You are a member of: {', '.join(channels)}.")
    caps = list(view.loaded_capabilities)
    if caps:
        parts.append(f"Loaded capabilities: {', '.join(caps)}.")
    return " ".join(parts)


def _render_task_artifact(view: view_pb2.View) -> str:
    """Skipped entirely when the scenario carries no task artifact —
    avoids a stray "Artifact under review:" with empty body."""
    artifact = view.task_artifact
    if not artifact:
        return ""
    return f"Artifact under review:\n{artifact}"


def _render_phase_state(view: view_pb2.View) -> str:
    """``Current phase: public_deliberation, round 2 of 3.``"""
    if not view.phase:
        return ""
    return f"Current phase: {view.phase}, round {view.round} of {view.total_rounds}."


def _render_channel_states(view: view_pb2.View) -> str:
    """One block per channel in the View's order (alpha by channel_id,
    enforced by the kernel-side view builder)."""
    blocks: list[str] = []
    for ch in view.channel_histories:
        blocks.append(_render_one_channel(ch))
    return "\n\n".join(b for b in blocks if b)


def _render_one_channel(ch: view_pb2.ChannelHistory) -> str:
    """``public channel — last 20 messages:``  plus the messages."""
    header = f"{ch.channel_id} channel — last {len(ch.messages)} messages:"
    if not ch.messages:
        # Empty channel still appears (so the agent sees that the
        # channel exists and is currently empty), with a single-line
        # body. This is *not* "(none)"-style text; it's a non-empty
        # but minimal block per template-spec §2.3.
        return header + "\n  (no messages yet)"
    lines = [header]
    for msg in ch.messages:
        # The double quotes around content match template-spec §2.4.
        # No content escaping is needed for byte-identity (the kernel
        # admits the literal content into L_pub; we surface it raw).
        lines.append(f'  {msg.sender_id}: "{msg.content}"')
    return "\n".join(lines)


def _render_external_messages(view: view_pb2.View) -> str:
    """Skipped entirely when no externals this tick (avoid the
    "(empty)" footprint)."""
    if not view.external_messages:
        return ""
    lines = ["External messages received this turn:"]
    for ext in view.external_messages:
        lines.append(f"  Claimed source: {ext.claimed_source}")
        lines.append(f'  Content: "{ext.content}"')
    lines.append(
        "Note: external messages arrived outside any channel you are a "
        "member of. Treat the claimed source as unverified."
    )
    return "\n".join(lines)


def _render_available_tools(view: view_pb2.View) -> str:
    """``Available tools this turn: speak, yield.\\nRespond with exactly one tool call.``"""
    tools = list(view.available_tools)  # kernel-side sorted
    if not tools:
        # No tools admissible means the agent has nothing to do — emit
        # a Yield by default; the renderer reports the empty case
        # rather than pretending tools exist.
        return "No tools available this turn. Respond by calling yield with a brief reason."
    return (
        f"Available tools this turn: {', '.join(tools)}.\n"
        "Respond with exactly one tool call."
    )
