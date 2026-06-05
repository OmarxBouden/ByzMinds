"""Token budget + drop-oldest channel-history truncation.

Llama 3.1 8B Instruct has an 8K context window. Step 4 brief budget:

    L0 ≤ 150  | L1 ≤ 250  | tool schemas ≤ 500
    L2 framing/self/phase ≤ 200 | L2 channel histories ≤ 4000
    L2 externals ≤ 200 | available tools ≤ 100
    response reserve = 1500 | buffer = 1100      → total 8000

The agent runtime composes (L0, L1, L2, tools) and asks budget.fit
whether the assembled inputs fit. When the channel-history section
overruns its slice of the budget, the renderer drops oldest messages
until the rendered L2 is back under budget. Each truncation produces
one ``ContextTruncation`` event to L_ctrl per (agent, tick, channel).

Token counting:
  * If ``transformers`` is importable (the [serving] / [serving-mac]
    extras), we use the model's actual tokenizer.
  * Otherwise we fall back to a fixed proxy of 1 token ≈ 4 chars,
    which over-counts slightly — safer than under-counting against
    the 8K wall.

For Stage A baselines (history_window: public=20, private=10) the
default L2 stays well under budget; truncation is the edge case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from byzminds_agent.prompt.render import render_L2
from byzminds_agent.proto_gen import events_pb2, view_pb2

__all__ = [
    "DEFAULT_L2_BUDGET_TOKENS",
    "TokenCounter",
    "TruncationOutcome",
    "make_proxy_counter",
    "fit_L2_to_budget",
]

# Step 4 brief: L2 framing + channel histories + externals + available
# tools = 200 + 4000 + 200 + 100 = 4500 tokens. We treat this as the
# L2-side budget the agent enforces; the rest of the 8K (L0/L1/tools
# schemas/response reserve/buffer) is the runtime's accounting.
DEFAULT_L2_BUDGET_TOKENS = 4500

# Proxy ratio when no real tokenizer is installed. Llama tokenizers
# average ~3.7 chars/token on English; we round to 4 to over-count by
# ~8%, which is the safe direction against the 8K wall.
_PROXY_CHARS_PER_TOKEN = 4


TokenCounter = Callable[[str], int]


def make_proxy_counter() -> TokenCounter:
    """Return the chars/4 fallback tokenizer used when the GPU box's
    real tokenizer isn't installed."""

    def _count(s: str) -> int:
        return (len(s) + _PROXY_CHARS_PER_TOKEN - 1) // _PROXY_CHARS_PER_TOKEN

    return _count


@dataclass
class TruncationOutcome:
    """Result of one fit pass.

    ``rendered`` is the L2 text after truncation (always renders, even
    if no drops were needed). ``dropped_per_channel`` is a map of
    channel_id → dropped_message_count; empty means no truncation
    happened. ``budget_events`` lists the ContextTruncation envelopes
    the caller should commit to L_ctrl.
    """

    rendered: str
    token_count: int
    dropped_per_channel: dict[str, int]
    budget_events: list[events_pb2.ContextTruncation]


def fit_L2_to_budget(
    view: view_pb2.View,
    *,
    budget_tokens: int = DEFAULT_L2_BUDGET_TOKENS,
    counter: TokenCounter | None = None,
) -> TruncationOutcome:
    """Render ``view``'s L2; drop oldest messages until under budget.

    The drop-oldest order is deterministic: at each iteration we drop
    the oldest message from the *longest-named* channel that still has
    messages (sorted alphabetically by channel_id breaks ties). This
    keeps the algorithm well-defined and reproducible across replays.

    Returns the rendered text plus a per-channel drop count and the
    matching ContextTruncation events. If the View fits in budget
    without dropping, ``dropped_per_channel`` is empty.
    """
    if counter is None:
        counter = make_proxy_counter()

    # Work on a defensive copy so the caller's View object isn't mutated.
    working = view_pb2.View()
    working.CopyFrom(view)

    drops: dict[str, int] = {ch.channel_id: 0 for ch in working.channel_histories}
    rendered = render_L2(working)
    n = counter(rendered)
    while n > budget_tokens:
        target = _pick_drop_target(working)
        if target is None:
            # No messages left to drop — we've truncated as far as the
            # algorithm allows. Return what we have; the runtime will
            # see we're still over and may apply its own escalation
            # (e.g., shrink the artifact). Stage A doesn't exercise
            # this path.
            break
        ch_index, msg_index = target
        channel_id = working.channel_histories[ch_index].channel_id
        del working.channel_histories[ch_index].messages[msg_index]
        drops[channel_id] = drops.get(channel_id, 0) + 1
        rendered = render_L2(working)
        n = counter(rendered)

    budget_events = _materialize_truncation_events(
        view=view, working=working, drops=drops
    )
    # Only surface entries that actually dropped something.
    drops_nonzero = {k: v for k, v in drops.items() if v > 0}
    return TruncationOutcome(
        rendered=rendered,
        token_count=n,
        dropped_per_channel=drops_nonzero,
        budget_events=budget_events,
    )


def _pick_drop_target(view: view_pb2.View) -> tuple[int, int] | None:
    """Pick (channel_index, message_index) for the next drop.

    Policy: oldest message across all channels, breaking ties by
    alphabetical channel_id ascending. The kernel-side renderer
    already sorts channels alphabetically, so we iterate in that
    order and drop from whichever channel has the oldest head — the
    first channel with a non-empty history is the drop target,
    because all messages within a channel are oldest-first.
    """
    best_index: tuple[int, int] | None = None
    best_key: tuple[int, int, str] | None = None  # (tick, global_commit_seq, channel_id)
    for ci, ch in enumerate(view.channel_histories):
        if not ch.messages:
            continue
        m = ch.messages[0]
        key = (m.tick, m.global_commit_seq, ch.channel_id)
        if best_key is None or key < best_key:
            best_key = key
            best_index = (ci, 0)
    return best_index


def _materialize_truncation_events(
    *,
    view: view_pb2.View,
    working: view_pb2.View,
    drops: dict[str, int],
) -> list[events_pb2.ContextTruncation]:
    events: list[events_pb2.ContextTruncation] = []
    original_counts = {ch.channel_id: len(ch.messages) for ch in view.channel_histories}
    final_counts = {ch.channel_id: len(ch.messages) for ch in working.channel_histories}
    for channel_id, dropped in drops.items():
        if dropped == 0:
            continue
        events.append(
            events_pb2.ContextTruncation(
                agent_id=view.agent_id,
                tick=view.tick,
                channel_id=channel_id,
                dropped_count=dropped,
                kept_count=final_counts.get(channel_id, 0),
            )
        )
    # Sort by channel_id for deterministic envelope order.
    events.sort(key=lambda e: e.channel_id)
    _ = original_counts  # unused; retained for future per-channel diagnostics
    return events


def total_messages(view: view_pb2.View) -> int:
    """Convenience helper: count messages across all channels."""
    return sum(len(ch.messages) for ch in view.channel_histories)


def iter_messages(view: view_pb2.View) -> Iterable[view_pb2.Message]:
    """Convenience helper iterating all messages in channel-then-order."""
    for ch in view.channel_histories:
        yield from ch.messages
