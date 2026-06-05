"""Pydantic mirrors of the eight Stage A event payloads in events.proto.

Field names use the proto's wire names (snake_case). Stage A
admissibility ranges (Speak.content length, OpenChannelReq member
count, Yield.reason length) match the Go ``internal/schema`` package so
malformed tool calls get caught client-side before a kernel round-trip.

Per byzminds-template-spec.md §6, double validation is intentional:
the Python adapter gives the agent a fast feedback loop, and the Go
kernel re-validates so a buggy or compromised adapter cannot bypass
admissibility.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Length bounds — kept in lockstep with kernel/internal/schema/validate.go.
MAX_SPEAK_CONTENT_BYTES = 4 * 1024
MAX_YIELD_REASON_BYTES = 200
MAX_DECLARE_INTENT_BYTES = 4 * 1024


class _Strict(BaseModel):
    """Forbid unknown fields so a hallucinated argument is a hard error
    rather than silently dropped."""

    model_config = ConfigDict(extra="forbid")


class Speak(_Strict):
    channel_id: Annotated[str, StringConstraints(min_length=1)]
    content: Annotated[str, StringConstraints(max_length=MAX_SPEAK_CONTENT_BYTES)]


class Vote(_Strict):
    option: Annotated[str, StringConstraints(min_length=1)]


class OpenChannelReq(_Strict):
    proposed_members: Annotated[list[str], Field(min_length=2)]


class CloseChannelReq(_Strict):
    channel_id: Annotated[str, StringConstraints(min_length=1)]


class RequestCapability(_Strict):
    cap_id: Annotated[str, StringConstraints(min_length=1)]
    justification: str = ""


class DropCapability(_Strict):
    cap_id: Annotated[str, StringConstraints(min_length=1)]


class Yield(_Strict):
    reason: Annotated[str, StringConstraints(max_length=MAX_YIELD_REASON_BYTES)] = ""


class DeclareIntent(_Strict):
    content: Annotated[str, StringConstraints(min_length=1, max_length=MAX_DECLARE_INTENT_BYTES)]


# Tool-call envelope — mirrors the OpenAI-compatible structure vLLM
# emits. ``arguments`` is kept as a dict here; the per-tool schema is
# applied in a second pass (dispatched on ``name``), because a typed
# Union[...] is ambiguous (e.g., DeclareIntent and Speak both accept a
# ``content`` field, so pydantic would coerce to whichever variant
# matched first).
class ToolCall(_Strict):
    name: Literal[
        "speak",
        "vote",
        "open_channel",
        "close_channel",
        "request_capability",
        "drop_capability",
        "yield",
        "declare_intent",
    ]
    arguments: dict


# Mapping from tool name → (proto event_type, pydantic class). The Go
# kernel's schema package uses the event_type string as the
# discriminator on the wire envelope, so we mirror the exact strings.
TOOL_TO_EVENT: dict[str, tuple[str, type[_Strict]]] = {
    "speak":              ("Speak", Speak),
    "vote":               ("Vote", Vote),
    "open_channel":       ("OpenChannelReq", OpenChannelReq),
    "close_channel":      ("CloseChannelReq", CloseChannelReq),
    "request_capability": ("RequestCapability", RequestCapability),
    "drop_capability":    ("DropCapability", DropCapability),
    "yield":              ("Yield", Yield),
    "declare_intent":     ("DeclareIntent", DeclareIntent),
}


def event_type_for_tool(name: str) -> str:
    """Tool name (lowercase) → event_type discriminator on the envelope."""
    return TOOL_TO_EVENT[name][0]
