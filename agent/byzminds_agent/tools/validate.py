"""JSON tool-call → signed EventEnvelope.

Pipeline (per byzminds-template-spec.md §6):

    JSON dict (from vLLM tool-call OR regex extraction)
      → pydantic validation against tools/schemas.py
      → on failure: collapse to Yield(reason="malformed_output");
        raw output recorded on L_ctrl by the kernel
      → on success: deterministic-marshal payload via the generated proto
      → build EventEnvelope (tick + seq supplied by caller)
      → Ed25519-sign canonical(SigningInput)
      → return ready-to-submit envelope plus a structured
        ValidationResult describing what happened (used by the
        malformation-rate metric per dial × α in Experiment 008).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric import ed25519
from pydantic import ValidationError

from byzminds_agent.crypto import sign
from byzminds_agent.proto_gen import events_pb2
from byzminds_agent.tools import schemas


@dataclass
class ValidationResult:
    """Outcome of one JSON tool-call validation.

    ``malformed`` is True when the model produced anything that did not
    pass pydantic; ``original_output`` carries the unparsed text so the
    kernel can log it to L_ctrl (Step 4 will exercise this; in Step 3
    the raw text just rides along in the receipt).
    """

    envelope: events_pb2.EventEnvelope
    malformed: bool
    original_output: str
    reason: str = ""  # set on malformed: e.g., pydantic error message
    event_type: str = ""  # final envelope.event_type


def _payload_bytes_for(name: str, validated: schemas._Strict) -> bytes:
    """Convert a validated pydantic model into the deterministic
    protobuf bytes that the kernel expects as envelope.payload.
    """
    if name == "speak":
        return events_pb2.Speak(
            channel_id=validated.channel_id, content=validated.content
        ).SerializeToString(deterministic=True)
    if name == "vote":
        return events_pb2.Vote(option=validated.option).SerializeToString(deterministic=True)
    if name == "open_channel":
        return events_pb2.OpenChannelReq(
            proposed_members=list(validated.proposed_members)
        ).SerializeToString(deterministic=True)
    if name == "close_channel":
        return events_pb2.CloseChannelReq(channel_id=validated.channel_id).SerializeToString(
            deterministic=True
        )
    if name == "request_capability":
        return events_pb2.RequestCapability(
            cap_id=validated.cap_id, justification=validated.justification
        ).SerializeToString(deterministic=True)
    if name == "drop_capability":
        return events_pb2.DropCapability(cap_id=validated.cap_id).SerializeToString(
            deterministic=True
        )
    if name == "yield":
        return events_pb2.Yield(reason=validated.reason).SerializeToString(deterministic=True)
    if name == "declare_intent":
        return events_pb2.DeclareIntent(content=validated.content).SerializeToString(
            deterministic=True
        )
    raise ValueError(f"unknown tool name {name!r}")


def envelope_from_tool_call(
    tool_call: dict,
    *,
    emitter_pubkey: bytes,
    emitter_priv: ed25519.Ed25519PrivateKey,
    tick: int,
    sequence_per_ledger: int,
    original_output: str = "",
) -> ValidationResult:
    """Validate a tool-call dict and produce a signed envelope.

    The malformed path (any pydantic / parse failure) emits a Yield
    with reason ``"malformed_output"``. The original ``tool_call`` and
    ``original_output`` text are preserved on the result so the kernel
    can record them to L_ctrl as a diagnostic.
    """
    try:
        call = schemas.ToolCall.model_validate(tool_call)
    except ValidationError as e:
        return _yield_for_malformed(
            emitter_pubkey,
            emitter_priv,
            tick,
            sequence_per_ledger,
            reason=str(e).splitlines()[0][:120],
            original_output=original_output or json.dumps(tool_call, default=str),
        )

    name = call.name
    event_type, payload_cls = schemas.TOOL_TO_EVENT[name]
    try:
        validated_args = payload_cls.model_validate(call.arguments)
    except ValidationError as e:
        return _yield_for_malformed(
            emitter_pubkey,
            emitter_priv,
            tick,
            sequence_per_ledger,
            reason=str(e).splitlines()[0][:120],
            original_output=original_output or json.dumps(tool_call, default=str),
        )
    payload = _payload_bytes_for(name, validated_args)
    env = events_pb2.EventEnvelope(
        emitter_pubkey=emitter_pubkey,
        tick=tick,
        sequence_per_ledger=sequence_per_ledger,
        event_type=event_type,
        payload=payload,
    )
    env.signature = sign.sign_envelope(emitter_priv, env)
    return ValidationResult(
        envelope=env,
        malformed=False,
        original_output=original_output,
        reason="",
        event_type=event_type,
    )


def envelope_from_text(
    text: str,
    *,
    emitter_pubkey: bytes,
    emitter_priv: ed25519.Ed25519PrivateKey,
    tick: int,
    sequence_per_ledger: int,
) -> ValidationResult:
    """Same as ``envelope_from_tool_call`` but the input is raw text
    (Route 2 fallback). Tries JSON parse; collapses to Yield on any
    failure.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _yield_for_malformed(
            emitter_pubkey,
            emitter_priv,
            tick,
            sequence_per_ledger,
            reason="not_json",
            original_output=text,
        )
    if not isinstance(obj, dict):
        return _yield_for_malformed(
            emitter_pubkey,
            emitter_priv,
            tick,
            sequence_per_ledger,
            reason="json_not_object",
            original_output=text,
        )
    return envelope_from_tool_call(
        obj,
        emitter_pubkey=emitter_pubkey,
        emitter_priv=emitter_priv,
        tick=tick,
        sequence_per_ledger=sequence_per_ledger,
        original_output=text,
    )


def _yield_for_malformed(
    emitter_pubkey: bytes,
    emitter_priv: ed25519.Ed25519PrivateKey,
    tick: int,
    sequence_per_ledger: int,
    *,
    reason: str,
    original_output: str,
) -> ValidationResult:
    """Build the synthetic Yield envelope that replaces malformed output."""
    payload = events_pb2.Yield(reason="malformed_output").SerializeToString(deterministic=True)
    env = events_pb2.EventEnvelope(
        emitter_pubkey=emitter_pubkey,
        tick=tick,
        sequence_per_ledger=sequence_per_ledger,
        event_type="Yield",
        payload=payload,
    )
    env.signature = sign.sign_envelope(emitter_priv, env)
    return ValidationResult(
        envelope=env,
        malformed=True,
        original_output=original_output,
        reason=reason,
        event_type="Yield",
    )
