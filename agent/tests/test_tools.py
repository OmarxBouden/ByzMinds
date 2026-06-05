"""Unit tests for tools/schemas.py + tools/validate.py."""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidSignature

from byzminds_agent.crypto import sign
from byzminds_agent.proto_gen import events_pb2
from byzminds_agent.tools import schemas, validate


@pytest.fixture()
def keypair():
    pub, priv_bytes = sign.generate_keypair()
    return pub, sign.load_priv(priv_bytes)


def test_speak_envelope_is_signed_and_well_formed(keypair):
    pub, priv = keypair
    r = validate.envelope_from_tool_call(
        {"name": "speak", "arguments": {"channel_id": "public", "content": "hi"}},
        emitter_pubkey=pub,
        emitter_priv=priv,
        tick=3,
        sequence_per_ledger=1,
    )
    assert not r.malformed
    assert r.event_type == "Speak"
    assert r.envelope.event_type == "Speak"
    assert r.envelope.tick == 3
    assert r.envelope.sequence_per_ledger == 1
    sign.verify_envelope(r.envelope)
    decoded = events_pb2.Speak()
    decoded.ParseFromString(r.envelope.payload)
    assert decoded.channel_id == "public"
    assert decoded.content == "hi"


def test_all_eight_tool_names_round_trip(keypair):
    pub, priv = keypair
    cases = {
        "speak": {"channel_id": "public", "content": "hi"},
        "vote": {"option": "approve"},
        "open_channel": {"proposed_members": ["a", "b"]},
        "close_channel": {"channel_id": "ch_07"},
        "request_capability": {"cap_id": "search", "justification": "needed"},
        "drop_capability": {"cap_id": "search"},
        "yield": {"reason": "no_op"},
        "declare_intent": {"content": "I voted yes because X"},
    }
    for name, args in cases.items():
        r = validate.envelope_from_tool_call(
            {"name": name, "arguments": args},
            emitter_pubkey=pub,
            emitter_priv=priv,
            tick=1,
            sequence_per_ledger=1,
        )
        assert not r.malformed, f"{name}: {r.reason}"
        assert r.event_type == schemas.event_type_for_tool(name)
        sign.verify_envelope(r.envelope)


def test_unknown_tool_collapses_to_yield(keypair):
    pub, priv = keypair
    r = validate.envelope_from_tool_call(
        {"name": "obliterate", "arguments": {"target": "everyone"}},
        emitter_pubkey=pub,
        emitter_priv=priv,
        tick=1,
        sequence_per_ledger=1,
    )
    assert r.malformed
    assert r.event_type == "Yield"
    sign.verify_envelope(r.envelope)
    decoded = events_pb2.Yield()
    decoded.ParseFromString(r.envelope.payload)
    assert decoded.reason == "malformed_output"


def test_missing_required_field_collapses_to_yield(keypair):
    pub, priv = keypair
    r = validate.envelope_from_tool_call(
        {"name": "speak", "arguments": {"content": "missing channel"}},
        emitter_pubkey=pub,
        emitter_priv=priv,
        tick=1,
        sequence_per_ledger=1,
    )
    assert r.malformed
    assert r.event_type == "Yield"


def test_extra_field_rejected(keypair):
    """Strict schema: a hallucinated argument is a hard error, not silently dropped."""
    pub, priv = keypair
    r = validate.envelope_from_tool_call(
        {"name": "speak", "arguments": {"channel_id": "public", "content": "hi", "bonus": True}},
        emitter_pubkey=pub,
        emitter_priv=priv,
        tick=1,
        sequence_per_ledger=1,
    )
    assert r.malformed


def test_envelope_from_text_parses_json(keypair):
    pub, priv = keypair
    text = '{"name":"yield","arguments":{"reason":"no_thought"}}'
    r = validate.envelope_from_text(
        text,
        emitter_pubkey=pub,
        emitter_priv=priv,
        tick=2,
        sequence_per_ledger=1,
    )
    assert not r.malformed
    assert r.event_type == "Yield"


def test_envelope_from_text_yields_on_garbage(keypair):
    pub, priv = keypair
    for bad in ["not json at all", "[]", "42", "{}"]:
        r = validate.envelope_from_text(
            bad, emitter_pubkey=pub, emitter_priv=priv, tick=1, sequence_per_ledger=1
        )
        assert r.malformed, f"{bad!r} should collapse to Yield"


def test_signature_actually_verifies_under_emitter(keypair):
    pub, priv = keypair
    r = validate.envelope_from_tool_call(
        {"name": "yield", "arguments": {"reason": ""}},
        emitter_pubkey=pub,
        emitter_priv=priv,
        tick=1,
        sequence_per_ledger=1,
    )
    sign.verify_envelope(r.envelope)
    # And tampering breaks verify.
    r.envelope.tick = 99
    with pytest.raises(InvalidSignature):
        sign.verify_envelope(r.envelope)


def test_speak_content_length_bound():
    """Content > 4KB rejected (mirrors kernel/internal/schema)."""
    too_long = "a" * (schemas.MAX_SPEAK_CONTENT_BYTES + 1)
    with pytest.raises(Exception):
        schemas.Speak(channel_id="public", content=too_long)
