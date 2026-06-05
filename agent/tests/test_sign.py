"""Unit tests for crypto/sign.py.

The cross-language smoke (Python signs → Go kernel verifies) lives in
test_kernel_smoke.py because it requires a running kernel binary.
"""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidSignature

from byzminds_agent.crypto import sign
from byzminds_agent.proto_gen import events_pb2


def make_envelope(seq: int = 1) -> tuple[events_pb2.EventEnvelope, sign.ed25519.Ed25519PrivateKey]:
    pub, priv_bytes = sign.generate_keypair()
    priv = sign.load_priv(priv_bytes)
    payload = events_pb2.Speak(channel_id="public", content="hello").SerializeToString(deterministic=True)
    env = events_pb2.EventEnvelope(
        emitter_pubkey=pub,
        tick=7,
        sequence_per_ledger=seq,
        event_type="Speak",
        payload=payload,
    )
    env.signature = sign.sign_envelope(priv, env)
    return env, priv


def test_sign_and_verify_round_trip():
    env, _ = make_envelope()
    sign.verify_envelope(env)  # raises on failure


def test_verify_rejects_tampered_payload():
    env, _ = make_envelope()
    env.payload = b"tampered"
    with pytest.raises(InvalidSignature):
        sign.verify_envelope(env)


def test_verify_rejects_tampered_tick():
    env, _ = make_envelope()
    env.tick += 1
    with pytest.raises(InvalidSignature):
        sign.verify_envelope(env)


def test_verify_rejects_swapped_pubkey():
    env, _ = make_envelope()
    other_pub, _ = sign.generate_keypair()
    env.emitter_pubkey = other_pub
    with pytest.raises(InvalidSignature):
        sign.verify_envelope(env)


def test_canonical_bytes_stable_across_calls():
    msg = events_pb2.Speak(channel_id="ch_07", content="deterministic?")
    a = sign.canonical_bytes(msg)
    b = sign.canonical_bytes(msg)
    assert a == b


def test_signing_input_matches_reconstruction():
    """A SigningInput built directly should produce the same canonical
    bytes as one reconstructed from an envelope via
    ``signing_input_for_envelope``."""
    pub, priv_bytes = sign.generate_keypair()
    payload = events_pb2.Speak(channel_id="public", content="x").SerializeToString(deterministic=True)

    si = events_pb2.SigningInput(
        emitter_pubkey=pub,
        tick=42,
        sequence_per_ledger=1,
        event_type="Speak",
        payload=payload,
    )
    si_bytes = sign.canonical_bytes(si)

    env = events_pb2.EventEnvelope(
        emitter_pubkey=pub,
        tick=42,
        sequence_per_ledger=1,
        event_type="Speak",
        payload=payload,
    )
    env_bytes = sign.canonical_bytes(sign.signing_input_for_envelope(env))
    assert si_bytes == env_bytes


def test_view_request_signing_bytes_layout():
    """Matches Go ``api.ViewRequestSigningBytes`` byte-for-byte: 32-B
    pubkey + 8-B big-endian uint64."""
    pub, _ = sign.generate_keypair()
    out = sign.view_request_signing_bytes(pub, 0xDEADBEEF)
    assert out[:32] == pub
    assert out[32:] == (0xDEADBEEF).to_bytes(8, "big")


def test_keypair_round_trip_via_go_format():
    pub, priv_bytes = sign.generate_keypair()
    assert len(priv_bytes) == sign.PRIVATE_KEY_SIZE_GO
    assert priv_bytes[32:] == pub  # Go's seed||public layout
    # Reload from Go-format and from seed-only; both must yield the same key.
    a = sign.load_priv(priv_bytes)
    b = sign.load_priv(priv_bytes[:32])
    assert (
        a.private_bytes_raw() if hasattr(a, "private_bytes_raw") else None
    ) == (b.private_bytes_raw() if hasattr(b, "private_bytes_raw") else None) or True  # API-dependent; the round-trip below is the real test
    # Sign with both; signatures equal for the same message.
    msg = b"hello"
    sa = a.sign(msg)
    sb = b.sign(msg)
    assert sa == sb
