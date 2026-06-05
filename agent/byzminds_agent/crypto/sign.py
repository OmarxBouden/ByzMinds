"""Ed25519 signing on the Python side.

Goal: produce signatures byte-identical to the Go kernel's. The Go side
defines (in events.proto + kernel/internal/crypto/sign.go):

    SigningInput { emitter_pubkey, tick, sequence_per_ledger,
                   event_type, payload }

and computes:

    canonical_bytes = proto.MarshalOptions{Deterministic: true}.Marshal(SigningInput)
    signature       = Ed25519(emitter_priv, canonical_bytes)

We mirror exactly: same proto message, same deterministic serialization
(``SerializeToString(deterministic=True)`` on the Python generated
``SigningInput``), same Ed25519 over the same bytes. Cross-language
agreement is tested via the kernel-binary smoke test in tests/.
"""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from byzminds_agent.proto_gen import events_pb2

PUBLIC_KEY_SIZE = 32
PRIVATE_KEY_SIZE_GO = 64  # ed25519.PrivateKey in Go = seed (32 B) || public (32 B)
SIGNATURE_SIZE = 64


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate an Ed25519 keypair.

    Returns
    -------
    (public_key_bytes_32, private_key_bytes_64)
        ``private_key_bytes_64`` is the Go-compatible form: seed (32 B)
        followed by the public-key bytes (32 B). This matches what
        ``ed25519.PrivateKey`` stores in the Go standard library and is
        the wire format ``--kernel-priv-hex`` expects on the Go side.
    """
    priv = ed25519.Ed25519PrivateKey.generate()
    pub_obj = priv.public_key()
    seed = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = pub_obj.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return pub, seed + pub


def load_priv(priv_bytes: bytes) -> ed25519.Ed25519PrivateKey:
    """Load an Ed25519 private key from Go's 64-byte form (seed||public).

    Accepts the raw 32-byte seed as well, for callers that round-trip
    through other libraries.
    """
    if len(priv_bytes) == PRIVATE_KEY_SIZE_GO:
        seed = priv_bytes[:32]
    elif len(priv_bytes) == 32:
        seed = priv_bytes
    else:
        raise ValueError(
            f"crypto.sign: private key must be 32 or {PRIVATE_KEY_SIZE_GO} bytes, "
            f"got {len(priv_bytes)}"
        )
    return ed25519.Ed25519PrivateKey.from_private_bytes(seed)


def load_pub(pub_bytes: bytes) -> ed25519.Ed25519PublicKey:
    """Load an Ed25519 public key from raw 32 bytes."""
    if len(pub_bytes) != PUBLIC_KEY_SIZE:
        raise ValueError(
            f"crypto.sign: public key must be {PUBLIC_KEY_SIZE} bytes, got {len(pub_bytes)}"
        )
    return ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)


def canonical_bytes(msg) -> bytes:
    """Deterministic protobuf serialization. Mirrors Go's
    ``proto.MarshalOptions{Deterministic: true}.Marshal``.

    The argument is a generated ``google.protobuf.message.Message``;
    ``SerializeToString(deterministic=True)`` exists in protobuf>=3.20
    and fixes map ordering. Our SigningInput contains no maps; the flag
    is belt-and-braces.
    """
    return msg.SerializeToString(deterministic=True)


def signing_input_for_envelope(env: events_pb2.EventEnvelope) -> events_pb2.SigningInput:
    """Build the SigningInput proto from envelope fields. Mirrors the
    Go helper ``crypto.SigningInputForEnvelope``.
    """
    return events_pb2.SigningInput(
        emitter_pubkey=env.emitter_pubkey,
        tick=env.tick,
        sequence_per_ledger=env.sequence_per_ledger,
        event_type=env.event_type,
        payload=env.payload,
    )


def sign_envelope(priv: ed25519.Ed25519PrivateKey, env: events_pb2.EventEnvelope) -> bytes:
    """Compute the Ed25519 signature for env and return the 64-byte sig.

    The caller assigns the returned bytes to ``env.signature``.
    """
    msg_bytes = canonical_bytes(signing_input_for_envelope(env))
    return priv.sign(msg_bytes)


def verify_envelope(env: events_pb2.EventEnvelope) -> None:
    """Verify env.signature against env.emitter_pubkey. Raises
    ``cryptography.exceptions.InvalidSignature`` on failure.
    """
    pub = load_pub(env.emitter_pubkey)
    msg_bytes = canonical_bytes(signing_input_for_envelope(env))
    pub.verify(env.signature, msg_bytes)


def sign_bytes(priv: ed25519.Ed25519PrivateKey, msg: bytes) -> bytes:
    """Sign raw bytes (used by callers who already have canonical form,
    e.g., the kernel's chain_hash signer)."""
    return priv.sign(msg)


def verify_bytes(pub: ed25519.Ed25519PublicKey, msg: bytes, sig: bytes) -> None:
    """Verify a signature over raw bytes."""
    pub.verify(sig, msg)


def view_request_signing_bytes(reader_pubkey: bytes, from_tick: int) -> bytes:
    """Canonical preimage for Subscribe / GetView signatures.

    Matches Go ``api.ViewRequestSigningBytes`` exactly:
    ``reader_pubkey || from_tick (big-endian uint64)``.
    """
    if len(reader_pubkey) != PUBLIC_KEY_SIZE:
        raise ValueError(f"reader_pubkey must be {PUBLIC_KEY_SIZE} bytes")
    return bytes(reader_pubkey) + int(from_tick).to_bytes(8, "big")
