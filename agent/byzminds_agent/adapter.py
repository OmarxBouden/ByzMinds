"""gRPC adapter — Subscribe stream + envelope sign + SubmitEvent.

Wraps the generated kernelv1.KernelStub behind a tight surface the
runtime consumes. Per-(emitter, ledger) sequence numbers are tracked
locally; the kernel rejects a wrong-seq envelope with a clear receipt,
and the adapter reconciles by bumping its counter on success and
re-trying the next emission with a fresh seq on a rejection (rare, only
expected when kernel-synthesized events bump the agent's per-emitter
counter on the L_pub or L_ctrl ledger).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import grpc
from cryptography.hazmat.primitives.asymmetric import ed25519

from byzminds_agent.crypto import sign
from byzminds_agent.proto_gen import events_pb2, kernel_pb2, kernel_pb2_grpc, view_pb2

log = logging.getLogger(__name__)

# Ledger destination keys used by the local per-emitter seq counter.
# These mirror the routing schema package's destination resolution
# (kernel/internal/schema/validate.go).
DEST_L_PUB = ("L_PUB", "")
DEST_L_COG_ELI = ("L_COG_ELI", "")
DEST_L_CTRL = ("L_CTRL", "")


def destination_for(event_type: str, payload: events_pb2.EventEnvelope | None = None) -> tuple[str, str]:
    """Return (ledger, channel_id) for the per-emitter seq counter.

    For Speak we need to inspect the payload's channel_id; for the
    other event types the destination is fixed by event_type.
    """
    if event_type == "Speak":
        # Caller should pass payload bytes already parsed; we accept
        # an EventEnvelope and re-decode here. To keep the dependency
        # one-way, callers normally pass the channel_id directly via
        # ``KernelAdapter.advance_seq``.
        return DEST_L_PUB  # Caller should override via channel_id
    if event_type in {"Vote", "Yield", "Yield_Kernel_Synthesized"}:
        return DEST_L_PUB
    if event_type == "DeclareIntent":
        return DEST_L_COG_ELI
    # Control intents (OpenChannelReq, CloseChannelReq, Request/Drop
    # capability) all route to L_ctrl.
    return DEST_L_CTRL


@dataclass
class KernelAdapter:
    """Thin sync wrapper around kernelv1.KernelStub.

    Use one adapter per agent process. Step 3 runtime calls
    ``subscribe`` once and then ``submit`` per tick.
    """

    kernel_addr: str
    agent_id: str
    agent_pubkey: bytes
    agent_priv: ed25519.Ed25519PrivateKey
    timeout_seconds: float = 30.0

    _channel: grpc.Channel | None = field(default=None, init=False, repr=False)
    _stub: kernel_pb2_grpc.KernelStub | None = field(default=None, init=False, repr=False)
    # Per-(emitter, dest) counter. Key: (ledger_string, channel_id).
    _seq_by_dest: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._channel = grpc.insecure_channel(self.kernel_addr)
        self._stub = kernel_pb2_grpc.KernelStub(self._channel)

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self) -> "KernelAdapter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---- Subscribe ----------------------------------------------------

    def subscribe(self, from_tick: int = 0) -> Iterator[view_pb2.View]:
        """Open a Subscribe stream and yield View messages until the
        server closes or the caller stops iterating.
        """
        sig = sign.sign_bytes(self.agent_priv, sign.view_request_signing_bytes(self.agent_pubkey, from_tick))
        req = kernel_pb2.SubscribeRequest(
            agent_pubkey=self.agent_pubkey,
            from_tick=from_tick,
            signature=sig,
        )
        return self._stub.Subscribe(req)

    # ---- SubmitEvent --------------------------------------------------

    def next_seq(self, dest: tuple[str, str]) -> int:
        """Allocate the next per-(emitter, dest) sequence number for
        the agent. Locally maintained; the kernel verifies on commit.
        """
        cur = self._seq_by_dest.get(dest, 0)
        nxt = cur + 1
        self._seq_by_dest[dest] = nxt
        return nxt

    def rollback_seq(self, dest: tuple[str, str]) -> None:
        """Decrement the local counter (used after a rejection so the
        next allocation reuses the slot we never managed to commit).
        Safe with no allocation below 0.
        """
        cur = self._seq_by_dest.get(dest, 0)
        if cur > 0:
            self._seq_by_dest[dest] = cur - 1

    def submit(self, envelope: events_pb2.EventEnvelope) -> kernel_pb2.CommitReceipt:
        """Submit a fully-signed envelope. Returns the CommitReceipt."""
        return self._stub.SubmitEvent(envelope, timeout=self.timeout_seconds)
