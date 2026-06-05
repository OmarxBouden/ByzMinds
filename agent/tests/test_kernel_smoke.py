"""End-to-end smoke per DoD #2.

Spins up an in-process Python-side scenario by talking to a running
Go kernel test binary that loads a one-agent scenario and waits for
the agent to attach. We then have the Python agent runtime call
Subscribe + SubmitEvent and assert the kernel commits the envelope
(receipt.committed=True, chain_hash non-empty).

The test compiles a small Go helper (``byzminds-test-kernel-1agent``)
once per pytest invocation and reuses it.

Mark this test ``slow`` so a default ``pytest agent/tests`` run skips
it if the Go toolchain is unavailable — the unit tests in
test_sign.py / test_tools.py / test_cache.py exercise the Python
side standalone.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import grpc
import pytest

from byzminds_agent.adapter import KernelAdapter, destination_for
from byzminds_agent.crypto import sign
from byzminds_agent.tools import validate

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "kernel" / "cmd" / "byzminds-test-kernel-1agent"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _have_go() -> bool:
    return shutil.which("go") is not None


def _wait_port(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"kernel did not start listening on {host}:{port} within {timeout}s")


@pytest.mark.skipif(not _have_go(), reason="Go toolchain not available")
def test_subscribe_and_submit_round_trip(tmp_path: Path) -> None:
    # 1. Generate agent keys, write to a file the runtime will load.
    pub, priv_bytes = sign.generate_keypair()
    keypair_path = tmp_path / "agent.key"
    keypair_path.write_bytes(priv_bytes)

    # 2. Spawn the test kernel helper. It loads a 1-agent scenario named
    # "reviewer_01" using the supplied --agent-pubkey-hex, runs the
    # scheduler, and accepts the Python agent's Subscribe + SubmitEvent.
    port = _free_port()
    addr = f"127.0.0.1:{port}"
    env = os.environ.copy()
    # Use a prebuilt binary if available (much faster than `go run`).
    binary = Path("/tmp/byzminds-test-kernel-1agent")
    if binary.exists():
        cmd = [
            str(binary),
            "--addr", addr,
            "--agent-pubkey-hex", pub.hex(),
        ]
        cwd = REPO / "kernel"
    else:
        cmd = [
            "go", "run", "./cmd/byzminds-test-kernel-1agent",
            "--addr", addr,
            "--agent-pubkey-hex", pub.hex(),
        ]
        cwd = REPO / "kernel"
    helper_log = tmp_path / "kernel.log"
    log_f = helper_log.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_port("127.0.0.1", port, timeout=30.0)
        print(f"[smoke] kernel listening on {addr}", file=sys.stderr, flush=True)

        priv = sign.load_priv(priv_bytes)
        with KernelAdapter(
            kernel_addr=addr,
            agent_id="reviewer_01",
            agent_pubkey=pub,
            agent_priv=priv,
        ) as adapter:
            stream = adapter.subscribe(from_tick=0)
            print("[smoke] subscribe stream opened", file=sys.stderr, flush=True)
            committed = 0
            for view in stream:
                print(f"[smoke] got view tick={view.tick}", file=sys.stderr, flush=True)
                dest = destination_for("Yield")
                seq = adapter.next_seq(dest)
                r = validate.envelope_from_tool_call(
                    {"name": "yield", "arguments": {"reason": "hardcoded_smoke"}},
                    emitter_pubkey=pub,
                    emitter_priv=priv,
                    tick=view.tick,
                    sequence_per_ledger=seq,
                )
                receipt = adapter.submit(r.envelope)
                print(
                    f"[smoke] submit returned committed={receipt.committed} "
                    f"reason={receipt.rejection_reason}",
                    file=sys.stderr,
                    flush=True,
                )
                assert receipt.committed, f"reject: {receipt.rejection_reason}"
                assert len(receipt.chain_hash) == 32
                committed += 1
                if committed >= 2:
                    break
        assert committed >= 2
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_f.close()
        try:
            print("--- kernel helper log ---", file=sys.stderr)
            print(helper_log.read_text(), file=sys.stderr)
        except Exception:
            pass
