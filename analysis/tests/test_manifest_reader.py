"""Unit tests for analysis/metrics/manifest_reader.py."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

# byzminds_agent is at agent/; analysis at analysis/.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "agent"))

from google.protobuf.json_format import MessageToDict  # noqa: E402

from analysis.metrics.manifest_reader import (  # noqa: E402
    Manifest,
    ManifestModelVersionMismatch,
    ManifestModelVersions,
    partition_by_model_versions,
    read,
    require_same_model_versions,
)
from byzminds_agent.proto_gen import (  # noqa: E402
    events_pb2,
    handler_pb2,
    ledger_pb2,
)


# --- Manifest synthesis helpers ----------------------------------------


def _committed(envelope: events_pb2.EventEnvelope, *, ledger_id: int, global_seq: int) -> dict:
    c = ledger_pb2.CommittedEvent()
    c.envelope.CopyFrom(envelope)
    c.ledger_id = ledger_id
    c.global_commit_seq = global_seq
    c.commit_unix_nanos = 1_700_000_000_000_000_000
    c.prev_chain_hash = b"\x00" * 32
    c.chain_hash = b"\x01" * 32
    c.kernel_signature = b"\x02" * 64
    return MessageToDict(c, preserving_proto_field_name=False)


def _spawn_event(
    *, agent_id: str, agent_pubkey: bytes, role: str = "reviewer", theta: tuple = (0.0,) * 6
) -> events_pb2.EventEnvelope:
    spawn = handler_pb2.SpawnAgentRequest(
        agent_id=agent_id,
        agent_pubkey=agent_pubkey,
        role=role,
        theta=list(theta),
    )
    ctrl = events_pb2.HandlerControlEvent(
        handler_rpc_name="SpawnAgent",
        handler_request_bytes=spawn.SerializeToString(),
        effective_tick=0,
    )
    env = events_pb2.EventEnvelope(
        emitter_pubkey=b"\xaa" * 32,  # kernel pubkey
        tick=0,
        sequence_per_ledger=1,
        event_type="Handler_SpawnAgent",
        payload=ctrl.SerializeToString(),
    )
    env.signature = b"\xbb" * 64
    return env


def _vote_event(*, agent_pubkey: bytes, tick: int, option: str) -> events_pb2.EventEnvelope:
    vote = events_pb2.Vote(option=option)
    env = events_pb2.EventEnvelope(
        emitter_pubkey=agent_pubkey,
        tick=tick,
        sequence_per_ledger=tick + 1,
        event_type="Vote",
        payload=vote.SerializeToString(),
    )
    env.signature = b"\xcc" * 64
    return env


def _cog_ind(*, agent_id: str, tick: int, theta: tuple) -> events_pb2.EventEnvelope:
    snap = events_pb2.CogIndSnapshot(agent_id=agent_id, theta=list(theta))
    env = events_pb2.EventEnvelope(
        emitter_pubkey=b"\xaa" * 32,
        tick=tick,
        sequence_per_ledger=tick + 1,
        event_type="CogIndSnapshot",
        payload=snap.SerializeToString(),
    )
    env.signature = b"\xdd" * 64
    return env


def _declare_intent(*, agent_pubkey: bytes, tick: int, content: str) -> events_pb2.EventEnvelope:
    di = events_pb2.DeclareIntent(content=content)
    env = events_pb2.EventEnvelope(
        emitter_pubkey=agent_pubkey,
        tick=tick,
        sequence_per_ledger=1,
        event_type="DeclareIntent",
        payload=di.SerializeToString(),
    )
    env.signature = b"\xee" * 64
    return env


def _malformed(*, agent_id: str, tick: int, raw: bytes, failure: str) -> events_pb2.EventEnvelope:
    ms = events_pb2.MalformedSubmission(
        agent_id=agent_id, tick=tick, raw_output=raw, failure=failure
    )
    env = events_pb2.EventEnvelope(
        emitter_pubkey=b"\xaa" * 32,
        tick=tick,
        sequence_per_ledger=1,
        event_type="MalformedSubmission",
        payload=ms.SerializeToString(),
    )
    env.signature = b"\xee" * 64
    return env


def _write_manifest(path: Path, events: list[dict], *, model_versions: dict | None = None) -> Path:
    body = {
        "schema_version": 1,
        "kernel_version": "test",
        "build_hash": "tests",
        "initial_state": {},
        "events": events,
        "final_chain_hash": "ab" * 32,
    }
    if model_versions:
        body["model_versions"] = model_versions
    with gzip.open(path, "wt") as f:
        json.dump(body, f)
    return path


# --- Tests --------------------------------------------------------------


def test_read_round_trip_with_one_agent(tmp_path: Path):
    a_pub = b"\x11" * 32
    events = [
        _committed(_spawn_event(agent_id="reviewer_01", agent_pubkey=a_pub), ledger_id=5, global_seq=1),
        _committed(_cog_ind(agent_id="reviewer_01", tick=0, theta=(0.0,) * 6), ledger_id=3, global_seq=2),
        _committed(_vote_event(agent_pubkey=a_pub, tick=3, option="approve"), ledger_id=1, global_seq=3),
        _committed(_cog_ind(agent_id="reviewer_01", tick=3, theta=(0.0,) * 6), ledger_id=3, global_seq=4),
        _committed(_declare_intent(agent_pubkey=a_pub, tick=3, content="I approve."), ledger_id=4, global_seq=5),
    ]
    p = _write_manifest(tmp_path / "m.json.gz", events)
    m = read(p)
    assert m.schema_version == 1
    assert "reviewer_01" in m.agents
    t = m.agents["reviewer_01"]
    assert t.role == "reviewer"
    assert t.pubkey_hex == a_pub.hex()
    assert len(t.cog_ind) == 2
    assert [r.tick for r in t.cog_ind] == [0, 3]
    assert len(t.actions) == 1
    assert t.actions[0].event_type == "Vote"
    assert t.actions[0].vote_option == "approve"
    assert len(t.cog_eli) == 1
    assert t.cog_eli[0].content == "I approve."


def test_read_buckets_malformations_to_right_agent(tmp_path: Path):
    a_pub = b"\x11" * 32
    events = [
        _committed(_spawn_event(agent_id="reviewer_01", agent_pubkey=a_pub), ledger_id=5, global_seq=1),
        _committed(_malformed(agent_id="reviewer_01", tick=4, raw=b"{ bogus", failure="parse_error"), ledger_id=5, global_seq=2),
        _committed(_malformed(agent_id="reviewer_01", tick=7, raw=b"<x>", failure="schema_mismatch"), ledger_id=5, global_seq=3),
    ]
    p = _write_manifest(tmp_path / "m.json.gz", events)
    m = read(p)
    t = m.agents["reviewer_01"]
    failures = [mal.failure for mal in t.malformations]
    assert failures == ["parse_error", "schema_mismatch"]


def test_read_model_versions_round_trip(tmp_path: Path):
    mv = {
        "agent_model": "meta-llama/Llama-3.1-8B-Instruct",
        "agent_model_revision_sha": "0e9e39f249a16976918f6564b8830bc894c89659",
        "judge_model": "Qwen/Qwen2.5-14B-Instruct",
        "judge_model_revision_sha": "deadbeef0123456789abcdef0123456789abcdef",
    }
    p = _write_manifest(tmp_path / "m.json.gz", events=[], model_versions=mv)
    m = read(p)
    assert m.model_versions.agent_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert m.model_versions.agent_model_revision_sha.startswith("0e9e39")


def test_aggregation_guard_accepts_identical(tmp_path: Path):
    mv = {"agent_model_revision_sha": "rev_a", "judge_model_revision_sha": "judge_a"}
    a = _write_manifest(tmp_path / "a.json.gz", events=[], model_versions=mv)
    b = _write_manifest(tmp_path / "b.json.gz", events=[], model_versions=mv)
    common = require_same_model_versions([read(a), read(b)])
    assert common.agent_model_revision_sha == "rev_a"


def test_aggregation_guard_rejects_mismatched(tmp_path: Path):
    mv_a = {"agent_model_revision_sha": "rev_a", "judge_model_revision_sha": "judge_a"}
    mv_b = {"agent_model_revision_sha": "rev_b", "judge_model_revision_sha": "judge_a"}
    a = _write_manifest(tmp_path / "a.json.gz", events=[], model_versions=mv_a)
    b = _write_manifest(tmp_path / "b.json.gz", events=[], model_versions=mv_b)
    with pytest.raises(ManifestModelVersionMismatch) as exc:
        require_same_model_versions([read(a), read(b)])
    msg = str(exc.value)
    assert "Found 2 distinct revision groups" in msg
    assert "Group A" in msg
    assert "Group B" in msg
    assert "rev_a" in msg
    assert "rev_b" in msg
    assert "partition_by_model_versions" in msg


def test_aggregation_guard_treats_unpopulated_as_distinct_from_populated(tmp_path: Path):
    a = _write_manifest(tmp_path / "a.json.gz", events=[])  # unpopulated
    b = _write_manifest(
        tmp_path / "b.json.gz",
        events=[],
        model_versions={"agent_model_revision_sha": "rev"},
    )
    with pytest.raises(ManifestModelVersionMismatch):
        require_same_model_versions([read(a), read(b)])


def test_partition_by_model_versions(tmp_path: Path):
    mv_a = {"agent_model_revision_sha": "rev_a"}
    mv_b = {"agent_model_revision_sha": "rev_b"}
    a = read(_write_manifest(tmp_path / "a.json.gz", events=[], model_versions=mv_a))
    b1 = read(_write_manifest(tmp_path / "b1.json.gz", events=[], model_versions=mv_b))
    b2 = read(_write_manifest(tmp_path / "b2.json.gz", events=[], model_versions=mv_b))
    parts = partition_by_model_versions([a, b1, b2])
    assert len(parts) == 2
    assert len(parts[a.model_versions]) == 1
    assert len(parts[b1.model_versions]) == 2


def test_unpopulated_model_versions_round_trip_is_clean(tmp_path: Path):
    """Step 1–4 manifests omit model_versions entirely; ManifestModelVersions
    instance should report ``is_unpopulated`` and aggregation across
    multiple such manifests should succeed."""
    a = read(_write_manifest(tmp_path / "a.json.gz", events=[]))
    b = read(_write_manifest(tmp_path / "b.json.gz", events=[]))
    assert a.model_versions.is_unpopulated()
    common = require_same_model_versions([a, b])
    assert common.is_unpopulated()


def test_empty_manifest_list_returns_default(tmp_path: Path):
    out = require_same_model_versions([])
    assert isinstance(out, ManifestModelVersions)
    assert out.is_unpopulated()
