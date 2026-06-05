"""Manifest → per-agent traces.

A Step 5 metric run takes a list of manifests (one per scenario run,
written by ``byzminds-run`` and stored under
``runs/<hash>/manifest.json.gz``) and projects them into per-agent
trace tuples the divergence / surface-honesty / BBI functions consume.

The reader surfaces the manifest's ``model_versions`` block so the
metrics pipeline can refuse to aggregate across mismatched model
revisions (Step 5 brief: "mixed-revision aggregation is forbidden").
The error format for that failure is research-facing and documented
in PAPER_NOTES.md.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# byzminds_agent's generated proto bindings live next to this package;
# extend sys.path so analysis tools work even when callers haven't
# pip-installed the agent package.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "agent") not in sys.path:
    sys.path.insert(0, str(_REPO / "agent"))

from byzminds_agent.proto_gen import (  # noqa: E402
    events_pb2,
    handler_pb2,
    ledger_pb2,
)
from google.protobuf.json_format import Parse  # noqa: E402

__all__ = [
    "ActionRecord",
    "CogIndRecord",
    "CogEliRecord",
    "MalformedRecord",
    "TruncationRecord",
    "TickTimeoutRecord",
    "AgentTrace",
    "ManifestModelVersions",
    "Manifest",
    "ManifestModelVersionMismatch",
    "read",
    "require_same_model_versions",
    "partition_by_model_versions",
]


# --- Per-agent record types --------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """One action event the agent emitted on L_pub.

    Fields are projected from the EventEnvelope + decoded payload:
      * tick                       — env.tick
      * global_commit_seq          — committed.global_commit_seq
      * event_type                 — env.event_type (Speak / Vote / Yield /
                                     Yield_Kernel_Synthesized)
      * vote_option                — set iff event_type == "Vote"
      * speak_channel / speak_content — set iff event_type == "Speak"
      * yield_reason               — set iff event_type starts "Yield"
    """

    tick: int
    global_commit_seq: int
    event_type: str
    vote_option: str = ""
    speak_channel: str = ""
    speak_content: str = ""
    yield_reason: str = ""


@dataclass(frozen=True)
class CogIndRecord:
    """One L_cog_ind snapshot for the agent at one tick.

    The kernel writes one CogIndSnapshot per live agent per tick (per
    Step 2 scheduler phase 4), so the agent's cog_ind list length is
    exactly the agent's live-tick count.
    """

    tick: int
    theta: tuple[float, ...]  # length-6 (six dials per state summary)


@dataclass(frozen=True)
class CogEliRecord:
    """One DeclareIntent on L_cog_eli.

    Sparse: only present at elicit-tick boundaries (per scenario's
    K_elicit, default K=3 per Step 4 brief). Content text is the
    agent's natural-language reasoning; the metric pipeline maps it
    to a BinaryDistribution via
    ``divergence.project_intent_to_binary``.
    """

    tick: int
    content: str


@dataclass(frozen=True)
class MalformedRecord:
    """One MalformedSubmission L_ctrl record.

    Step 4 added these — the Python adapter collapses unparseable
    forward-pass outputs to Yield(reason="malformed_output") and the
    kernel records the original raw bytes + failure class to L_ctrl.
    Step 5's BBI drops these ticks from the index by default.
    """

    tick: int
    raw_output: bytes
    failure: str  # parse_error | schema_mismatch | inadmissible | token_limit | elicit_non_compliance | elicit_timeout | other


@dataclass(frozen=True)
class TruncationRecord:
    """One ContextTruncation L_ctrl record (Step 4)."""

    tick: int
    channel_id: str
    dropped_count: int
    kept_count: int


@dataclass(frozen=True)
class TickTimeoutRecord:
    """One TickTimeoutIncident L_ctrl record (Step 2)."""

    tick: int
    budget_nanos: int


@dataclass(frozen=True)
class AgentTrace:
    """All records for one agent across one scenario run."""

    agent_id: str
    pubkey_hex: str
    role: str
    spawn_tick: int
    initial_theta: tuple[float, ...]
    actions: list[ActionRecord] = field(default_factory=list)
    cog_ind: list[CogIndRecord] = field(default_factory=list)
    cog_eli: list[CogEliRecord] = field(default_factory=list)
    malformations: list[MalformedRecord] = field(default_factory=list)
    truncations: list[TruncationRecord] = field(default_factory=list)
    timeouts: list[TickTimeoutRecord] = field(default_factory=list)


# --- Manifest-level types ----------------------------------------------


@dataclass(frozen=True)
class ManifestModelVersions:
    """Mirror of kernel/internal/manifest.ModelVersions.

    Used as the aggregation guard key. Equality is structural over
    every populated field.
    """

    agent_model: str = ""
    agent_model_revision_sha: str = ""
    agent_model_dtype: str = ""
    vllm_build_sha: str = ""
    judge_model: str = ""
    judge_model_revision_sha: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ManifestModelVersions:
        d = d or {}
        return cls(
            agent_model=d.get("agent_model", "") or "",
            agent_model_revision_sha=d.get("agent_model_revision_sha", "") or "",
            agent_model_dtype=d.get("agent_model_dtype", "") or "",
            vllm_build_sha=d.get("vllm_build_sha", "") or "",
            judge_model=d.get("judge_model", "") or "",
            judge_model_revision_sha=d.get("judge_model_revision_sha", "") or "",
        )

    def is_unpopulated(self) -> bool:
        """True iff every field is empty (Step 1–4 manifests, where
        no LLM is in the loop)."""
        return (
            not self.agent_model
            and not self.agent_model_revision_sha
            and not self.vllm_build_sha
            and not self.judge_model
            and not self.judge_model_revision_sha
        )

    def render_block(self, indent: str = "  ") -> str:
        """Pretty-print this ModelVersions for error messages and
        PAPER_NOTES-style logs."""
        lines = [
            f"{indent}agent_model:               {self.agent_model or '(unset)'}",
            f"{indent}agent_model_revision_sha:  {self.agent_model_revision_sha or '(unset)'}",
            f"{indent}agent_model_dtype:         {self.agent_model_dtype or '(unset)'}",
            f"{indent}vllm_build_sha:            {self.vllm_build_sha or '(unset)'}",
            f"{indent}judge_model:               {self.judge_model or '(unset)'}",
            f"{indent}judge_model_revision_sha:  {self.judge_model_revision_sha or '(unset)'}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class Manifest:
    """Parsed manifest with per-agent traces.

    ``path`` is the on-disk path the manifest was read from (or empty
    when synthesized in-memory for tests). ``final_chain_hash`` is the
    hex-encoded chain head from the Step 1 manifest writer.
    """

    schema_version: int
    kernel_version: str
    build_hash: str
    final_chain_hash: str
    model_versions: ManifestModelVersions
    agents: dict[str, AgentTrace]
    path: str = ""

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError(f"manifest_reader: invalid schema_version {self.schema_version}")


# --- Errors ------------------------------------------------------------


class ManifestModelVersionMismatch(Exception):
    """Raised when aggregating manifests with disagreeing model_versions.

    The error message format is research-facing (a reviewer debugging
    a cross-revision failure will hit this first) and is documented in
    PAPER_NOTES.md §"Step 5 — manifest model-version mismatch error format".

    Layout:

        ManifestModelVersionMismatch: cannot aggregate across mismatched model_versions.

        Found N distinct revision groups across M manifests:

        Group A (k manifests):
          agent_model:               <model>
          agent_model_revision_sha:  <sha>
          ...
          manifests:
            - runs/abc/manifest.json.gz
            - runs/def/manifest.json.gz

        Group B (j manifests):
          ...

        To proceed:
          * If both revisions should be reported separately, use
            partition_by_model_versions(manifests) and aggregate
            per-partition.
          * If the runs should be re-done at a single pinned revision,
            re-run byzminds-run with the appropriate flags so the
            written manifest's model_versions matches.
    """


# --- Reading -----------------------------------------------------------


def read(path: str | Path) -> Manifest:
    """Parse a gzipped JSON manifest from disk.

    Walks the events list, decodes each CommittedEvent via protojson,
    and projects per-agent records. Returns a Manifest carrying the
    per-agent traces and the model_versions block (empty for Step 1–4
    manifests).
    """
    p = Path(path)
    with gzip.open(p, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    return _from_json(raw, path=str(p))


def _from_json(raw: dict, *, path: str = "") -> Manifest:
    schema_version = int(raw.get("schema_version", 0))
    kernel_version = str(raw.get("kernel_version", ""))
    build_hash = str(raw.get("build_hash", ""))
    final_chain_hash = str(raw.get("final_chain_hash", ""))
    model_versions = ManifestModelVersions.from_dict(raw.get("model_versions"))

    # First pass: build pubkey → AgentTrace by walking Handler_SpawnAgent
    # events on L_ctrl. Each Handler_* envelope payload is a
    # HandlerControlEvent whose handler_request_bytes contains the
    # original RPC request's canonical bytes (with HandlerAuth cleared).
    agents_by_pubkey: dict[str, AgentTrace] = {}
    agents_by_id: dict[str, AgentTrace] = {}

    parsed_events = list(_iter_committed_events(raw.get("events", [])))

    for committed in parsed_events:
        env = committed.envelope
        if env.event_type != "Handler_SpawnAgent":
            continue
        ctrl = events_pb2.HandlerControlEvent()
        ctrl.ParseFromString(env.payload)
        spawn = handler_pb2.SpawnAgentRequest()
        spawn.ParseFromString(ctrl.handler_request_bytes)
        pubkey_hex = spawn.agent_pubkey.hex()
        trace = AgentTrace(
            agent_id=spawn.agent_id,
            pubkey_hex=pubkey_hex,
            role=spawn.role,
            spawn_tick=env.tick,
            initial_theta=tuple(spawn.theta) if len(spawn.theta) > 0 else tuple([0.0] * 6),
        )
        agents_by_pubkey[pubkey_hex] = trace
        agents_by_id[spawn.agent_id] = trace

    # Second pass: bucket every other event into the right agent's
    # trace. We mutate AgentTrace fields in-place (dataclass is frozen
    # but the lists are mutable references).
    for committed in parsed_events:
        env = committed.envelope
        et = env.event_type
        if et.startswith("Handler_"):
            continue
        if et == "CogIndSnapshot":
            snap = events_pb2.CogIndSnapshot()
            snap.ParseFromString(env.payload)
            trace = agents_by_id.get(snap.agent_id)
            if trace is not None:
                trace.cog_ind.append(
                    CogIndRecord(tick=env.tick, theta=tuple(snap.theta))
                )
            continue
        if et == "DeclareIntent":
            decl = events_pb2.DeclareIntent()
            decl.ParseFromString(env.payload)
            trace = agents_by_pubkey.get(env.emitter_pubkey.hex())
            if trace is not None:
                trace.cog_eli.append(CogEliRecord(tick=env.tick, content=decl.content))
            continue
        if et == "MalformedSubmission":
            ms = events_pb2.MalformedSubmission()
            ms.ParseFromString(env.payload)
            trace = agents_by_id.get(ms.agent_id)
            if trace is not None:
                trace.malformations.append(
                    MalformedRecord(
                        tick=ms.tick,
                        raw_output=bytes(ms.raw_output),
                        failure=ms.failure,
                    )
                )
            continue
        if et == "ContextTruncation":
            ct = events_pb2.ContextTruncation()
            ct.ParseFromString(env.payload)
            trace = agents_by_id.get(ct.agent_id)
            if trace is not None:
                trace.truncations.append(
                    TruncationRecord(
                        tick=ct.tick,
                        channel_id=ct.channel_id,
                        dropped_count=ct.dropped_count,
                        kept_count=ct.kept_count,
                    )
                )
            continue
        if et == "TickTimeoutIncident":
            tt = events_pb2.TickTimeoutIncident()
            tt.ParseFromString(env.payload)
            trace = agents_by_id.get(tt.agent_id)
            if trace is not None:
                trace.timeouts.append(
                    TickTimeoutRecord(tick=tt.tick, budget_nanos=tt.budget_nanos)
                )
            continue
        if et in ("ElicitationRequest",):
            # Kernel-emitted control event; the agent inferred from it
            # via View.elicit_request and may have responded with a
            # DeclareIntent (handled above). Skip here.
            continue
        # All remaining types are agent-emitted action events on L_pub
        # (or the synthetic Yield variant). Bucket into the action list.
        trace = agents_by_pubkey.get(env.emitter_pubkey.hex())
        if trace is None:
            continue
        ar = _action_record_from(committed)
        trace.actions.append(ar)

    return Manifest(
        schema_version=schema_version,
        kernel_version=kernel_version,
        build_hash=build_hash,
        final_chain_hash=final_chain_hash,
        model_versions=model_versions,
        agents=agents_by_id,
        path=path,
    )


def _iter_committed_events(events_list: list) -> Iterable[ledger_pb2.CommittedEvent]:
    """Parse each event from its protojson form into a CommittedEvent."""
    for event_dict in events_list:
        # ``event_dict`` is already the dict form of CommittedEvent.
        # protojson.Parse takes a JSON string; we round-trip through
        # json.dumps here. ignore_unknown_fields=True so a newer
        # kernel's added fields don't break old readers.
        raw_str = json.dumps(event_dict)
        committed = ledger_pb2.CommittedEvent()
        Parse(raw_str, committed, ignore_unknown_fields=True)
        yield committed


def _action_record_from(committed: ledger_pb2.CommittedEvent) -> ActionRecord:
    env = committed.envelope
    et = env.event_type
    base = dict(
        tick=env.tick,
        global_commit_seq=committed.global_commit_seq,
        event_type=et,
    )
    if et == "Speak":
        msg = events_pb2.Speak()
        msg.ParseFromString(env.payload)
        return ActionRecord(speak_channel=msg.channel_id, speak_content=msg.content, **base)
    if et == "Vote":
        msg = events_pb2.Vote()
        msg.ParseFromString(env.payload)
        return ActionRecord(vote_option=msg.option, **base)
    if et in ("Yield", "Yield_Kernel_Synthesized"):
        msg = events_pb2.Yield()
        msg.ParseFromString(env.payload)
        return ActionRecord(yield_reason=msg.reason, **base)
    # OpenChannelReq / CloseChannelReq / Request|DropCapability /
    # DeclareIntent — DeclareIntent is captured above; the others are
    # control intents the agent emitted but bucketed under actions for
    # completeness.
    return ActionRecord(**base)


# --- Aggregation guard --------------------------------------------------


def require_same_model_versions(
    manifests: Sequence[Manifest],
) -> ManifestModelVersions:
    """Aggregation guard. Raises ``ManifestModelVersionMismatch`` if
    any two manifests disagree on ``model_versions``. Returns the
    common ``ManifestModelVersions`` on success.

    Manifests with an unpopulated model_versions (Step 1–4 byte-
    identical runs that don't touch an LLM) are treated as compatible
    with each other but mutually incompatible with any populated
    manifest — preventing accidental Stage A aggregation that mixes
    pre-Step-5 stub runs with Step 5 LLM runs.
    """
    if not manifests:
        return ManifestModelVersions()
    groups: dict[ManifestModelVersions, list[Manifest]] = {}
    for m in manifests:
        groups.setdefault(m.model_versions, []).append(m)
    if len(groups) == 1:
        return next(iter(groups))
    raise ManifestModelVersionMismatch(_format_mismatch(groups))


def partition_by_model_versions(
    manifests: Iterable[Manifest],
) -> dict[ManifestModelVersions, list[Manifest]]:
    """Group manifests by their model_versions block. Used by callers
    that want to aggregate per-revision rather than fail on mismatch."""
    out: dict[ManifestModelVersions, list[Manifest]] = {}
    for m in manifests:
        out.setdefault(m.model_versions, []).append(m)
    return out


def _format_mismatch(
    groups: dict[ManifestModelVersions, list[Manifest]],
) -> str:
    """Build the research-facing error message. Format locked in
    PAPER_NOTES.md."""
    n_manifests = sum(len(v) for v in groups.values())
    parts = [
        "cannot aggregate across mismatched model_versions.",
        "",
        f"Found {len(groups)} distinct revision groups across {n_manifests} manifests:",
        "",
    ]
    # Sort groups by their agent_model_revision_sha for deterministic
    # output (manifests fed in different order produce the same error).
    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: (
            kv[0].agent_model_revision_sha,
            kv[0].judge_model_revision_sha,
            kv[0].agent_model,
        ),
    )
    for label_idx, (mv, group_manifests) in enumerate(sorted_groups):
        label = chr(ord("A") + label_idx) if label_idx < 26 else f"#{label_idx}"
        parts.append(f"Group {label} ({len(group_manifests)} manifest{'s' if len(group_manifests) != 1 else ''}):")
        parts.append(mv.render_block(indent="  "))
        parts.append("  manifests:")
        for m in sorted(group_manifests, key=lambda m: m.path or ""):
            parts.append(f"    - {m.path or '(in-memory)'}")
        parts.append("")
    parts.extend(
        [
            "To proceed:",
            "  * If both revisions should be reported separately, partition",
            "    via analysis.metrics.manifest_reader.partition_by_model_versions",
            "    and aggregate per-partition.",
            "  * If the runs should be re-done at a single pinned revision,",
            "    re-run byzminds-run with the appropriate flags so each",
            "    manifest's model_versions matches.",
        ]
    )
    return "\n".join(parts)


# --- Misc helpers ------------------------------------------------------


def manifest_digest(m: Manifest) -> str:
    """Stable per-manifest identifier for indexing aggregations.

    Used by the headline experiment to key per-run results without
    relying on the on-disk filename.
    """
    if m.final_chain_hash:
        return m.final_chain_hash
    h = hashlib.sha256()
    h.update(m.kernel_version.encode())
    h.update(b"|")
    h.update(m.build_hash.encode())
    h.update(b"|")
    h.update(repr(sorted(m.agents.keys())).encode())
    return h.hexdigest()
