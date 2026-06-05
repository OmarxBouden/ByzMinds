"""export_frontend.py -- decode signed manifests into a frontend-friendly JSON.

The manifest event payloads are base64 protobuf (opaque to JS), but the envelope
carries event_type / ledger_id / global_commit_seq / chain hashes as plain
fields. So we read order, ledgers and the hash chain from the raw gzipped JSON,
and join the decoded *content* (speak text, vote, theta, declare-intent) from
analysis.metrics.manifest_reader by global_commit_seq and (agent, tick).

Output per run -> frontend/data/<name>.json:
  agents[]   id, role, pubkey, profile{theta, dial, strength, biased}
  channels[] id, members           (from observed speak_channel; "public" always)
  ticks[]    {tick, phase, events[]}   events ordered by global_commit_seq, each
             {seq, ledger, agent, type, channel?, content?, vote?, theta?}
  ledgers    L_pub/L_prv/L_cog_ind/L_cog_eli/L_ctrl  (each event tagged)
  replay     {ok, n_events, final_chain_hash}   hash-chain linkage check
  summary    title, n_agents, n_ticks, biased_ids, has_private
Plus frontend/data/index.json listing all runs.

    python experiments/export_frontend.py runs/_demo/smoke_collude.json.gz [more...]
    python experiments/export_frontend.py            # default demo set
"""

from __future__ import annotations

import base64
import glob
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent"))

from byzminds_agent import DIALS  # noqa: E402
from analysis.metrics import manifest_reader  # noqa: E402

OUT = REPO / "frontend" / "data"


def _theta_to_profile(theta) -> dict:
    theta = list(theta or [])
    if not theta or all(abs(x) < 1e-9 for x in theta):
        return {"theta": theta, "dial": None, "strength": "none", "biased": False}
    idx = max(range(len(theta)), key=lambda i: theta[i])
    mag = theta[idx]
    strength = "strong" if mag > 0.8 else "moderate" if mag > 0.5 else "mild"
    return {"theta": theta, "dial": DIALS[idx] if idx < len(DIALS) else None,
            "strength": strength, "biased": True}


def _b64_to_hex(b64: str) -> str:
    return base64.b64decode(b64).hex()


def export_one(path: str) -> dict:
    raw = json.load(gzip.open(path))
    m = manifest_reader.read(path)

    # pubkey(hex) -> agent_id, and decoded content joins
    hex_to_aid, agents = {}, []
    act_by_seq = {}      # global_commit_seq -> (aid, ActionRecord)
    cogind_by = {}       # (aid, tick) -> theta
    cogeli_by = {}       # (aid, tick) -> [content]
    for aid, tr in m.agents.items():
        hex_to_aid[tr.pubkey_hex.lower()] = aid
        theta = tr.cog_ind[0].theta if tr.cog_ind else tr.initial_theta
        agents.append({"id": aid, "role": tr.role, "pubkey": tr.pubkey_hex[:12],
                       "profile": _theta_to_profile(theta)})
        for a in tr.actions:
            act_by_seq[int(a.global_commit_seq)] = (aid, a)
        for ci in tr.cog_ind:
            cogind_by[(aid, ci.tick)] = list(ci.theta)
        for ce in tr.cog_eli:
            cogeli_by.setdefault((aid, ce.tick), []).append(ce.content)

    # walk the raw events in committed order; tag ledger; attach decoded content
    channels = {"public": set()}
    ticks: dict[int, list] = {}
    private_ticks, vote_ticks = set(), set()
    prev_hash, chain_ok = None, True
    for ev in raw["events"]:
        env = ev["envelope"]
        etype = env["event_type"].replace("Handler_", "")
        seq = int(ev["global_commit_seq"])
        emitter = hex_to_aid.get(_b64_to_hex(env["emitter_pubkey"]).lower())
        # hash-chain linkage
        if prev_hash is not None and ev.get("prev_chain_hash") != prev_hash:
            chain_ok = False
        prev_hash = ev.get("chain_hash")

        # CogIndSnapshot / DeclareIntent / ElicitationRequest are re-added below
        # from the decoded per-tick maps (which carry theta / intent text).
        if etype in ("CogIndSnapshot", "DeclareIntent", "ElicitationRequest"):
            continue

        node = {"seq": seq, "type": etype, "agent": emitter, "ledger": "L_ctrl"}
        tick = None
        if seq in act_by_seq and etype in ("Speak", "Vote", "Yield"):
            aid, a = act_by_seq[seq]
            tick = a.tick
            if etype == "Speak":
                ch = a.speak_channel or "public"
                node.update(channel=ch, content=a.speak_content,
                            ledger="L_pub" if ch in ("", "public") else "L_prv")
                channels.setdefault(ch, set()).add(aid)
                if ch not in ("", "public"):
                    private_ticks.add(tick)
            elif etype == "Vote":
                node.update(vote=("accept" if (a.vote_option or "").lower().startswith("acc") else "reject"),
                            ledger="L_pub")
                vote_ticks.add(tick)
            else:
                node.update(reason=a.yield_reason, ledger="L_pub")
        # else: control events (SpawnAgent/LoadScenario/OpenChannel) keep L_ctrl.

        # control events carry no agent tick -> bucket at tick 0
        ticks.setdefault(tick if tick is not None else 0, []).append(node)

    # attach per-tick cog_ind theta + declare-intents (keyed by tick, not seq)
    for (aid, t), theta in cogind_by.items():
        ticks.setdefault(t, []).append({"seq": -1, "type": "CogIndSnapshot", "agent": aid,
                                        "ledger": "L_cog_ind", "theta": theta})
    for (aid, t), contents in cogeli_by.items():
        for c in contents:
            ticks.setdefault(t, []).append({"seq": -1, "type": "DeclareIntent", "agent": aid,
                                            "ledger": "L_cog_eli", "content": c})

    def phase_of(t):
        if t in vote_ticks:
            return "vote"
        if t in private_ticks:
            return "private_consultation"
        return "public_deliberation"

    tick_list = [{"tick": t, "phase": phase_of(t),
                  "events": sorted(ticks[t], key=lambda n: (n["seq"] < 0, n["seq"]))}
                 for t in sorted(ticks)]
    biased = [a["id"] for a in agents if a["profile"]["biased"]]
    name = Path(path).name.replace(".json.gz", "")

    # flat committed log for the replay/ledger dashboard (authoritative ledger_id)
    lmap = {"LEDGER_ID_L_PUB": "L_pub", "LEDGER_ID_L_PRV": "L_prv", "LEDGER_ID_L_COG_IND": "L_cog_ind",
            "LEDGER_ID_L_COG_ELI": "L_cog_eli", "LEDGER_ID_L_CTRL": "L_ctrl"}
    events = sorted(({"seq": int(ev["global_commit_seq"]),
                      "ledger": lmap.get(ev.get("ledger_id", ""), ev.get("ledger_id", "")),
                      "type": ev["envelope"]["event_type"].replace("Handler_", ""),
                      "agent": hex_to_aid.get(_b64_to_hex(ev["envelope"]["emitter_pubkey"]).lower()),
                      "chain": (ev.get("chain_hash") or "")[:10]} for ev in raw["events"]),
                    key=lambda e: e["seq"])

    return {
        "name": name,
        "agents": agents,
        "events": events,
        "channels": [{"id": c, "members": sorted(mem)} for c, mem in channels.items()],
        "ticks": tick_list,
        # per-event chain_hash is base64; final_chain_hash is hex -> compare as bytes
        "replay": {"ok": chain_ok and bool(prev_hash)
                   and _b64_to_hex(prev_hash) == raw.get("final_chain_hash"),
                   "n_events": len(raw["events"]),
                   "final_chain_hash": (raw.get("final_chain_hash") or "")[:16],
                   "chain_linked": chain_ok},
        "summary": {"title": name, "n_agents": len(agents), "n_ticks": len(tick_list),
                    "biased_ids": biased, "has_private": bool(private_ticks),
                    "kernel_version": raw.get("kernel_version", "")},
    }


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    paths = argv or [p for p in [
        str(REPO / "runs" / "_demo" / "smoke_collude.json.gz"),
        str(REPO / "runs" / "M6" / "smoke_authority_k2_long.json.gz"),
    ] if Path(p).exists()]
    if not paths:
        print("no manifests found; pass paths or generate demo manifests first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for p in paths:
        for mp in (glob.glob(p) if "*" in p else [p]):
            try:
                run = export_one(mp)
            except Exception as e:  # robust: skip a bad manifest, keep going
                print(f"  SKIP {mp}: {type(e).__name__}: {e}", flush=True)
                continue
            (OUT / f"{run['name']}.json").write_text(json.dumps(run, indent=1))
            index.append({"name": run["name"], **run["summary"], "replay_ok": run["replay"]["ok"]})
            print(f"  wrote {run['name']}.json  agents={run['summary']['n_agents']} "
                  f"ticks={run['summary']['n_ticks']} biased={run['summary']['biased_ids']} "
                  f"replay_ok={run['replay']['ok']}", flush=True)
    (OUT / "index.json").write_text(json.dumps({"runs": index}, indent=1))
    print(f"\nwrote {OUT/'index.json'} ({len(index)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
