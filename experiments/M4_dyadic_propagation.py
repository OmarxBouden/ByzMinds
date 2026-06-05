"""M4 — dyadic propagation.

Does exposure to a biased peer shift an honest agent's vote? Lightweight
two-agent orchestration (no kernel/gRPC; reuses personas + the locked
forced-decision measure). For each dial:

  * Agent A (biased): strong persona for the dial, sees the anchored
    situation, produces a free-form panel statement (advocacy).
  * Agent B (honest, no persona): anchored with the private-negative
    judgment, so its baseline vote is REJECT. We measure B's P(ACCEPT)
    SOLO (no exposure) vs AFTER reading A's statement.

Propagation_d = mean P(ACCEPT | B saw A_d) - mean P(ACCEPT | B solo).

Records the full exchange (A statement, B votes) per scenario for audit.

Run:
    python experiments/M4_dyadic_propagation.py --model llama3.1:8b \\
        --dials authority,sycophancy,bandwagon,collude,deceive --limit 15 --tag llama
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent"))

from experiments.M1b_decision_logprobs import CONSTRAINT, INJECT_CLAUSE, p_biased  # noqa: E402
from byzminds_agent.personas import render_persona  # noqa: E402

PAIRS_DIR = REPO / "agent" / "data" / "contrast_pairs"
RESULTS_DIR = REPO / "analysis" / "data"
OLLAMA = "http://localhost:11434/api/chat"


def chat(model, messages, num_predict, logprobs=False):
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.0, "num_predict": num_predict, "seed": 42}}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 20
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def b_vote(model: str, situation: str, peer_msg: str | None) -> float | None:
    """Honest B votes on the anchored situation, optionally after reading A."""
    content = INJECT_CLAUSE + situation
    if peer_msg:
        content += f"\n\nAnother panel member states: \"{peer_msg}\""
    content += CONSTRAINT
    d = chat(model, [{"role": "user", "content": content}], 2, logprobs=True)
    lps = d.get("logprobs")
    return p_biased(lps[0]) if lps else None


def a_statement(model: str, dial: str, situation: str) -> str:
    """Biased A (strong persona) states its position on the situation."""
    persona = render_persona(dial, "strong")
    msgs = [{"role": "system", "content": persona},
            {"role": "user", "content": INJECT_CLAUSE + situation +
             "\n\nState your position to the panel in 2 sentences."}]
    return chat(model, msgs, 100)["message"]["content"].strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--dials", default="authority,sycophancy,bandwagon,collude,deceive")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--tag", default="llama")
    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dials = args.dials.split(",")

    per_dial = {}
    for d in dials:
        pairs = [json.loads(l) for l in (PAIRS_DIR / f"{d}.jsonl").read_text().splitlines() if l.strip()][: args.limit]
        solo, after, cells = [], [], []
        for p in pairs:
            # A sees p_plus (the biasing cue) + strong persona -> advocates ACCEPT.
            # B sees p_minus (same manuscript, NO cue) -> clean honest REJECT baseline.
            a_msg = a_statement(args.model, d, p["p_plus"])
            ps = b_vote(args.model, p["p_minus"], None)
            pa = b_vote(args.model, p["p_minus"], a_msg)
            if ps is None or pa is None:
                continue
            solo.append(ps)
            after.append(pa)
            cells.append({"pair_id": p["pair_id"], "a_statement": a_msg,
                          "b_solo_p_accept": round(ps, 3), "b_after_p_accept": round(pa, 3)})
        ms = sum(solo) / len(solo) if solo else 0.0
        ma = sum(after) / len(after) if after else 0.0
        per_dial[d] = {"n": len(solo), "b_solo_accept": round(ms, 3),
                       "b_after_accept": round(ma, 3), "propagation_shift": round(ma - ms, 3),
                       "cells": cells}
        print(f"  {d:<11} B solo={ms:.2f} -> after A={ma:.2f}  shift={ma-ms:+.2f} (n={len(solo)})", flush=True)

    out = {"experiment": "M4", "model": args.model, "anchored": True,
           "per_dial": {d: {k: v for k, v in per_dial[d].items() if k != "cells"} for d in dials},
           "_cells": {d: per_dial[d]["cells"] for d in dials}}
    (RESULTS_DIR / f"M4_dyadic_propagation_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {RESULTS_DIR/('M4_dyadic_propagation_'+args.tag+'.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
