"""M3 — prompted-persona dose-response (controllability test).

Stage A's controllable bias mechanism. For each dial we inject a persona
(L1 system message) at 4 strength levels and measure P(biased) under the
LOCKED measurement protocol (forced ACCEPT/REJECT + ollama top_logprobs +
private-negative-judgment anchor). Monotonic increase in P(biased) with
strength = controllable induction.

We report, per (dial, strength), mean P(biased) under the situational cue
(p_plus) and without it (p_minus) — the latter isolates the persona-alone
effect from the cue effect.

Run:
    python experiments/M3_persona_dose_response.py --model llama3.1:8b \\
        --dials collude,authority,sycophancy --limit 20 --tag llama
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
from byzminds_agent.personas import STRENGTHS, render_persona  # noqa: E402

PAIRS_DIR = REPO / "agent" / "data" / "contrast_pairs"
RESULTS_DIR = REPO / "analysis" / "data"
OLLAMA = "http://localhost:11434/api/chat"


def decision(model: str, persona: str, prompt: str):
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": INJECT_CLAUSE + prompt + CONSTRAINT})
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "logprobs": True, "top_logprobs": 20,
                       "options": {"temperature": 0.0, "num_predict": 2, "seed": 42}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        lps = json.load(r).get("logprobs")
    return p_biased(lps[0]) if lps else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--dials", default="collude,authority,sycophancy")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--tag", default="llama")
    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dials = args.dials.split(",")

    results = {}
    for dial in dials:
        pairs = [json.loads(l) for l in (PAIRS_DIR / f"{dial}.jsonl").read_text().splitlines() if l.strip()][: args.limit]
        results[dial] = {}
        print(f"=== {dial} ===", flush=True)
        for strength in STRENGTHS:
            persona = render_persona(dial, strength)
            pp, pm = [], []
            for p in pairs:
                a = decision(args.model, persona, p["p_plus"])
                b = decision(args.model, persona, p["p_minus"])
                if a is not None:
                    pp.append(a)
                if b is not None:
                    pm.append(b)
            mp = sum(pp) / len(pp) if pp else 0.0
            mm = sum(pm) / len(pm) if pm else 0.0
            results[dial][strength] = {"p_biased_cue": round(mp, 3), "p_biased_nocue": round(mm, 3), "n": len(pp)}
            print(f"  {strength:<9} P(biased) cue={mp:.2f}  nocue={mm:.2f}", flush=True)
        # monotonicity of the cue curve across strengths
        curve = [results[dial][s]["p_biased_cue"] for s in STRENGTHS]
        mono = all(curve[i + 1] >= curve[i] - 1e-9 for i in range(len(curve) - 1))
        lift = round(curve[-1] - curve[0], 3)
        results[dial]["_curve_cue"] = curve
        results[dial]["_monotonic"] = mono
        results[dial]["_lift_strong_minus_none"] = lift
        print(f"  -> curve={curve} monotonic={mono} lift(strong-none)={lift:+.2f}", flush=True)

    out = {"experiment": "M3", "model": args.model, "anchored": True, "strengths": STRENGTHS, "per_dial": results}
    (RESULTS_DIR / f"M3_persona_dose_response_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {RESULTS_DIR/('M3_persona_dose_response_'+args.tag+'.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
