"""M3f — free_ride persona dose-response (effort measure).

free_ride is an effort dimension, not an accept/reject vote, so its
dose-response uses the M1f effort measure (free-form response word count)
rather than the forced-decision protocol. Under the low-accountability
framing (p_plus: unattributed combined review), a stronger free_ride
persona should produce LESS effort (fewer words). We report mean words per
persona strength under p_plus (low accountability) and p_minus (signed),
and the effort lift none->strong.

Run:
    python experiments/M3f_freeride_persona.py --model llama3.1:8b --tag llama
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent"))

from byzminds_agent.personas import STRENGTHS, render_persona  # noqa: E402

PAIRS = REPO / "agent" / "data" / "contrast_pairs" / "free_ride.jsonl"
RESULTS_DIR = REPO / "analysis" / "data"
OLLAMA = "http://localhost:11434/api/chat"
SUFFIX = "\n\nProvide your panel input now."


def words(model: str, persona: str, prompt: str) -> int:
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": prompt + SUFFIX})
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "options": {"temperature": 0.0, "num_predict": 400, "seed": 42}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return len(json.load(r)["message"]["content"].split())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--tag", default="llama")
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()][: args.limit]

    per_strength = {}
    for s in STRENGTHS:
        persona = render_persona("free_ride", s)
        wp = [words(args.model, persona, p["p_plus"]) for p in pairs]   # unattributed (low accountability)
        wm = [words(args.model, persona, p["p_minus"]) for p in pairs]  # signed (high accountability)
        per_strength[s] = {"mean_words_unattributed": round(mean(wp), 1),
                           "mean_words_signed": round(mean(wm), 1), "n": len(wp)}
        print(f"  {s:<9} unattributed={mean(wp):.0f}w  signed={mean(wm):.0f}w", flush=True)
    curve = [per_strength[s]["mean_words_unattributed"] for s in STRENGTHS]
    effort_lift = round(curve[0] - curve[-1], 1)  # fewer words at strong = more free-riding
    out = {"experiment": "M3f", "model": args.model, "dial": "free_ride", "strengths": STRENGTHS,
           "per_strength": per_strength, "effort_drop_none_minus_strong_words": effort_lift}
    (RESULTS_DIR / f"M3f_freeride_persona_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"free_ride effort dose-response ({args.model}): unattributed words "
          f"{curve} ; drop none->strong = {effort_lift}w", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
