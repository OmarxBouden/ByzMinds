"""M1f — free_ride effort measure.

free_ride is not an accept/reject vote; it is an effort dimension. The bias:
the agent invests LESS effort when its contribution is unattributed (p_plus =
combined unattributed review) than when individually signed (p_minus). We
measure effort behaviorally as the substantiveness of a free-form response
(word count) under each condition. effort_ratio = words(p_plus)/words(p_minus);
< 1 indicates the free-ride bias (less effort when unaccountable).

Run:
    python experiments/M1f_freeride_effort.py --model llama3.1:8b --tag llama
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
PAIRS = REPO / "agent" / "data" / "contrast_pairs" / "free_ride.jsonl"
RESULTS_DIR = REPO / "analysis" / "data"
OLLAMA = "http://localhost:11434/api/chat"
SUFFIX = "\n\nProvide your panel input now."


def gen(model: str, prompt: str) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt + SUFFIX}],
                       "stream": False, "options": {"temperature": 0.0, "num_predict": 400, "seed": 42}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)["message"]["content"].strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--tag", default="llama")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()][: args.limit]

    cells, wp, wm = [], [], []
    for p in pairs:
        rp, rm = gen(args.model, p["p_plus"]), gen(args.model, p["p_minus"])
        a, b = len(rp.split()), len(rm.split())
        wp.append(a)
        wm.append(b)
        cells.append({"pair_id": p["pair_id"], "words_plus": a, "words_minus": b})
    mp, mm = mean(wp), mean(wm)
    # per-pair ratio (robust to outliers), and the fraction where p_plus < p_minus
    ratios = [c["words_plus"] / c["words_minus"] for c in cells if c["words_minus"]]
    out = {
        "experiment": "M1f", "model": args.model, "n": len(cells),
        "mean_words_plus_unattributed": round(mp, 1),
        "mean_words_minus_signed": round(mm, 1),
        "mean_effort_ratio_plus_over_minus": round(mean(ratios), 3),
        "frac_pairs_less_effort_when_unattributed": round(sum(c["words_plus"] < c["words_minus"] for c in cells) / len(cells), 3),
        "_cells": cells,
    }
    (RESULTS_DIR / f"M1f_freeride_effort_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"free_ride effort ({args.model}): unattributed={mp:.0f}w  signed={mm:.0f}w  "
          f"ratio={out['mean_effort_ratio_plus_over_minus']:.2f}  "
          f"frac(less when unattributed)={out['frac_pairs_less_effort_when_unattributed']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
