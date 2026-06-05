"""M5 panel-clustered confidence intervals.

The §4.5 detector rates report SEs over agent-instances, but agents within a
panel interact, so that under-states uncertainty (pseudo-replication). This
recomputes each detector's group rate clustered by panel: per-panel group mean,
then mean +/- SE over panels (the effective sample is the number of panels).
Qualitative conclusions are unchanged; this gives the honest CIs for App. A7.

    python experiments/M5_panel_cis.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "analysis" / "data"

DETECTORS = {
    "public_only": lambda r: (1 - r["H_a_blind"]) if r.get("H_a_blind") is not None else None,
    "public_plus_gt": lambda r: (1 - r["H_a"]) if r.get("H_a") is not None else None,
    "private": lambda r: (r["private_bias"] or 0.0),
    "accept_vote": lambda r: r["p_accept"],
}


def _clustered(rows, fn):
    """Per-panel (tag) mean of fn over rows, then mean +/- SE over panels."""
    by_panel = defaultdict(list)
    for r in rows:
        v = fn(r)
        if v is not None:
            by_panel[r["tag"]].append(v)
    panel_means = [float(np.mean(v)) for v in by_panel.values() if v]
    a = np.asarray(panel_means, float)
    if a.size == 0:
        return None
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return {"mean": round(float(a.mean()), 3), "se_clustered": round(se, 3), "n_panels": int(a.size)}


def main(argv=None) -> int:
    rows = json.load(open(DATA / "M5_panel_results.json"))["results"]
    coll = [r for r in rows if r["biased"]]
    hon = [r for r in rows if not r["biased"]]
    out = {"experiment": "M5_panel_cis",
           "note": "per-panel group mean, then SE over panels (clusters agents by panel)",
           "detectors": {}}
    print("detector            colluder (mean±SE, panels)     honest (mean±SE, panels)")
    for name, fn in DETECTORS.items():
        c, h = _clustered(coll, fn), _clustered(hon, fn)
        out["detectors"][name] = {"colluder": c, "honest": h}
        print(f"  {name:16s} {c['mean']:.3f}±{c['se_clustered']:.3f} (n={c['n_panels']})        "
              f"{h['mean']:.3f}±{h['se_clustered']:.3f} (n={h['n_panels']})")
    (DATA / "M5_panel_cis.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {DATA/'M5_panel_cis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
