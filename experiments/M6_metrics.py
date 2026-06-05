"""M6 -- protocol robustness metrics: cost-to-attack from recorded votes.

The kernel records votes; aggregation is downstream, so a single panel scores
every aggregation rule. Per cell (deliberation length x dial) we measure, from
the manifests:

  * biased_accept_rate q -- fraction of biased agents that vote accept (~1; they
    advocate acceptance).
  * honest_flip_rate p  -- fraction of *honest* agents that vote accept, i.e.
    how often the biased minority's deliberation propagates and flips an
    anchored honest reviewer. This is the quantity deliberation length is
    hypothesised to move.

A protocol's decision on a flawed manuscript (ground truth: reject) is wrong if
too many accept. For a 5-panel with k biased agents:
    accept_count = Binom(k, q) + Binom(5-k, p)
    majority      fails if accept_count >= 3
    supermajority fails if accept_count >= 4
Cost-to-attack(protocol) = min k in 1..5 with P(fail) >= 0.5, projected from the
measured (p, q). We also report the empirical fail rate at the k actually run.

    python experiments/M6_metrics.py --index runs/M6/index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent"))

from experiments.M5_metrics import agent_records  # noqa: E402

DATA = REPO / "analysis" / "data"
N_AGENTS = 5
PROTOCOLS = {"majority": 3, "supermajority": 4}  # accept_count >= threshold => accept (wrong here)


def _binom_pmf(n: int, k: int, p: float) -> float:
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) if 0 <= k <= n else 0.0


def _accept_count_dist(k: int, q: float, p: float) -> list[float]:
    """P(accept_count = c) for c in 0..5, from Binom(k,q)+Binom(5-k,p)."""
    hb = 5 - k
    dist = [0.0] * (N_AGENTS + 1)
    for a in range(k + 1):
        for b in range(hb + 1):
            dist[a + b] += _binom_pmf(k, a, q) * _binom_pmf(hb, b, p)
    return dist


def _p_fail(k: int, q: float, p: float, threshold: int) -> float:
    dist = _accept_count_dist(k, q, p)
    return sum(dist[c] for c in range(threshold, N_AGENTS + 1))


def cost_to_attack(q: float, p: float, threshold: int) -> int:
    for k in range(1, N_AGENTS + 1):
        if _p_fail(k, q, p, threshold) >= 0.5:
            return k
    return N_AGENTS + 1  # never breaks (>5 => unbreakable in a 5-panel)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(REPO / "runs" / "M6" / "index.json"))
    args = ap.parse_args(argv)
    index = json.load(open(args.index))

    # Collect per (delib, dial): honest accepts/total, biased accepts/total, and
    # empirical per-k fail counts.
    hon_acc = defaultdict(lambda: [0, 0])   # (delib,dial) -> [accepts, total]
    bia_acc = defaultdict(lambda: [0, 0])
    emp = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # (delib,dial)->k->[n, maj_fail, sup_fail]
    for row in index:
        if not row.get("manifest"):
            continue
        key = (row["delib"], row["dial"])
        biased = set(row["biased_ids"])
        k = row["biased_fraction"]
        recs = agent_records(row["manifest"])
        accept_count = 0
        for aid, (vote, _pub, _prv) in recs.items():
            if vote is None:
                continue
            acc = 1 if vote == 1 else 0
            accept_count += acc
            tgt = bia_acc if aid in biased else hon_acc
            tgt[key][0] += acc
            tgt[key][1] += 1
        cell = emp[key][k]
        cell[0] += 1
        cell[1] += 1 if accept_count >= PROTOCOLS["majority"] else 0
        cell[2] += 1 if accept_count >= PROTOCOLS["supermajority"] else 0

    results = []
    for key in sorted(hon_acc):
        delib, dial = key
        p = hon_acc[key][0] / hon_acc[key][1] if hon_acc[key][1] else 0.0
        q = bia_acc[key][0] / bia_acc[key][1] if bia_acc[key][1] else 1.0
        row = {"delib": delib, "dial": dial,
               "honest_flip_rate": round(p, 3), "biased_accept_rate": round(q, 3),
               "n_honest": hon_acc[key][1], "n_biased": bia_acc[key][1],
               "cost_to_attack": {pr: cost_to_attack(q, p, th) for pr, th in PROTOCOLS.items()},
               "empirical": {str(k): {"n": c[0], "majority_fail": c[1], "supermajority_fail": c[2]}
                             for k, c in sorted(emp[key].items())}}
        results.append(row)
        cta = row["cost_to_attack"]
        print(f"  {delib:5s} {dial:11s} flip_rate={p:.2f} biased_accept={q:.2f}  "
              f"cost-to-attack: majority={cta['majority']} super={cta['supermajority']}", flush=True)

    out = {"experiment": "M6", "protocols": PROTOCOLS, "results": results}
    (DATA / "M6_protocol_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {DATA/'M6_protocol_results.json'} ({len(results)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
