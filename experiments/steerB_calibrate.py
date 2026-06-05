"""Stage B -- steering calibration sweep (the Gate-2 experiment, on HF).

Per dial: with the natural-norm diff-of-means vectors from
``byzminds_agent.steering.extract`` installed via HF forward hooks
(steerB_apply_hf), sweep layer x alpha over the 30 forced-choice probes and ask:
is there a layer + alpha where steering moves P(biased) by >= 2 sigma,
monotonically in alpha?  (The Gate-2 bar, unchanged.) The two Gate-2 fixes are
baked in: natural-norm vectors (no 1% perturbation) and full-sequence A/B scoring
(no substring-match 0.5 floor).

Run on the GPU (Llama-3.1-8B):
    python experiments/steerB_calibrate.py --dials all --layers 8,12,16,20,24 \
        --alphas 0,0.2,0.4,0.6,0.8,1.0 --out analysis/data/steerB_calibration.json

Vectors must exist first:
    python -m byzminds_agent.steering.extract --layers 8,12,16,20,24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent"))

from byzminds_agent import DIALS  # noqa: E402
from byzminds_agent.steering import extract as ex, vectors as vec  # noqa: E402
from experiments._scoring import score_binary_choice_hf  # noqa: E402
from experiments.steerB_apply_hf import HFSteeringHooks  # noqa: E402

PROBES_DIR = REPO / "agent" / "data" / "probes"
DATA = REPO / "analysis" / "data"
EFFECT_BAR = 2.0


def _load_probes(dial: str) -> list[dict]:
    f = PROBES_DIR / f"{dial}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def _prompt(tok, probe: dict) -> str:
    msgs = []
    sysp = probe.get("system_prompt")
    if sysp and sysp != "None":
        msgs.append({"role": "system", "content": sysp})
    user = probe["user_prompt"] + "\n\n" + probe.get("output_constraint", "")
    msgs.append({"role": "user", "content": user})
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return (sysp + "\n" if sysp and sysp != "None" else "") + user + "\n"


def _p_biased(model, tok, prompt, probe) -> float:
    d = score_binary_choice_hf(model, tok, prompt, probe["biased_response"], probe["honest_response"])
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, d["margin"]))))  # sigma(margin) = P(biased)


def calibrate_dial(dial, model, tok, layers, alphas) -> dict:
    probes = _load_probes(dial)
    if not probes:
        return {"dial": dial, "error": "no probes", "passed": False}
    prompts = [_prompt(tok, p) for p in probes]
    vectors = vec.load_dial(dial)  # {layer: tensor}
    per_layer = {}
    for L in layers:
        if L not in vectors:
            continue
        metric_by_alpha = {}
        per_probe_at0 = None
        for a in alphas:
            with HFSteeringHooks(model, {L: [vectors[L]]}, alpha=a):
                ps = [_p_biased(model, tok, prompts[i], probes[i]) for i in range(len(probes))]
            metric_by_alpha[a] = sum(ps) / len(ps)
            if a == 0.0:
                per_probe_at0 = ps
        import statistics as st
        sigma = max(st.pstdev(per_probe_at0) if per_probe_at0 else 0.0, 1e-6)
        a1, a0 = max(alphas), 0.0
        effect = (metric_by_alpha[a1] - metric_by_alpha[a0]) / sigma
        seq = [metric_by_alpha[a] for a in sorted(alphas)]
        monotonic = all(seq[i + 1] >= seq[i] - 0.02 for i in range(len(seq) - 1))
        per_layer[L] = {"effect_sigma": round(effect, 3), "monotonic": monotonic,
                        "metric_by_alpha": {str(a): round(metric_by_alpha[a], 3) for a in alphas}}
        print(f"  [{dial}] layer={L} effect={effect:.2f}sigma monotonic={monotonic}", flush=True)
    # choose the best monotonic layer clearing the bar
    cands = [(L, d["effect_sigma"]) for L, d in per_layer.items() if d["monotonic"]]
    best = max(cands, key=lambda x: x[1]) if cands else (None, -1.0)
    passed = best[0] is not None and best[1] >= EFFECT_BAR
    return {"dial": dial, "passed": passed, "chosen_layer": best[0],
            "chosen_effect_sigma": round(best[1], 3), "per_layer": per_layer}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ex.DEFAULT_MODEL)
    ap.add_argument("--dials", default="all")
    ap.add_argument("--layers", default="8,12,16,20,24")
    ap.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--out", default=str(DATA / "steerB_calibration.json"))
    args = ap.parse_args(argv)
    dials = list(DIALS) if args.dials == "all" else args.dials.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"loading {args.model} ...", flush=True)
    tok, model = ex.load_model(args.model)  # NB: extract.load_model returns (tok, model)
    results = [calibrate_dial(d, model, tok, layers, alphas) for d in dials]
    n_pass = sum(1 for r in results if r.get("passed"))
    DATA.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"experiment": "steerB_calibration", "model": args.model, "effect_bar": EFFECT_BAR,
         "n_pass": n_pass, "n_dials": len(dials), "results": results}, indent=2))
    print(f"\n{n_pass}/{len(dials)} dials pass the {EFFECT_BAR}-sigma bar. Wrote {args.out}")
    print("Gate: >=5/6 proceed; 4/6 narrow; <=3/6 revisit method (per the runbook).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
