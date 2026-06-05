"""Stage B -- held-out validation of response-contrast (CAA) steering.

The response-contrast extractor builds the steering vector from the biased-vs-
honest *response tokens* and the calibration then scores that same biased/honest
forced choice on the *same* 30 probes -- so the in-sample effect is partly
mechanical (steering toward the biased-response direction raises its logprob on
the very probes it was fit on). This script removes that circularity: for each
dial it splits the 30 probes 50/50, extracts the response-contrast vector on the
*train* half, and measures the steered shift on the disjoint *test* half against
baseline and the prompted persona. If steering still reproduces a large fraction
of the prompted shift out-of-sample, the dial is a genuine linear direction.

    python experiments/steerB_heldout.py --model NousResearch/Meta-Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent"))

from byzminds_agent import DIALS  # noqa: E402
from byzminds_agent.steering import extract as ex  # noqa: E402
from experiments.steerB_apply_hf import HFSteeringHooks  # noqa: E402
from experiments.steerB_calibrate import _load_probes, _p_biased, _prompt  # noqa: E402
from experiments.steerB_evaluate import _persona_prompt  # noqa: E402
from experiments.steerB_extract_response import _response_reps  # noqa: E402

DATA = REPO / "analysis" / "data"
# layer chosen by the full-data response-contrast calibration (reported layer)
CHOSEN_LAYER = {"authority": 12, "bandwagon": 12, "sycophancy": 12,
                "free_ride": 16, "collude": 12, "deceive": 12}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ex.DEFAULT_MODEL)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--calibration", default=None,
                    help="read each dial's chosen_layer from this calibration JSON "
                         "(per-model layers); falls back to CHOSEN_LAYER where absent/None")
    ap.add_argument("--out", default=str(DATA / "steerB_heldout.json"))
    args = ap.parse_args(argv)

    import torch

    chosen = dict(CHOSEN_LAYER)
    if args.calibration:
        cal = json.load(open(args.calibration))
        for r in cal.get("results", []):
            if r.get("chosen_layer") is not None:
                chosen[r["dial"]] = r["chosen_layer"]
        print(f"chosen layers from {args.calibration}: {chosen}", flush=True)

    print(f"loading {args.model} ...", flush=True)
    tok, model = ex.load_model(args.model)
    results = []
    for dial in DIALS:
        probes = _load_probes(dial)
        train, test = probes[::2], probes[1::2]                 # disjoint halves
        L = chosen[dial]
        # extract response-contrast vector on TRAIN only
        diffs = []
        for p in train:
            prompt = _prompt(tok, p)
            rp = _response_reps(model, tok, prompt, p["biased_response"], [L], 0)[L]
            rm = _response_reps(model, tok, prompt, p["honest_response"], [L], 0)[L]
            diffs.append((rp - rm).detach().cpu())
        v = torch.stack(diffs).mean(dim=0)
        # evaluate on disjoint TEST
        base_prompts = [_prompt(tok, p) for p in test]
        pers_prompts = [_persona_prompt(tok, p, dial) for p in test]
        p_base = sum(_p_biased(model, tok, base_prompts[i], test[i]) for i in range(len(test))) / len(test)
        p_pers = sum(_p_biased(model, tok, pers_prompts[i], test[i]) for i in range(len(test))) / len(test)
        with HFSteeringHooks(model, {L: [v]}, alpha=args.alpha):
            p_steer = sum(_p_biased(model, tok, base_prompts[i], test[i]) for i in range(len(test))) / len(test)
        r = {"dial": dial, "layer": L, "alpha": args.alpha, "n_train": len(train), "n_test": len(test),
             "vector_norm": round(float(v.norm()), 3),
             "p_biased_baseline": round(p_base, 3), "p_biased_persona": round(p_pers, 3),
             "p_biased_steered": round(p_steer, 3),
             "steered_shift": round(p_steer - p_base, 3), "persona_shift": round(p_pers - p_base, 3),
             "reproduction_frac": round((p_steer - p_base) / (p_pers - p_base), 3) if abs(p_pers - p_base) > 1e-6 else None}
        results.append(r)
        print(f"  {dial:<11} held-out: baseline={r['p_biased_baseline']} persona={r['p_biased_persona']} "
              f"steered={r['p_biased_steered']}  steer_shift={r['steered_shift']:+.3f} "
              f"persona_shift={r['persona_shift']:+.3f} repro={r['reproduction_frac']}", flush=True)
    Path(args.out).write_text(json.dumps(
        {"experiment": "steerB_heldout", "model": args.model,
         "method": "response-contrast (CAA) extracted on 15 train probes, steered shift measured on 15 disjoint test probes",
         "results": results}, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
