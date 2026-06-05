#!/usr/bin/env python3
"""Experiment 0b — deceive layer/coherence diagnostic (HF transformers).

Follows exp_0's negative L16 result. Two questions:
  1. Does ANY candidate layer steer deceive cleanly at a perturbation
     magnitude commensurate with the residual stream (not the near-
     destructive alpha=16 of exp_0)?
  2. At the alpha where the rate moves, is the generated text COHERENT
     (real biased behaviour) or collapsed (artifact)?

Method: per layer, measure residual ||h|| at the last token, then sweep
alpha so the perturbation ratio r = ||alpha*v|| / ||h|| hits {0, .25,
.5, 1.0}. Score biased_rate by full-sequence log-likelihood. Then dump
greedy generations at baseline vs the strongest setting for eyeballing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "agent"))

from experiments._vllm_backend import format_probe_prompt  # noqa: E402
from experiments.exp_0_deceive_smoke import seq_logprob  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VECTORS_DIR = _REPO / "agent" / "data" / "steering_vectors"
PROBES_DIR = _REPO / "agent" / "data" / "probes"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", default="deceive")
    ap.add_argument("--layers", default="8,16,20,24")
    ap.add_argument("--ratios", default="0.0,0.25,0.5,1.0")
    ap.add_argument("--n-probes", type=int, default=20)
    ap.add_argument("--output-file", default="/tmp/exp_0b.json")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from byzminds_agent.steering.apply import SteeringHookManager

    layers = [int(x) for x in args.layers.split(",")]
    ratios = [float(x) for x in args.ratios.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    raw = torch.load(VECTORS_DIR / f"{args.dial}.pt", weights_only=False)
    probes = [json.loads(l) for l in (PROBES_DIR / f"{args.dial}.jsonl").read_text().splitlines() if l.strip()][: args.n_probes]
    prompts = [format_probe_prompt(p, tok) for p in probes]

    # --- per-layer residual ||h|| at last token (mean over first 5 probes) ---
    hnorm: dict[int, float] = {}
    for L in layers:
        caps = []

        def cap(mod, inp, out, _c=caps):
            h = out[0] if isinstance(out, tuple) else out
            _c.append(float(h[0, -1, :].float().norm()))

        handle = model.model.layers[L].register_forward_hook(cap)
        with torch.no_grad():
            for pr in prompts[:5]:
                ids = tok(pr, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                model(ids)
        handle.remove()
        hnorm[L] = mean(caps)

    out: dict = {"dial": args.dial, "residual_norms": hnorm, "layers": {}}
    for L in layers:
        v = raw[f"layer_{L}"]
        vn = float(v.float().norm())
        mgr = SteeringHookManager(
            model=model, dial_to_layer={args.dial: L}, dial_to_vector={args.dial: v}
        )
        rows = []
        with mgr:
            for r in ratios:
                alpha = 0.0 if r == 0 else r * hnorm[L] / vn
                mgr.set_theta({args.dial: alpha})
                comp = []
                for pr, p in zip(prompts, probes):
                    lb = seq_logprob(model, tok, pr, p["biased_response"])
                    lh = seq_logprob(model, tok, pr, p["honest_response"])
                    comp.append(1.0 if lb > lh else 0.0)
                rows.append({"ratio": r, "alpha": round(alpha, 3), "biased_rate": mean(comp),
                             "std": pstdev(comp) if len(comp) > 1 else 0.0})
        base = rows[0]["biased_rate"]
        sigma = max(rows[0]["std"], 1e-6)
        best = max(rows, key=lambda x: x["biased_rate"])
        mono = all(rows[i + 1]["biased_rate"] >= rows[i]["biased_rate"] - 1e-9 for i in range(len(rows) - 1))
        out["layers"][L] = {
            "v_norm": round(vn, 3), "residual_norm": round(hnorm[L], 2),
            "rows": rows, "baseline_rate": base,
            "effect_at_r1.0_sigma": round((rows[-1]["biased_rate"] - base) / sigma, 3),
            "monotonic": mono,
        }
        print(f"L{L}: ||v||={vn:.2f} ||h||={hnorm[L]:.2f} | "
              + " ".join(f"r{x['ratio']}(a={x['alpha']}):{x['biased_rate']:.2f}" for x in rows)
              + f" | mono={mono}", flush=True)

    # --- coherence: greedy text at baseline vs strongest layer/ratio ---
    # pick layer+ratio with the largest biased_rate gain over its baseline
    bestL = max(layers, key=lambda L: max(x["biased_rate"] for x in out["layers"][L]["rows"]) - out["layers"][L]["baseline_rate"])
    bestrow = max(out["layers"][bestL]["rows"], key=lambda x: x["biased_rate"])
    v = raw[f"layer_{bestL}"]
    print(f"\n=== coherence check @ L{bestL}, ratio {bestrow['ratio']} (alpha={bestrow['alpha']}) ===", flush=True)
    gens = []
    mgr = SteeringHookManager(model=model, dial_to_layer={args.dial: bestL}, dial_to_vector={args.dial: v})
    for p, pr in list(zip(probes, prompts))[:2]:
        ids = tok(pr, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        texts = {}
        for label, a in [("baseline", 0.0), ("steered", bestrow["alpha"])]:
            with mgr:
                mgr.set_theta({args.dial: a})
                with torch.no_grad():
                    g = model.generate(ids, max_new_tokens=40, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                texts[label] = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
        gens.append({"biased_option": p["biased_response"], "honest_option": p["honest_response"], **texts})
        print(f"\n[probe] biased='{p['biased_response']}'\n  baseline: {texts['baseline'][:160]!r}\n  steered : {texts['steered'][:160]!r}", flush=True)
    out["coherence_check"] = {"layer": bestL, "ratio": bestrow["ratio"], "alpha": bestrow["alpha"], "samples": gens}
    Path(args.output_file).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
