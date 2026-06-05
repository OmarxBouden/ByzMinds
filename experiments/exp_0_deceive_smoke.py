#!/usr/bin/env python3
"""Experiment 0 — single-dial steering smoke test (HF transformers path).

Validates the milestone-5 Gate 2 natural-norm extraction fix on one dial
without vLLM. vLLM's modern multiprocess engine runs the model in a
separate process where the in-process ``SteeringHookManager`` forward
hook cannot reach it; HF ``transformers`` keeps the model in-process so
the *identical* steering vector / layer / alpha / scoring apply.

Scoring: full-sequence log-likelihood A/B comparison (lm-eval-harness
``loglikelihood`` style). NOTE we compare the summed log-prob of each
option's *whole* continuation, not just its first token — the probes'
biased/honest options share a prefix ("I publicly ...") so first-token
comparison cannot discriminate them.

Run (on the GPU box):
    cd ~/byzminds && PYTHONPATH=. agent/.venv/bin/python \\
        experiments/exp_0_deceive_smoke.py --dial deceive --layer 16 \\
        --alphas 0.0,0.5,1.0
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

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VECTORS_DIR = _REPO / "agent" / "data" / "steering_vectors"
PROBES_DIR = _REPO / "agent" / "data" / "probes"


def load_probes(dial: str, n: int) -> list[dict]:
    path = PROBES_DIR / f"{dial}.jsonl"
    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return out[:n]


def seq_logprob(model, tokenizer, prompt: str, continuation: str) -> float:
    """Summed log-prob of ``continuation`` tokens given ``prompt`` under
    teacher forcing. ``prompt`` already carries the chat template's BOS,
    so neither piece adds special tokens."""
    import torch

    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    c_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    full = torch.cat([p_ids, c_ids], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(full).logits  # (1, T, V)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    Lp = p_ids.shape[1]
    total = 0.0
    for i in range(c_ids.shape[1]):
        tok = int(c_ids[0, i])
        total += float(logprobs[0, Lp + i - 1, tok])
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/exp_0_deceive_smoke.py")
    ap.add_argument("--dial", default="deceive")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--alphas", default="0.0,0.5,1.0")
    ap.add_argument("--n-probes", type=int, default=30)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-file", default="/tmp/exp_0_results.json")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from byzminds_agent.steering.apply import SteeringHookManager

    alphas = [float(a) for a in args.alphas.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    raw = torch.load(VECTORS_DIR / f"{args.dial}.pt", map_location="cpu", weights_only=False)
    v = raw[f"layer_{args.layer}"]
    meta = raw.get("_metadata", {})
    print(
        f"[{args.dial}] vector L{args.layer}: ‖v‖={float(v.float().norm()):.3f} "
        f"unit_normalized={meta.get('unit_normalized')} "
        f"chat_template_applied={meta.get('chat_template_applied')}",
        flush=True,
    )

    probes = load_probes(args.dial, args.n_probes)
    print(f"[{args.dial}] scoring {len(probes)} probes × {len(alphas)} alphas", flush=True)

    manager = SteeringHookManager(
        model=model,
        dial_to_layer={args.dial: args.layer},
        dial_to_vector={args.dial: v},
    )

    results: dict[str, dict] = {}
    with manager:
        for alpha in alphas:
            manager.set_theta({args.dial: alpha})
            compliances: list[float] = []
            for probe in probes:
                prompt = format_probe_prompt(probe, tok)
                lp_biased = seq_logprob(model, tok, prompt, probe["biased_response"])
                lp_honest = seq_logprob(model, tok, prompt, probe["honest_response"])
                compliances.append(1.0 if lp_biased > lp_honest else 0.0)
            results[f"{alpha:g}"] = {
                "biased_rate": mean(compliances),
                "std": pstdev(compliances) if len(compliances) > 1 else 0.0,
                "n": len(compliances),
            }
            print(
                f"  alpha={alpha:<4g} biased_rate={results[f'{alpha:g}']['biased_rate']:.3f} "
                f"(n={len(compliances)})",
                flush=True,
            )

    base = results[f"{alphas[0]:g}"]
    top = results[f"{alphas[-1]:g}"]
    sigma = max(base["std"], 1e-6)
    effect = (top["biased_rate"] - base["biased_rate"]) / sigma
    monotonic = all(
        results[f"{alphas[i + 1]:g}"]["biased_rate"]
        >= results[f"{alphas[i]:g}"]["biased_rate"] - 1e-9
        for i in range(len(alphas) - 1)
    )
    summary = {
        "dial": args.dial,
        "layer": args.layer,
        "model": args.model,
        "vector_natural_norm": float(v.float().norm()),
        "per_alpha": results,
        "effect_size_at_alpha_top": effect,
        "alpha_top": alphas[-1],
        "monotonic": monotonic,
        "scoring": "full-sequence loglikelihood A/B",
    }
    Path(args.output_file).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(
        f"\n=== GATE: effect@alpha={alphas[-1]:g} = {effect:.3f}σ  "
        f"(baseline rate {base['biased_rate']:.2f} → {top['biased_rate']:.2f}, "
        f"monotonic={monotonic}) ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
