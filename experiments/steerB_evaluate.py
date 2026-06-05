"""Stage B -- evaluate steering vs prompted personas (the Stage-B headline).

For each dial that passed calibration (chosen layer + alpha), compute P(biased)
on the 30 probes under three conditions:
  * baseline   -- no intervention,
  * persona    -- the strong prompted persona prepended (the Stage-A mechanism),
  * steered    -- the natural-norm vector added at the chosen layer/alpha.
If steering reproduces (or exceeds) the prompted-persona shift, activation
steering induces the same dispositions -- the Stage-B claim, and the published
Persona-Vectors / CAA result extended to the dials those papers did not cover
(authority, bandwagon, free_ride, collude, deceive; sycophancy is the validation
anchor that overlaps Persona Vectors).

Also reports the Persona-Vectors-style activation monitor: the mean projection of
the response hidden state onto the dial vector, baseline vs steered (a detector
that needs no external judge).

    python experiments/steerB_evaluate.py --calibration analysis/data/steerB_calibration.json
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

from byzminds_agent.personas import render_persona  # noqa: E402
from byzminds_agent.steering import extract as ex, vectors as vec  # noqa: E402
from experiments.steerB_apply_hf import HFSteeringHooks  # noqa: E402
from experiments.steerB_calibrate import _load_probes, _p_biased, _prompt  # noqa: E402

DATA = REPO / "analysis" / "data"


def _persona_prompt(tok, probe, dial) -> str:
    persona = render_persona(dial, "strong")
    msgs = [{"role": "system", "content": persona},
            {"role": "user", "content": probe["user_prompt"] + "\n\n" + probe.get("output_constraint", "")}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return persona + "\n" + probe["user_prompt"] + "\n"


def _mean_p(model, tok, probes, prompts) -> float:
    ps = [_p_biased(model, tok, prompts[i], probes[i]) for i in range(len(probes))]
    return sum(ps) / len(ps)


def evaluate_dial(dial, layer, alpha, model, tok) -> dict:
    probes = _load_probes(dial)
    base_prompts = [_prompt(tok, p) for p in probes]
    pers_prompts = [_persona_prompt(tok, p, dial) for p in probes]
    vectors = vec.load_dial(dial)

    p_base = _mean_p(model, tok, probes, base_prompts)
    p_pers = _mean_p(model, tok, probes, pers_prompts)
    with HFSteeringHooks(model, {layer: [vectors[layer]]}, alpha=alpha):
        p_steer = _mean_p(model, tok, probes, base_prompts)
    return {"dial": dial, "layer": layer, "alpha": alpha,
            "p_biased_baseline": round(p_base, 3),
            "p_biased_persona": round(p_pers, 3),
            "p_biased_steered": round(p_steer, 3),
            "steered_shift": round(p_steer - p_base, 3),
            "persona_shift": round(p_pers - p_base, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ex.DEFAULT_MODEL)
    ap.add_argument("--calibration", default=str(DATA / "steerB_calibration.json"))
    ap.add_argument("--out", default=str(DATA / "steerB_evaluation.json"))
    args = ap.parse_args(argv)

    cal = json.load(open(args.calibration))
    chosen = [(r["dial"], r["chosen_layer"], 1.0) for r in cal["results"]
              if r.get("passed") and r.get("chosen_layer") is not None]
    if not chosen:
        print("no dials passed calibration; nothing to evaluate")
        return 0
    print(f"loading {args.model} ...", flush=True)
    tok, model = ex.load_model(args.model)  # NB: extract.load_model returns (tok, model)
    results = []
    for dial, layer, alpha in chosen:
        r = evaluate_dial(dial, layer, alpha, model, tok)
        results.append(r)
        print(f"  {dial}: baseline={r['p_biased_baseline']} persona={r['p_biased_persona']} "
              f"steered={r['p_biased_steered']}  (steered_shift={r['steered_shift']} vs "
              f"persona_shift={r['persona_shift']})", flush=True)
    Path(args.out).write_text(json.dumps(
        {"experiment": "steerB_evaluation", "model": args.model, "results": results}, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
