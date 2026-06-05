"""Stage B -- persona-contrast steering vectors (Persona-Vectors / CAA recipe).

The default extractor (``byzminds_agent.steering.extract``) contrasts *situational
cue* pairs (``p_plus``/``p_minus`` differing by one inducing clause), read at the
assistant-to-speak position. The calibration sweep showed those directions are
weak -- including for the **sycophancy** validation anchor -- which is consistent
with the Stage-A finding that instruction-tuned models behaviorally *resist* bare
situational cues (M1b). When the anchor fails, the runbook prescribes re-deriving
against the Persona-Vectors system-prompt-contrast construction before extending.

This extractor does exactly that: it contrasts the **strong prompted persona**
(``render_persona(dial, "strong")`` -- the mechanism that *does* induce the
disposition in Stage A) against the neutral baseline prompt, averaging the
diff-of-means over the 30 probes per dial. The plus/minus prompts are reused
verbatim from ``steerB_evaluate._persona_prompt`` and ``steerB_calibrate._prompt``
so the extracted direction is precisely the persona-vs-baseline contrast that
calibration and evaluation later measure. Output uses the same
``steering_vectors/{dial}.pt`` schema, so steerB_calibrate / steerB_evaluate
consume it unchanged.

    python experiments/steerB_extract_persona.py \
        --model NousResearch/Meta-Llama-3.1-8B-Instruct --layers 8,12,16,20,24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent"))

from byzminds_agent import DIALS  # noqa: E402
from byzminds_agent.steering import extract as ex, vectors as vec  # noqa: E402
from experiments.steerB_calibrate import _load_probes, _prompt  # noqa: E402
from experiments.steerB_evaluate import _persona_prompt  # noqa: E402


def extract_persona_dial(dial, layers, model, tok, last_k):
    import torch

    probes = _load_probes(dial)
    plus_prompts = [_persona_prompt(tok, p, dial) for p in probes]   # strong persona system
    minus_prompts = [_prompt(tok, p) for p in probes]                # neutral baseline
    diffs = {l: [] for l in layers}
    for pp, mp in zip(plus_prompts, minus_prompts):
        reps = {}
        with torch.no_grad():
            for kind, text in (("plus", pp), ("minus", mp)):
                inputs = tok(text, return_tensors="pt").to(model.device)
                out = model(**inputs, output_hidden_states=True)
                for L in layers:
                    reps[(kind, L)] = out.hidden_states[L + 1][0, -last_k:, :].mean(dim=0)
        for L in layers:
            diffs[L].append((reps[("plus", L)] - reps[("minus", L)]).detach().cpu())
    vectors = {}
    for L in layers:
        v = torch.stack(diffs[L]).mean(dim=0)
        vectors[L] = v
        print(f"[{dial}] layer={L} ‖v‖={v.norm().item():.3f} (persona-contrast)", flush=True)
    return vectors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ex.DEFAULT_MODEL)
    ap.add_argument("--layers", default="8,12,16,20,24")
    ap.add_argument("--last-k-tokens", type=int, default=4)
    ap.add_argument("--dials", default=",".join(DIALS))
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    dials = [d for d in args.dials.split(",") if d in DIALS]

    import torch

    print(f"loading {args.model} for persona-contrast extraction ...", flush=True)
    tok, model = ex.load_model(args.model)
    vec.STEERING_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    for d in dials:
        vectors = extract_persona_dial(d, layers, model, tok, args.last_k_tokens)
        payload = {f"layer_{L}": vectors[L] for L in layers}
        payload["_metadata"] = {
            "dial": d, "construction": "persona_contrast", "unit_normalized": False,
            "last_k_tokens": args.last_k_tokens,
            "norms": {str(L): float(vectors[L].norm()) for L in layers},
        }
        torch.save(payload, vec.vectors_path(d))
        print(f"=== wrote {vec.vectors_path(d)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
