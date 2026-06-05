"""Stage B -- response-contrast steering vectors (CAA recipe, response positions).

Two earlier constructions failed the gate: the situational-cue contrast
(``steering.extract``) and the persona-prompt contrast (``steerB_extract_persona``)
both read activations at the assistant-to-speak position -- the final
chat-template tokens, which are *identical* between the plus/minus sides and
differ only through attention, yielding weak directions.

This extractor follows Contrastive Activation Addition (CAA, ``rimsky2024caa``)
literally: for each probe it appends the **biased response** (plus) and the
**honest response** (minus) to the same prompt, and reads the diff-of-means over
the *response* token positions -- where the two sides actually differ in content.
This is the direction that distinguishes producing a biased vs honest answer,
read where it manifests. Averaged over the 30 probes per dial. Output uses the
same ``steering_vectors/{dial}.pt`` schema so steerB_calibrate / steerB_evaluate
consume it unchanged.

    python experiments/steerB_extract_response.py \
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


def _response_reps(model, tok, prompt_text, response_text, layers, last_k):
    """Mean hidden state over the response-token span (the appended answer)."""
    import torch

    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids
    full_ids = tok(prompt_text + response_text, return_tensors="pt").to(model.device)
    p_len = prompt_ids.shape[1]
    with torch.no_grad():
        out = model(**full_ids, output_hidden_states=True)
    reps = {}
    for L in layers:
        h = out.hidden_states[L + 1][0]            # (seq, hidden)
        resp = h[p_len:, :]                         # response tokens only
        if resp.shape[0] == 0:                      # response merged into last prompt tok
            resp = h[-1:, :]
        if last_k > 0:
            resp = resp[-last_k:, :]
        reps[L] = resp.mean(dim=0)
    return reps


def extract_response_dial(dial, layers, model, tok, last_k):
    import torch

    probes = _load_probes(dial)
    diffs = {l: [] for l in layers}
    for p in probes:
        prompt = _prompt(tok, p)
        rp = _response_reps(model, tok, prompt, p["biased_response"], layers, last_k)
        rm = _response_reps(model, tok, prompt, p["honest_response"], layers, last_k)
        for L in layers:
            diffs[L].append((rp[L] - rm[L]).detach().cpu())
    vectors = {}
    for L in layers:
        v = torch.stack(diffs[L]).mean(dim=0)
        vectors[L] = v
        print(f"[{dial}] layer={L} ‖v‖={v.norm().item():.3f} (response-contrast)", flush=True)
    return vectors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ex.DEFAULT_MODEL)
    ap.add_argument("--layers", default="8,12,16,20,24")
    ap.add_argument("--last-k-tokens", type=int, default=0, help="0 = mean over all response tokens")
    ap.add_argument("--dials", default=",".join(DIALS))
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    dials = [d for d in args.dials.split(",") if d in DIALS]

    import torch

    print(f"loading {args.model} for response-contrast extraction ...", flush=True)
    tok, model = ex.load_model(args.model)
    vec.STEERING_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    for d in dials:
        vectors = extract_response_dial(d, layers, model, tok, args.last_k_tokens)
        payload = {f"layer_{L}": vectors[L] for L in layers}
        payload["_metadata"] = {
            "dial": d, "construction": "response_contrast_caa", "unit_normalized": False,
            "last_k_tokens": args.last_k_tokens,
            "norms": {str(L): float(vectors[L].norm()) for L in layers},
        }
        torch.save(payload, vec.vectors_path(d))
        print(f"=== wrote {vec.vectors_path(d)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
