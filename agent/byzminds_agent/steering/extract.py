"""Offline contrast-pair → steering-vector extraction.

For each dial d in DIALS:
  1. Load agent/data/contrast_pairs/{dial}.jsonl (researcher-owned).
     Each line: {"plus": "...", "minus": "..."} — text pair that
     differs only in the labeled dimension.
  2. Render each side through the tokenizer's **native chat template**
     (Llama 3.1 format) and forward it through the model with
     residual-stream activations recorded at every candidate layer over
     the last K tokens.
  3. Compute v_d^(l) = mean_i(h+_i - h-_i) at each layer l. The vector
     is kept at its **natural diff-of-means norm** — it is NOT
     unit-normalized. (Unit-normalizing here was milestone-5 Gate 2
     root cause #1: a unit vector added with α≤1 is a sub-percent
     perturbation of a residual stream whose norm is in the tens-to-
     hundreds. See byzminds-m5-gate2-diagnosis.md.)
  4. Save to agent/data/steering_vectors/{dial}.pt as
     {"layer_<l>": tensor[hidden_dim], ..., "_metadata": {...}}.
     The ``_metadata`` block is the audit trail for the fix
     (``unit_normalized: False``, per-layer natural norms,
     ``chat_template_applied``).

Requires the ``[serving]`` extras. The script stops with a clear error
if the researcher-owned .jsonl files are absent; it does not fabricate
them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from byzminds_agent import DIALS

CONTRAST_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "contrast_pairs"
STEERING_VECTORS_DIR = Path(__file__).resolve().parents[2] / "data" / "steering_vectors"
DEFAULT_LAYERS = [8, 12, 16, 20, 24]
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def check_contrast_pair_files() -> list[Path]:
    """Returns a list of missing files. Empty list ⇒ all present."""
    missing: list[Path] = []
    for dial in DIALS:
        p = CONTRAST_PAIRS_DIR / f"{dial}.jsonl"
        if not p.exists():
            missing.append(p)
    return missing


def _abort_if_missing() -> None:
    missing = check_contrast_pair_files()
    if missing:
        print(
            "byzminds-extract-vectors: contrast pair files missing. "
            "Per Step 3 judgment call 3 these are researcher-owned and "
            "must be authored before extraction. Missing:\n  - "
            + "\n  - ".join(str(p) for p in missing),
            file=sys.stderr,
        )
        sys.exit(2)


def parse_pairs(jsonl_path: Path) -> list[tuple[str, str]]:
    """Read a {plus, minus} JSONL file. Strict: missing fields fail loudly."""
    pairs: list[tuple[str, str]] = []
    with jsonl_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            # Provisioned contrast pairs use p_plus/p_minus (alongside
            # pair_id/notes/context_token_count); accept the legacy
            # plus/minus names too.
            plus = obj.get("plus", obj.get("p_plus"))
            minus = obj.get("minus", obj.get("p_minus"))
            if plus is None or minus is None:
                raise ValueError(
                    f"{jsonl_path}:{line_no}: missing 'plus'/'minus' "
                    "(or 'p_plus'/'p_minus') field"
                )
            pairs.append((plus, minus))
    if not pairs:
        raise ValueError(f"{jsonl_path}: empty contrast pair file")
    return pairs


def chat_template_available(tokenizer) -> bool:
    """True if the tokenizer carries a native chat template.

    Llama 3.1 Instruct does; base models / some test tokenizers don't.
    When absent we fall back to raw text and record the fact in the
    metadata so the audit trail stays honest.
    """
    return bool(getattr(tokenizer, "chat_template", None))


def render_for_extraction(tokenizer, text: str) -> str:
    """Render one contrast-pair side into the prompt the model sees.

    Routes through the tokenizer's native chat template as a single
    user turn with ``add_generation_prompt=True`` so the extraction
    activations are taken at the assistant-to-speak position — the
    same regime in which the vector is later applied. This fixes
    milestone-5 Gate 2 root cause #2 (the prior path forwarded raw
    text, off-distribution for an instruct model).

    Falls back to raw ``text`` if the tokenizer has no chat template.
    """
    if not chat_template_available(tokenizer):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_save_payload(
    vectors: dict[int, "torch.Tensor"],
    *,
    model_id: str,
    last_k_tokens: int,
    chat_template_applied: bool,
) -> dict:
    """Assemble the on-disk dict: per-layer tensors under ``layer_<l>``
    keys plus a ``_metadata`` block.

    ``_metadata`` is deliberately non-tensor and keyed by a name that
    does not start with ``layer_`` so ``steering.vectors.load_dial``
    (which filters on that prefix) ignores it — backward compatible
    with every downstream consumer.
    """
    natural_norms = {
        str(layer): float(v.float().norm().item()) for layer, v in vectors.items()
    }
    payload: dict = {f"layer_{layer}": v for layer, v in vectors.items()}
    payload["_metadata"] = {
        "model_id": model_id,
        "natural_norms": natural_norms,
        "extraction_method": "diff-of-means (CAA), natural-norm",
        "unit_normalized": False,  # CRITICAL: the Gate 2 root-cause-1 fix
        "chat_template_applied": chat_template_applied,
        "last_k_tokens": last_k_tokens,
    }
    return payload


def load_model(model_name: str):
    """Load tokenizer + model once. Hoisted out of ``extract_one_dial``
    per milestone 5 brief Part 1 — the previous per-dial load-and-
    forget pattern OOMed at dial 4 because old model weights weren't
    released before the next allocation.

    Caller is responsible for the cleanup (``del model;
    torch.cuda.empty_cache()``); see ``main`` for the pattern.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, output_hidden_states=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tok, model


def extract_one_dial(
    dial: str,
    layers: list[int],
    *,
    model,
    tokenizer,
    last_k_tokens: int = 4,
) -> dict[int, "torch.Tensor"]:
    """Extract per-layer steering vectors for one dial against the
    pre-loaded model + tokenizer. Caller hoists ``load_model`` to
    process start so weights are only allocated once across all dials.
    """
    import torch

    pairs = parse_pairs(CONTRAST_PAIRS_DIR / f"{dial}.jsonl")
    print(f"[{dial}] loaded {len(pairs)} contrast pairs", flush=True)

    diffs_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for plus, minus in pairs:
        with torch.no_grad():
            for kind, text in [("plus", plus), ("minus", minus)]:
                # Render through Llama 3.1's native chat template so the
                # activations are sampled in-distribution (Gate 2 fix #2).
                rendered = render_for_extraction(tokenizer, text)
                inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
                out = model(**inputs, output_hidden_states=True)
                # out.hidden_states is a tuple of (n_layers+1) tensors
                # shape (1, seq_len, hidden_dim). Index 0 = embeddings,
                # index 1..n = decoder layer outputs.
                for layer in layers:
                    h = out.hidden_states[layer + 1]  # +1 for embedding
                    last = h[0, -last_k_tokens:, :].mean(dim=0)  # (hidden_dim,)
                    if kind == "plus":
                        diffs_per_layer[layer].append(last)
                    else:
                        diffs_per_layer[layer][-1] = diffs_per_layer[layer][-1] - last

    vectors: dict[int, torch.Tensor] = {}
    for layer, diffs in diffs_per_layer.items():
        stacked = torch.stack(diffs)  # (n_pairs, hidden_dim)
        # Mean of differences, kept at its NATURAL norm — no unit
        # normalization (Gate 2 root cause #1). The norm reflects the
        # true activation gap and is what makes α∈[0,1] a meaningful
        # perturbation of the residual stream.
        v = stacked.mean(dim=0)
        vectors[layer] = v.detach().cpu()
        print(f"[{dial}] layer={layer} ‖v‖={v.norm().item():.3f} (natural)", flush=True)

    return vectors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="byzminds-extract-vectors")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--layers",
        default=",".join(str(l) for l in DEFAULT_LAYERS),
        help="candidate residual-stream layer indices",
    )
    p.add_argument("--last-k-tokens", type=int, default=4)
    p.add_argument(
        "--dials",
        default=",".join(DIALS),
        help="subset of dials to extract (comma-separated)",
    )
    args = p.parse_args(argv)
    _abort_if_missing()

    STEERING_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    dials = [d for d in args.dials.split(",") if d.strip()]
    for d in dials:
        if d not in DIALS:
            raise SystemExit(f"unknown dial {d!r}")

    import gc

    import torch  # local import: only reachable if --dials non-empty

    # Load the model once at process start. Per milestone 5 brief
    # Decision 6, all subsequent dials share this object — the OOM bug
    # was caused by per-dial AutoModelForCausalLM.from_pretrained
    # calls without freeing the previous weights.
    print(f"loading {args.model} once for all {len(dials)} dial(s)", flush=True)
    tokenizer, model = load_model(args.model)
    templated = chat_template_available(tokenizer)
    if not templated:
        print(
            f"WARNING: tokenizer for {args.model!r} has no chat template; "
            "extracting from raw text (metadata will record chat_template_applied=False)",
            flush=True,
        )
    try:
        for dial in dials:
            out_path = STEERING_VECTORS_DIR / f"{dial}.pt"
            print(f"=== extracting {dial} → {out_path}", flush=True)
            vectors = extract_one_dial(
                dial,
                layers,
                model=model,
                tokenizer=tokenizer,
                last_k_tokens=args.last_k_tokens,
            )
            payload = build_save_payload(
                vectors,
                model_id=args.model,
                last_k_tokens=args.last_k_tokens,
                chat_template_applied=templated,
            )
            torch.save(payload, out_path)
            norms = payload["_metadata"]["natural_norms"]
            print(f"=== wrote {out_path}  natural_norms={norms}", flush=True)
    finally:
        # Explicit cleanup so a long-running parent process (e.g.,
        # a Jupyter kernel or a RunAI driver) sees memory released.
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
