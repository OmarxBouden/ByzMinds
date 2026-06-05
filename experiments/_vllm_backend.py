"""Shared vLLM + SteeringHookManager bootstrap for Step 5 m5 experiments.

Per byzminds-step5-milestone5-brief.md §"Implementation work / Part 4":
each of experiments 008 / 009 / 011 / 013 / 014_probe / 014_headline
needs to replace its ``raise SystemExit("milestone 5")`` stub with a
real vLLM-served code path. The boilerplate (vLLM constructor, vector
load, hook-manager wiring, sampling params) is shared; per-experiment
loops live in their own files.

This module is **import-safe without vLLM / torch installed** — the
heavy imports happen inside the helper functions, so the experiments
remain runnable in ``--backend mock`` mode on Apple Silicon (the
researcher's plumbing path from milestone 4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent"))

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_LAYER = 16  # brief Question 1 pre-decided as (a) — L16 across dials
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
# Llama 3.1's native context is 131072; allocating its full KV cache OOMs a
# 40 GB card. Our probes / scenarios are short, so cap it. Override via the
# VLLM_MAX_MODEL_LEN env var (e.g. larger on an 80 GB GPU).
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_VECTORS_DIR = REPO / "agent" / "data" / "steering_vectors"


def load_llm(
    *,
    model_name: str = DEFAULT_MODEL,
    seed: int = 42,
    dtype: str = "bfloat16",
    enforce_eager: bool = True,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int | None = None,
    model_revision_sha: str = "",
):
    """Return a ``vllm.LLM`` configured per brief Decision 3.

    ``enforce_eager=True`` disables CUDA graphs so PyTorch forward
    hooks fire reliably; brief Question 2 pre-decided (a) on this
    despite the ~2-3× slowdown.

    NOTE on the engine: the SteeringHookManager registers PyTorch
    forward hooks on the in-process model module, which requires vLLM's
    legacy (V0) in-process engine. vLLM's V1 engine runs the model in a
    separate ``EngineCore`` process where those hooks cannot reach it,
    so steered runs must set ``VLLM_USE_V1=0`` before constructing the
    LLM. (Set it in the environment; vLLM reads it at import time.)
    """
    import os

    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "experiments._vllm_backend.load_llm: requires the byzminds-agent[serving] "
            "extras + an NVIDIA GPU. Local plumbing should use --backend=mock; the "
            "GPU box installs vLLM and runs this path."
        ) from exc
    if max_model_len is None:
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN))
    return LLM(
        model=model_name,
        dtype=dtype,
        enforce_eager=enforce_eager,
        seed=seed,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        revision=model_revision_sha or None,
    )


def sampling_params(*, seed: int = 42, max_tokens: int = 256):
    """Brief Decision 3 sampling settings: deterministic, top_p=1.0,
    top_k=-1, repetition_penalty=1.0, temperature=0.0."""
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "experiments._vllm_backend.sampling_params: vLLM not installed."
        ) from exc
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        seed=seed,
    )


def load_vectors(
    dials: list[str],
    *,
    vectors_dir: Path = DEFAULT_VECTORS_DIR,
    layer: int = DEFAULT_LAYER,
) -> dict[str, Any]:
    """Load steering vectors for ``dials`` at the given residual-stream
    layer. Returns ``dict[dial_name, torch.Tensor(hidden_dim,)]``.

    Aborts with a clear error if any expected ``.pt`` file is missing
    or the requested layer's tensor is absent. Researcher's
    ``byzminds-extract-vectors`` populates this directory.
    """
    import torch

    out: dict[str, Any] = {}
    for d in dials:
        p = vectors_dir / f"{d}.pt"
        if not p.exists():
            raise SystemExit(
                f"experiments._vllm_backend: steering vector for {d!r} missing at {p}.\n"
                "Run `byzminds-extract-vectors` first."
            )
        raw = torch.load(p, map_location="cpu", weights_only=True)
        key = f"layer_{layer}"
        if key not in raw:
            raise SystemExit(
                f"experiments._vllm_backend: vector for layer {layer} missing in {p} "
                f"(found keys: {sorted(raw.keys())}). Re-run extract.py with that "
                "layer in --layers, or pick one of the available layers."
            )
        out[d] = raw[key]
    return out


def make_hook_manager(
    llm,
    dials: list[str],
    *,
    layer_for_dial: dict[str, int] | None = None,
    default_layer: int = DEFAULT_LAYER,
    vectors_dir: Path = DEFAULT_VECTORS_DIR,
):
    """Build a SteeringHookManager wired with per-dial layer + vector.

    ``layer_for_dial`` is read from Experiment 008's calibration output
    if available (``analysis/data/calibration/{dial}.json::chosen_layer``).
    Until 008 runs on real model, the brief pre-decides default_layer=16
    everywhere (Question 1).
    """
    from byzminds_agent.steering.apply import SteeringHookManager

    layer_for_dial = layer_for_dial or {}
    dial_to_layer = {d: layer_for_dial.get(d, default_layer) for d in dials}
    # If layers differ across dials, load each dial's vector at its
    # assigned layer; otherwise a single layer load suffices.
    if len(set(dial_to_layer.values())) == 1:
        dial_to_vector = load_vectors(
            dials,
            vectors_dir=vectors_dir,
            layer=next(iter(dial_to_layer.values())),
        )
    else:
        # Per-dial layer assignment: each dial loads from its own layer.
        import torch

        dial_to_vector = {}
        for d, layer in dial_to_layer.items():
            p = vectors_dir / f"{d}.pt"
            raw = torch.load(p, map_location="cpu", weights_only=True)
            dial_to_vector[d] = raw[f"layer_{layer}"]
    return SteeringHookManager(
        llm=llm,
        dial_to_layer=dial_to_layer,
        dial_to_vector=dial_to_vector,
    )


def calibrated_layer_for_dial(
    dial: str,
    *,
    calibration_dir: Path = REPO / "analysis" / "data" / "calibration",
    fallback: int = DEFAULT_LAYER,
) -> int:
    """Read Experiment 008's calibration output (if present) to
    discover the chosen layer for ``dial``. Until 008 runs, returns
    ``fallback`` (the brief's pre-decided L16)."""
    p = calibration_dir / f"{dial}.json"
    if not p.exists():
        return fallback
    try:
        data = json.loads(p.read_text())
        layer = data.get("chosen_layer")
        if isinstance(layer, int):
            return layer
    except (json.JSONDecodeError, OSError):
        pass
    return fallback


def format_probe_prompt(probe: dict, tokenizer) -> str:
    """Render a probe into the prompt the agent model sees, using
    Llama 3.1's **native chat template** via the tokenizer.

    Replaces the prior handwritten formatter that emitted literal
    ``<|system|>`` / ``<|user|>`` tags — which are NOT Llama 3.1 special
    tokens and placed the model off-distribution (milestone-5 Gate 2
    root cause #2; see byzminds-m5-gate2-diagnosis.md).

    The probe's ``output_constraint`` (if present) is appended to the
    user turn. ``add_generation_prompt=True`` leaves the prompt ready
    for the assistant turn so logit/AB scoring reads the first response
    token.
    """
    messages: list[dict] = []
    if probe.get("system_prompt"):
        messages.append({"role": "system", "content": probe["system_prompt"]})
    user_content = probe.get("user_prompt") or probe.get("prompt", "")
    constraint = probe.get("output_constraint", "")
    if constraint:
        user_content = f"{user_content}\n\n{constraint}"
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def vllm_version_string() -> str:
    """Return the vLLM build identifier for cache key + manifest
    model_versions. Empty string on machines without vLLM (mock path)."""
    try:
        import vllm

        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return ""


def model_revision_for_manifest(model_revision_sha: str = "") -> dict[str, str]:
    """Build the ``ModelVersions`` JSON block (Step 5 manifest schema
    extension) for inclusion in the experiment's run record."""
    return {
        "agent_model": DEFAULT_MODEL,
        "agent_model_revision_sha": model_revision_sha,
        "agent_model_dtype": "bf16",
        "vllm_build_sha": vllm_version_string(),
    }
