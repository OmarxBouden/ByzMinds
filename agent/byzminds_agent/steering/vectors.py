"""Load / save steering vectors as .pt files.

File schema per dial:

    agent/data/steering_vectors/{dial}.pt
        → dict[str, torch.Tensor]
        → keys "layer_<l>" for l in candidate_layers
"""

from __future__ import annotations

from pathlib import Path

from byzminds_agent import DIALS

STEERING_VECTORS_DIR = Path(__file__).resolve().parents[2] / "data" / "steering_vectors"


def vectors_path(dial: str) -> Path:
    if dial not in DIALS:
        raise ValueError(f"unknown dial {dial!r}")
    return STEERING_VECTORS_DIR / f"{dial}.pt"


def load_dial(dial: str):
    """Returns dict[layer_idx (int) → torch.Tensor]. Imports torch lazily."""
    import torch

    raw = torch.load(vectors_path(dial), map_location="cpu", weights_only=True)
    out: dict[int, "torch.Tensor"] = {}
    for k, v in raw.items():
        if not k.startswith("layer_"):
            continue
        out[int(k.removeprefix("layer_"))] = v
    return out


def load_all() -> dict[str, dict[int, "torch.Tensor"]]:
    out: dict[str, dict[int, "torch.Tensor"]] = {}
    for d in DIALS:
        if vectors_path(d).exists():
            out[d] = load_dial(d)
    return out
