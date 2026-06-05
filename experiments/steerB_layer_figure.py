"""Stage B layer-localization figure: response-contrast steering effect by layer.

Reads the response-contrast calibration JSONs (Llama + Apertus) and draws, per
model, a dial x layer heatmap of the effect size (sigma) of steering at each
candidate layer. Shows that the steerable dispositions localize to mid-network
(layer 12 of 32) consistently across both architectures, decaying to ~0 by the
late layers. Cells clearing the 2-sigma gate are outlined.
Writes paper/figures/fig13_steering_layers.{pdf,png}.

    python experiments/steerB_layer_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "analysis" / "data"
FIGS = REPO / "paper" / "figures"

LAYERS = [8, 12, 16, 20, 24]
DORDER = ["collude", "deceive", "authority", "free_ride", "sycophancy", "bandwagon"]
GATE = 2.0


def _matrix(path):
    res = {r["dial"]: r for r in json.load(open(path))["results"]}
    return np.array([[res[d]["per_layer"][str(L)]["effect_sigma"] for L in LAYERS] for d in DORDER])


def _panel(ax, M, title, show_ylabels):
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=8, aspect="auto")
    ax.set_xticks(range(len(LAYERS))); ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_yticks(range(len(DORDER)))
    ax.set_yticklabels(DORDER if show_ylabels else [""] * len(DORDER))
    ax.set_title(title, fontsize=10)
    for i in range(len(DORDER)):
        for j in range(len(LAYERS)):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 4.5 else "black")
            if v >= GATE:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#39d98a", linewidth=2.0))
    ax.set_xlabel("steering layer (of 32)")
    return im


def main() -> int:
    LL = _matrix(DATA / "steerB_calibration_response.json")
    AP = _matrix(DATA / "steerB_calibration_response_apertus.json")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    _panel(a0, LL, "Llama 3.1 8B", True)
    im = _panel(a1, AP, "Apertus 8B", False)
    cbar = fig.colorbar(im, ax=[a0, a1], fraction=0.046, pad=0.04)
    cbar.set_label(r"steering effect ($\sigma$, capped at 8)")
    fig.suptitle("Where the dispositions live: response-contrast steering effect by layer "
                 "(green = clears the 2$\\sigma$ gate)", fontsize=10.5, y=1.0)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig13_steering_layers.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig13_steering_layers.png", dpi=200, bbox_inches="tight")
    print("wrote", FIGS / "fig13_steering_layers.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
