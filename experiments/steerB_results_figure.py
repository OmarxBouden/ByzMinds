"""Stage B figure: held-out CAA steering vs prompted-persona shift, per dial.

Reads analysis/data/steerB_heldout.json (Llama) and, if present,
steerB_heldout_apertus.json (Apertus), and draws a per-dial comparison of the
prompted-persona shift in P(biased) against the shift produced by the held-out
response-contrast (CAA) steering vector. With both files it is a two-panel
cross-model figure (same dial order); otherwise a single Llama panel.
Writes paper/figures/fig12_steering.{pdf,png}.

    python experiments/steerB_results_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "analysis" / "data"
FIGS = REPO / "paper" / "figures"

WONG = {  # the paper's palette
    "deceive": "#D55E00", "authority": "#0072B2", "bandwagon": "#009E73",
    "sycophancy": "#CC79A7", "free_ride": "#F0E442", "collude": "#56B4E9",
}


def _panel(ax, by_dial, order, title):
    x = np.arange(len(order))
    w = 0.38
    prompted = [by_dial[d]["persona_shift"] for d in order]
    steered = [by_dial[d]["steered_shift"] for d in order]
    ax.bar(x - w / 2, prompted, w, label="prompted persona", color="#bdbdbd",
           edgecolor="#5a5a5a", linewidth=0.6)
    ax.bar(x + w / 2, steered, w, label="CAA steering (held-out)",
           color=[WONG[d] for d in order], edgecolor="#2a2a2a", linewidth=0.6)
    for i, d in enumerate(order):
        r = by_dial[d]["reproduction_frac"]
        ax.annotate("--" if r is None else f"{round(100 * r)}%",
                    (x[i] + w / 2, max(steered[i], 0)), textcoords="offset points",
                    xytext=(0, 3), ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#222")
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    lla = {r["dial"]: r for r in json.load(open(DATA / "steerB_heldout.json"))["results"]}
    order = sorted(lla, key=lambda d: lla[d]["reproduction_frac"] or -1, reverse=True)
    ape_path = DATA / "steerB_heldout_apertus.json"

    if ape_path.exists():
        ape = {r["dial"]: r for r in json.load(open(ape_path))["results"]}
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True)
        _panel(a0, lla, order, "Llama 3.1 8B")
        _panel(a1, ape, order, "Apertus 8B")
        a0.set_ylabel(r"$\Delta\,P(\mathrm{biased})$ vs baseline")
        a0.legend(frameon=False, loc="upper right", fontsize=9)
        fig.suptitle("Held-out activation steering vs prompted persona, per dial "
                     "(percent = fraction of the prompted shift reproduced out-of-sample)",
                     fontsize=10.5, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(7.6, 3.7))
        _panel(ax, lla, order, "")
        ax.set_ylabel(r"$\Delta\,P(\mathrm{biased})$ vs baseline")
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        ax.set_title("Activation steering reproduces the prompted-persona shift for 4/6 dials\n"
                     "(response-contrast CAA, extracted and evaluated on disjoint probe halves)",
                     fontsize=10)

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig12_steering.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig12_steering.png", dpi=200, bbox_inches="tight")
    print("wrote", FIGS / "fig12_steering.pdf", "(2-panel)" if ape_path.exists() else "(Llama only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
