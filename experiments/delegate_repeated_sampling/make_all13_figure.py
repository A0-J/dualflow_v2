"""Saves the full 13-scenario result (safety + utility) as CSV + a
two-panel static figure (PNG/PDF/SVG) -- 0 API calls, reads already-
computed CSVs. Separate from `make_results_figure.py` (the frozen
9-scenario safety-only figure, tagged `v2.1-main-eval`) -- this one adds
the V2-V5 utility result alongside it rather than replacing it.

Left panel: unsafe executions / 13 (safety). Right panel: false rejects
/ 5 valid scenarios (utility) -- different denominator, so kept as a
separate panel rather than one shared axis (never plot two different-
denominator rates on the same bar height without saying so).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE.parent.parent / "figures"
_ALL13 = _HERE.parent.parent / "results" / "delegate_repeated_sampling" / "derived" / "five_arm_all13.csv"
_VALID = _HERE.parent.parent / "results" / "delegate_repeated_sampling" / "valid_scenarios" / "derived" / "five_arm_valid.csv"

BASELINE_COLOR = "#898781"
SAFE_COLOR = "#1baf7a"
WARN_COLOR = "#eb6834"
INK, GRID = "#0b0b0b", "#e1e0d9"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

_LABELS = {
    "no_verification": "No\nverif.",
    "entropy_only": "Entropy\nonly",
    "semantic_grounding_only": "Semantic\nonly",
    "authority_only": "Authority\nonly",
    "dualflow": "DualFlow",
}
_ORDER = ["no_verification", "entropy_only", "semantic_grounding_only", "authority_only", "dualflow"]


def main() -> int:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all13 = {r["arm"]: r for r in csv.DictReader(open(_ALL13, encoding="utf-8"))}
    valid_rows = list(csv.DictReader(open(_VALID, encoding="utf-8")))
    n13 = int(all13["dualflow"]["n"])
    n_valid = len({r["scenario_id"] for r in valid_rows})

    labels = [_LABELS[a] for a in _ORDER]
    unsafe = [int(all13[a]["unsafe"]) for a in _ORDER]
    false_reject = [sum(r[f"{a}_false_reject"] == "True" for r in valid_rows) for a in _ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    colors_unsafe = [SAFE_COLOR if a == "dualflow" else BASELINE_COLOR for a in _ORDER]
    bars = axes[0].bar(labels, unsafe, color=colors_unsafe, width=0.6)
    for bar, c in zip(bars, unsafe):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{c}/{n13}",
                     ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    axes[0].set_ylabel(f"Unsafe executions (of {n13})", color=INK, fontsize=10)
    axes[0].set_ylim(0, max(unsafe) + 1.5)
    axes[0].set_title("Safety (13 scenarios)", fontsize=11, color=INK, pad=10)

    colors_fr = [WARN_COLOR if c > 0 else BASELINE_COLOR for c in false_reject]
    bars2 = axes[1].bar(labels, false_reject, color=colors_fr, width=0.6)
    for bar, c in zip(bars2, false_reject):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{c}/{n_valid}",
                     ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    axes[1].set_ylabel(f"False rejects (of {n_valid} valid)", color=INK, fontsize=10)
    axes[1].set_ylim(0, max(false_reject) + 1.5)
    axes[1].set_title("Utility (5 valid scenarios, V1-V5)", fontsize=11, color=INK, pad=10)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=9, colors=INK)
        ax.tick_params(axis="y", labelsize=9, colors="#5b5952")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_facecolor("#fcfcfb")

    fig.suptitle("DualFlow 5-arm comparison -- safety vs. utility, different denominators",
                 fontsize=11.5, color=INK, y=1.02)
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        out = _FIGURES_DIR / f"delegate_five_arm_all13.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else None, facecolor="white", bbox_inches="tight")
        print(f"saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
