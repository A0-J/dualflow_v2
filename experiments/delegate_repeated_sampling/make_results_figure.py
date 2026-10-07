"""Saves the combined 9-scenario 5-arm comparison as CSV + a static bar
figure (PNG/PDF/SVG) -- 0 API calls, reads the already-computed
results/delegate_repeated_sampling/derived/five_arm_combined.csv.

This is the headline result from RESULTS.md's "Combined 5-arm
comparison" section: unsafe-execution count out of 9 scenarios, per
method. A single muted color marks the 4 baseline methods (each fails a
different, specific subset -- that's the point, not a ranking to encode
in hue); DualFlow gets the same "safe/good" aqua used for EXECUTE
elsewhere in this pack (E1/E2 figures), since it's the only arm that
reaches zero. This is a status distinction (safe vs. not), not a
4-way categorical one, so one baseline color + one accent color is used
rather than 5 cycled categorical hues.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE.parent.parent / "figures"
_INPUT = _HERE.parent.parent / "results" / "delegate_repeated_sampling" / "derived" / "five_arm_combined.csv"

BASELINE_COLOR = "#898781"  # muted -- the 4 methods that each leave a gap
SAFE_COLOR = "#1baf7a"      # aqua -- same "EXECUTE/correct" color as E1/E2, here meaning "0 unsafe"
INK, GRID = "#0b0b0b", "#e1e0d9"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

_LABELS = {
    "no_verification": "No\nverification",
    "entropy_only": "Entropy\nonly",
    "semantic_grounding_only": "Semantic\ngrounding only",
    "authority_only": "Authority\nonly",
    "dualflow": "DualFlow",
}
_ORDER = ["no_verification", "entropy_only", "semantic_grounding_only", "authority_only", "dualflow"]


def main() -> int:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = {r["arm"]: r for r in csv.DictReader(open(_INPUT, encoding="utf-8"))}
    n = int(rows["dualflow"]["n"])

    labels = [_LABELS[a] for a in _ORDER]
    unsafe = [int(rows[a]["unsafe"]) for a in _ORDER]
    colors = [SAFE_COLOR if a == "dualflow" else BASELINE_COLOR for a in _ORDER]

    csv_path = _HERE / "combined_unsafe_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n", "unsafe", "false_reject", "correct_execute"])
        for a in _ORDER:
            r = rows[a]
            w.writerow([a, r["n"], r["unsafe"], r["false_reject"], r["correct_execute"]])
    print(f"saved {csv_path}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, unsafe, color=colors, width=0.6)
    for bar, count in zip(bars, unsafe):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{count}/{n}",
                ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")

    ax.set_ylabel(f"Unsafe executions (of {n} scenarios)", color=INK, fontsize=10)
    ax.set_ylim(0, max(unsafe) + 1.5)
    ax.set_title("Combined 5-arm comparison -- 6 semantic-risk + 3 authority-risk scenarios\n"
                 "(controlled proof-of-mechanism, not a statistical benchmark -- see RESULTS.md)",
                 fontsize=10.5, color=INK, pad=14)
    ax.tick_params(axis="x", labelsize=9.5, colors=INK)
    ax.tick_params(axis="y", labelsize=9, colors="#5b5952")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    out = _FIGURES_DIR / "delegate_five_arm_combined.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
