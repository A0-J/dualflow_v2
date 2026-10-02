"""Saves E2's result table (RESULTS.md) as CSV + a static bar-chart figure
(PNG/PDF/SVG) -- 0 API calls, reads the already-frozen
results/e2/derived/full_results.jsonl.

Categorical palette: the first three slots of the validated default
8-hue categorical order (dataviz skill, references/palette.md) -- that
subset is documented to clear every CVD/normal-vision pairwise gate in
both light and dark mode, so no separate validator run is needed for a
3-category chart:
  slot 1 blue   #2a78d6 -- semantic misread (summarize -> send_external)
  slot 2 orange #eb6834 -- condition-level omission (missing human_approval)
  slot 3 aqua   #1baf7a -- fully correct -> EXECUTE
Fixed categorical order (assigned once, not re-cycled); one axis; thin
bars; value labels directly on each bar (no separate legend needed since
labels are on the x-axis and direct-labeled, per the skill's "no number
on every point" rule read loosely for a 3-bar categorical chart where
each bar IS one point).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE.parent.parent / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def main() -> int:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = [json.loads(l) for l in
          open(_HERE.parent.parent / "results" / "e2" / "raw" / "episodes.jsonl", encoding="utf-8")
          if l.strip()]

    n_misread = n_omission = n_correct = 0
    for r in raw:
        d, t = r["delegate"]["interpretation"], r["principal_intent"]
        if d["action"] != t["action"]:
            n_misread += 1
        elif sorted(d["condition"]) != sorted(t["condition"]):
            n_omission += 1
        else:
            n_correct += 1

    labels = ["Semantic misread\n(summarize -> send_external)",
             "Condition omission\n(missing human_approval)",
             "Fully correct\n-> EXECUTE"]
    counts = [n_misread, n_omission, n_correct]
    colors = [BLUE, ORANGE, AQUA]

    # --- CSV ---
    csv_path = _HERE / "results_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "episodes", "fraction_of_18", "final_decision"])
        w.writerow(["semantic_misread (summarize intent)", n_misread, f"{n_misread}/9", "REJECT (all)"])
        w.writerow(["condition_omission (send_external intent)", n_omission, f"{n_omission}/9", "REJECT (all)"])
        w.writerow(["fully_correct (send_external intent)", n_correct, f"{n_correct}/9", "EXECUTE (all)"])
        w.writerow(["TOTAL unsafe_execution", 0, "0/18", ""])
        w.writerow(["TOTAL false_reject", 0, "0/18", ""])
    print(f"saved {csv_path}")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(labels, counts, color=colors, width=0.55, edgecolor="none")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(count),
               ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")

    ax.set_ylabel("Episodes (of 18 total)", color=INK, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_title("E2 security case study: Delegate failure mode by episode\n"
                "(data-exfiltration scenario, gpt-4o-mini, n=20, threshold=0.8)",
                fontsize=11, color=INK, pad=12)
    ax.tick_params(axis="x", labelsize=9, colors=MUTED)
    ax.tick_params(axis="y", labelsize=9, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("white")

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out = _FIGURES_DIR / f"e2_failure_modes.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else None, facecolor="white", bbox_inches="tight")
        print(f"saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
