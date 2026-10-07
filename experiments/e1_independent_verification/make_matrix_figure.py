"""Saves E1's 2x2 (semantic x authority) result as CSV + a static grid
figure (PNG/PDF/SVG) -- 0 API calls, reads the already-frozen
results/e1/core_matrix.jsonl.

Two-color encoding only (status-like, not the 3-slot categorical set
E2's figure uses -- a different chart, a different job): EXECUTE vs
REJECT is the one thing color carries; WHY a REJECT happened (semantic,
authority, or both) is carried by text inside the cell, never by a 3rd
or 4th hue, per the skill's "text carries meaning the color channel
can't safely add a slot for" principle at small N.
  EXECUTE -> aqua   #1baf7a (same aqua used for "fully correct" in E2 --
             consistent color-to-meaning mapping across both figures)
  REJECT  -> a single neutral red-orange, reserved/status-like, not
             pulled from the categorical set: #d03b3b (status "critical")
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE.parent.parent / "figures"
_INPUT = _HERE.parent.parent / "results" / "e1" / "core_matrix.jsonl"

EXECUTE_COLOR = "#1baf7a"
REJECT_COLOR = "#d03b3b"
INK = "#0b0b0b"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

_SHORT_REASON = {
    "both_checks_passed": "normal execution",
    "semantic_failed": "blocked: unintended action",
    "authority_failed": "blocked: over scope",
    "semantic_and_authority_failed": "blocked: both violated",
}


def main() -> int:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(_INPUT, encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in rows}

    # --- CSV ---
    csv_path = _HERE / "core_matrix_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "semantic_ok", "authority_ok", "decision",
                   "expected_decision", "matches_expected", "fusion_reason"])
        for r in rows:
            w.writerow([r["id"], r["semantic_ok"], r["authority_ok"], r["decision"],
                       r["expected_decision"], r["matches_expected"], r["fusion_reason"]])
    print(f"saved {csv_path}")

    # --- 2x2 grid figure ---
    layout = {
        (True, True): "Q1_semantic_pass_authority_pass",
        (False, True): "Q2_semantic_fail_authority_pass",
        (True, False): "Q3_semantic_pass_authority_fail",
        (False, False): "Q4_semantic_fail_authority_fail",
    }

    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")

    col_labels = ["Authority allowed", "Authority denied"]
    row_labels = ["Semantic\ncorrect", "Semantic\nwrong"]

    for (semantic_ok, authority_ok), case_id in layout.items():
        r = by_id[case_id]
        col = 0 if authority_ok else 1
        row = 0 if semantic_ok else 1
        x, y = col, 1 - row  # row 0 (semantic correct) on top
        color = EXECUTE_COLOR if r["decision"] == "EXECUTE" else REJECT_COLOR
        box = FancyBboxPatch((x + 0.05, y + 0.05), 0.9, 0.9,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=0, facecolor=color, alpha=0.88)
        ax.add_patch(box)
        reason = _SHORT_REASON.get(r["fusion_reason"], r["fusion_reason"])
        ax.text(x + 0.5, y + 0.62, r["decision"], ha="center", va="center",
               fontsize=15, fontweight="bold", color="white")
        ax.text(x + 0.5, y + 0.38, reason, ha="center", va="center",
               fontsize=9.5, color="white")

    for i, label in enumerate(col_labels):
        ax.text(i + 0.5, 2.06, label, ha="center", va="bottom", fontsize=11,
               fontweight="bold", color=INK)
    for i, label in enumerate(row_labels):
        ax.text(-0.08, 1.5 - i, label, ha="right", va="center", fontsize=11,
               fontweight="bold", color=INK)

    ax.set_title("E1: independent semantic/authority verification\n"
                "(data-exfiltration scenario, deterministic, 0 API calls)",
                fontsize=12, color=INK, pad=28)
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    out = _FIGURES_DIR / "e1_matrix.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
