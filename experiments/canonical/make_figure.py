"""Results figure for the canonical C1-C4 evaluation (PNG only, 0 API calls).

  (a) unsafe-execution rate, comparison method x case, every cell labeled
      (a matrix rather than bars so that zero values stay visible)
  (b) irreversible misinterpretation of ambiguous delegation variants by
      Executor Agent model, split into
      confident (entropy <= 0.8) and not confident
Planner with task context; (a) is pooled over Executor models. False
rejects are reported in the text table. Sequential single-hue ramp for the
magnitude in (a); one series hue plus a lighter step of it in (b).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from dualflow.io import read_jsonl

from analyze import CASES, METHODS
from common import DERIVED, EXECUTOR_MODELS, ROOT, THRESHOLD

FIGURES = ROOT / "figures"
NAMES = {"no_verification": "No Verification", "self_consistency": "Consistency-based Gate",
         "semantic_only": "Semantic Flow-Only", "authorization_only": "Authorization Flow-Only",
         "dualflow": "DualFlow"}
CASE_LABELS = ["C1\nclear,\nauthorized", "C2\nambiguous, misinterpretation\nauthorized",
               "C3\nclear,\nnot authorized", "C4\nambiguous, misinterpretation\nnot authorized"]
RAMP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BLUE, LIGHT_BLUE = "#2a78d6", "#9ec5f4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def main() -> int:
    cells = [r for r in read_jsonl(DERIVED / "cells.jsonl") if r["planner_condition"] == "context"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    values = []
    for m in METHODS:
        row = []
        for case in CASES:
            sub = [r for r in cells if r["case"] == case]
            k = sum(r[f"{m}__unsafe"] for r in sub)
            row.append((k, len(sub)))
        values.append(row)
    rates = [[k / n for k, n in row] for row in values]
    ax.imshow(rates, cmap=RAMP, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(values):
        for j, (k, n) in enumerate(row):
            color = "white" if k / n > 0.55 else INK
            weight = "bold" if METHODS[i] == "dualflow" else "normal"
            ax.text(j, i, f"{k / n:.0%}\n{k}/{n}", ha="center", va="center", fontsize=9, color=color, fontweight=weight)
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels(CASE_LABELS, fontsize=8.8, color=MUTED)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([NAMES[m] for m in METHODS], fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(CASES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(METHODS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("(a) Unsafe Execution, by approach and condition", loc="left", fontsize=11, color=INK, fontweight="bold")

    ax = axes[1]
    amb = [r for r in cells if r["case"] == "C2"]  # each ambiguous delegation text once per model
    for i, model in enumerate(EXECUTOR_MODELS):
        sub = [r for r in amb if r["executor_model"] == model]
        confident = sum(r["misread"] and r["executor_entropy"] <= THRESHOLD for r in sub)
        hesitant = sum(r["misread"] and r["executor_entropy"] > THRESHOLD for r in sub)
        ax.bar(i, confident / len(sub), width=0.55, color=BLUE,
               label="permitted by Consistency-based Gate (entropy ≤ 0.8 bits)" if i == 0 else None)
        ax.bar(i, hesitant / len(sub), width=0.55, bottom=confident / len(sub), color=LIGHT_BLUE,
               label="blocked by Consistency-based Gate" if i == 0 else None)
        ax.text(i, (confident + hesitant) / len(sub) + 0.03, f"{confident + hesitant}/{len(sub)}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(EXECUTOR_MODELS)))
    ax.set_xticklabels(EXECUTOR_MODELS, fontsize=9, color=MUTED)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("ambiguous delegation messages with\nirreversible misinterpretation", color=MUTED, fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")
    ax.set_title("(b) Irreversible misinterpretation, by Executor model", loc="left", fontsize=11, color=INK, fontweight="bold")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "canonical_results.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
