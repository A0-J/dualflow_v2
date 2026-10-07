"""E4 figure (PNG only, 0 API calls) from results/e4/derived/e4_cells.jsonl.

Three panels, each with its own axis and denominator (never one shared
axis across different denominators):
  (a) confident action misread on generated `summarize` paraphrases, by
      Delegate model -- one series, so one hue and no legend box
  (b) outcome-unsafe rate by Principal knowledge level (gpt-4o-mini
      Delegate, generated scenarios) -- semantic-only vs DualFlow
  (c) outcome false-reject rate, same cells, denominator = should-execute
Error bars are exact 95% Clopper-Pearson intervals. Colors are slots 1-2
of the validated reference categorical palette (blue, orange).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from dualflow.io import read_jsonl

from common import DELEGATE_MODELS, DERIVED, LEVELS, ROOT, clopper_pearson

FIGURES_DIR = ROOT / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
PRIMARY_MODEL = "gpt-4o-mini"
ARMS = (("semantic_grounding_only", "Semantic only", BLUE), ("dualflow", "DualFlow", ORANGE))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def _rate(k: int, n: int):
    lo, hi = clopper_pearson(k, n)
    p = k / n
    return p, (p - lo, hi - p)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _bar_label(ax, x, y, err_hi, text):
    ax.text(x, y + err_hi + 0.03, text, ha="center", va="bottom", fontsize=8.5, color=INK)


def main() -> int:
    cells = read_jsonl(DERIVED / "e4_cells.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})

    # (a) misread by Delegate model
    ax = axes[0]
    misread = [r for r in cells if r["level"] == "L2" and r["source"] == "generated" and r["intent"] == "summarize"]
    for i, model in enumerate(DELEGATE_MODELS):
        sub = [r for r in misread if r["delegate_model"] == model]
        k, n = sum(r["confident_action_misread"] for r in sub), len(sub)
        p, err = _rate(k, n)
        ax.bar(i, p, width=0.55, color=BLUE)
        ax.errorbar(i, p, yerr=[[err[0]], [err[1]]], color=INK, capsize=3, linewidth=1)
        _bar_label(ax, i, p, err[1], f"{k}/{n}")
    ax.set_xticks(range(len(DELEGATE_MODELS)))
    ax.set_xticklabels(DELEGATE_MODELS)
    _style(ax, "(a) Confident misread, by Delegate", "summarize paraphrases sent externally")

    # (b) outcome unsafe and (c) outcome false reject, by Principal level
    base = [r for r in cells if r["delegate_model"] == PRIMARY_MODEL and r["source"] == "generated"]
    width = 0.36
    for ax, metric, denom_key, title, ylabel in (
        (axes[1], "outcome_unsafe", None, "(b) Unsafe executions", "of all generated scenarios"),
        (axes[2], "outcome_false_reject", "should_execute_outcome", "(c) False rejects", "of should-execute scenarios"),
    ):
        for j, (arm, label, color) in enumerate(ARMS):
            for i, level in enumerate(LEVELS):
                sub = [r for r in base if r["level"] == level]
                n = sum(r[denom_key] for r in sub) if denom_key else len(sub)
                k = sum(r[f"{arm}__{metric}"] for r in sub)
                p, err = _rate(k, n)
                x = i + (j - 0.5) * width
                ax.bar(x, p, width=width - 0.03, color=color, label=label if i == 0 else None)
                ax.errorbar(x, p, yerr=[[err[0]], [err[1]]], color=INK, capsize=2.5, linewidth=1)
                _bar_label(ax, x, p, err[1], f"{k}/{n}")
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(["L0\nno private info", "L1\nsituational", "L2\nexplicit"])
        _style(ax, title, ylabel)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle(f"E4: Delegate cross-model and Principal knowledge level (Principal = gpt-4o-mini, "
                 f"panels b-c: Delegate = {PRIMARY_MODEL})", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "e4_information_asymmetry.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
