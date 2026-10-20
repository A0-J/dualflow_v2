"""Figures from the canonical samples for the follow-up work (PNG only, 0 API calls).

  followup_mismatch_by_model.png
      the 40 ambiguous delegation texts per Executor Agent model under forced
      choice (C2 cells, Planner Agent with task context, modal of 20 samples),
      split into irreversible misinterpretation, reversible semantic mismatch
      and the intended operation
  ambiguity_signals.png
      (a) share of ambiguous and of clear delegation texts each signal flags,
          with its AUROC
      (b) texts flagged per source delegation family

Design labels, not human judgements; the 80 texts are a development set.
Mismatch keeps one blue hue (irreversible = full step, reversible = light
step) and the intended operation is aqua, as in the clarification-option
figure; two-series bars use categorical slots 1-2 in fixed order.

    PYTHONPATH=src python experiments/ambiguity_screening/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_signals import auroc, load_banks, text_rows  # noqa: E402  (puts experiments/canonical on the path)
from common import DERIVED as CANONICAL_DERIVED, EXECUTOR_MODELS, ROOT  # noqa: E402
from dualflow.io import read_jsonl  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIGURES = ROOT / "figures"
INK, MUTED, AXIS, GRID, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9", "#ffffff"
BLUE, LIGHT_BLUE, ORANGE, AQUA = "#2a78d6", "#9ec5f4", "#eb6834", "#1baf7a"
RAMP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BAR = 0.34

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def _ink_on(fill: str) -> str:
    """White or ink, whichever contrasts more with the fill (WCAG relative luminance)."""
    def lum(h):
        def ch(c):
            c = int(c, 16) / 255
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = (ch(h[i:i + 2]) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    lf = lum(fill)
    return "white" if 1.05 / (lf + 0.05) > (lf + 0.05) / (lum(INK) + 0.05) else INK


def _x_axis(ax, label: str, xmax: float = 1.0) -> None:
    ticks = [t for t in (0, 0.25, 0.5, 0.75, 1.0) if t <= xmax + 1e-9]
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlabel(label, color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=9, length=0)


def mismatch_by_model() -> Path:
    cells = [c for c in read_jsonl(CANONICAL_DERIVED / "cells.jsonl")
             if c["case"] == "C2" and c["planner_condition"] == "context"]
    segments = (("Irreversible misinterpretation", BLUE, lambda c: c["irreversible_misinterpretation"]),
                ("Reversible semantic mismatch", LIGHT_BLUE,
                 lambda c: not c["intended"] and not c["irreversible_misinterpretation"]),
                ("Intended operation", AQUA, lambda c: c["intended"]))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for y, model in enumerate(EXECUTOR_MODELS):
        sub = [c for c in cells if c["executor_model"] == model]
        left = 0.0
        for _, color, test in segments:
            k = sum(map(test, sub))
            if k:
                ax.barh(y, k / len(sub), left=left, height=BAR, color=color, edgecolor=SURFACE, linewidth=0.8)
                if k / len(sub) >= 0.06:
                    ax.text(left + k / len(sub) / 2, y, str(k), ha="center", va="center", fontsize=8.5,
                            color=_ink_on(color))
            left += k / len(sub)
        mismatch = sum(not c["intended"] for c in sub)
        irreversible = sum(c["irreversible_misinterpretation"] for c in sub)
        ax.text(1.015, y, f"mismatch {mismatch}/{len(sub)}\nirreversible {irreversible}/{len(sub)}", ha="left",
                va="center", fontsize=8, color=MUTED, linespacing=1.3)
    ax.set_yticks(range(len(EXECUTOR_MODELS)))
    ax.set_yticklabels(EXECUTOR_MODELS)
    ax.invert_yaxis()
    _x_axis(ax, "share of the 40 ambiguous delegation texts")
    ax.set_title("Semantic mismatch under forced choice, by Executor Agent model", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold", pad=26)
    ax.legend(handles=[Patch(facecolor=c, label=name) for name, c, _ in segments], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, handlelength=1.2)
    fig.text(0.01, -0.04, "Canonical C2 cells (Planner Agent with task context): one modal proposal over 20 samples per "
             "delegation text and model. Differences between models,\nnot a trend over model development. "
             "gpt-5.5 and gpt-6.1-sol ran at provider-default sampling.", fontsize=8, color=MUTED, ha="left", va="top")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    out = FIGURES / "followup_mismatch_by_model.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def ambiguity_signals() -> Path:
    rows = text_rows(*load_banks())
    amb = [r for r in rows if r["ambiguous"]]
    clr = [r for r in rows if not r["ambiguous"]]
    signals = (("Executor Agent self-entropy\n(mean > 0.8 bits)", "self_entropy_mean", lambda r: r["self_entropy_mean"] > 0.8),
               ("Disagreement across the\n4 Executor Agent models", "cross_model_distinct", lambda r: r["cross_model_distinct"] >= 2),
               ("Executor Agent vs\ncontext-free Planner reading", "reader_disagreement", lambda r: r["reader_disagreement"] > 0),
               ("Planner Agent context shift\n(TV distance >= 0.5)", "context_shift_tv", lambda r: r["context_shift_tv"] >= 0.5),
               ("Planner Agent context shift\n(modal changes)", "context_shift_modal", lambda r: r["context_shift_modal"] > 0))

    fig, (ax, hx) = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    for y, (name, key, rule) in enumerate(signals):
        for offset, rs, color in ((-0.2, amb, BLUE), (0.2, clr, ORANGE)):
            k = sum(map(rule, rs))
            ax.barh(y + offset, k / len(rs), height=BAR, color=color)
            ax.text(k / len(rs) + 0.012, y + offset, f"{k}/{len(rs)}", ha="left", va="center", fontsize=8, color=INK)
        ax.text(1.32, y, f"AUROC {auroc([r[key] for r in amb], [r[key] for r in clr]):.2f}", ha="right",
                va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels([name for name, _, _ in signals])
    ax.invert_yaxis()
    _x_axis(ax, "share of delegation texts flagged")
    ax.set_xlim(0, 1.32)
    ax.set_title("(a) Texts flagged, ambiguous vs clear", loc="left", fontsize=10.5, color=INK,
                 fontweight="bold", pad=26)
    ax.legend(handles=[Patch(facecolor=BLUE, label="Ambiguous (40 texts)"),
                       Patch(facecolor=ORANGE, label="Clear (40 texts)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, handlelength=1.2)

    families = [f"{d}_{c}" for c in ("ambiguous", "clear") for d in ("audit", "hr", "incident", "legal")]
    grid = [[sum(rule(r) for r in rows if r["family"] == fam) for _, _, rule in signals] for fam in families]
    hx.imshow(grid, cmap=RAMP, vmin=0, vmax=10, aspect="auto")
    for i, row in enumerate(grid):
        for j, k in enumerate(row):
            hx.text(j, i, str(k), ha="center", va="center", fontsize=9, color="white" if k >= 6 else INK)
    hx.set_xticks(range(len(signals)))
    hx.set_xticklabels(["self-\nentropy", "across\nmodels", "vs context-\nfree reading", "context\nshift (TV)",
                        "context\nshift (modal)"], fontsize=8.5, color=MUTED)
    hx.set_yticks(range(len(families)))
    hx.set_yticklabels([f.replace("_", ", ") for f in families], fontsize=9, color=INK)
    hx.tick_params(length=0)
    for side in hx.spines.values():
        side.set_visible(False)
    hx.set_xticks([x - 0.5 for x in range(1, len(signals))], minor=True)
    hx.set_yticks([y - 0.5 for y in range(1, len(families))], minor=True)
    hx.grid(which="minor", color="white", linewidth=2)
    hx.tick_params(which="minor", length=0)
    hx.axhline(3.5, color=INK, linewidth=0.8)
    hx.set_title("(b) Texts flagged per source delegation family (of 10)", loc="left", fontsize=10.5,
                 color=INK, fontweight="bold", pad=26)

    without = {m: sum(len({r[f"p_exec__{o}"] for o in EXECUTOR_MODELS if o != m}) >= 2 for r in amb)
               for m in EXECUTOR_MODELS}
    fig.text(0.01, -0.03,
             "Post-hoc, 0 API calls, canonical samples; clear / ambiguous are design labels; the 80 texts are a "
             "development set (8 source delegations). Flag rules are fixed, illustrative\ncut points; AUROC is "
             "threshold-free. Disagreement across models depends on which models are included: leaving one out "
             "flags " + ", ".join(f"{without[m]}/40 (without {m})" for m in EXECUTOR_MODELS) + ".",
             fontsize=8, color=MUTED, ha="left", va="top")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    out = FIGURES / "ambiguity_signals.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (mismatch_by_model(), ambiguity_signals()):
        print(f"saved {path}")
