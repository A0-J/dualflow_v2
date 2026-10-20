"""Stage-1 figures for the clarification-option experiment (PNG only, 0 API calls).

clarification_option_stage1.png
  (a) ambiguous delegations, by Executor Agent model and condition: forced
      choice (canonical, modal of the first 5 samples), the option to ask,
      and the option with an explicit ambiguity cue
  (b) the option to ask, by source delegation family and model
clarification_option_format_failures.png
  first-attempt output-format failures, by model, delegation type and prompt,
  split into units valid on a retry and units with no valid attempt

Each bar is 100% of its (delegation text, model) pairs, split into the three
pair categories of analysis A (primary). Colors are categorical slots 1-3 of
the reference palette in fixed order; every segment that fits carries its
count, which also covers the low-contrast third slot.

    PYTHONPATH=src python experiments/clarification_option/make_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# This experiment's analyze.py first: it puts experiments/canonical (which has its
# own analyze.py) on the path.
from analyze import DERIVED, K, PARSE_FAILURES, RAW, forced_choice  # noqa: E402
from common import ROOT, delegation_texts  # noqa: E402
from dualflow.flows import Proposal  # noqa: E402
from dualflow.io import read_jsonl  # noqa: E402
from scenarios import DOMAINS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = ROOT / "figures" / "clarification_option_stage1.png"
MODELS = ("gpt-4o-mini", "gpt-4.1")
# stack order = legend order = palette slot order
SEGMENTS = (("executed_mismatch", "Unqueried semantic mismatch", "#2a78d6"),
            ("asked", "Asked the Planner Agent", "#eb6834"),
            ("executed_intended", "Intended operation proposed", "#1baf7a"))
CONDITIONS = (("forced", "Forced choice"), ("neutral", "Option to ask"), ("explicit", "Option + explicit cue"))
FAMILIES = ("audit", "hr", "incident", "legal")
FAMILY_NAMES = {"audit": "Audit", "hr": "HR", "incident": "Incident", "legal": "Legal"}
INK, MUTED, AXIS, GRID, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9", "#ffffff"
BAR = 0.34

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def _luminance(hex_color: str) -> float:
    def channel(c):
        c = int(c, 16) / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(hex_color[i:i + 2]) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _label_color(fill: str) -> str:
    """White or ink, whichever contrasts more with the fill."""
    lum = _luminance(fill)
    return "white" if (1.05 / (lum + 0.05)) > ((lum + 0.05) / (_luminance(INK) + 0.05)) else INK


def _bar(ax, y, counts: dict[str, int]) -> None:
    n = sum(counts.values())
    left = 0.0
    for key, _, color in SEGMENTS:
        share = counts.get(key, 0) / n
        if share:
            ax.barh(y, share, left=left, height=BAR, color=color, edgecolor=SURFACE, linewidth=0.8)
            if share >= 0.06:  # the count fits inside with padding
                ax.text(left + share / 2, y, str(counts[key]), ha="center", va="center",
                        fontsize=8.5, color=_label_color(color))
        left += share
    ax.text(1.015, y, f"n={n}", ha="left", va="center", fontsize=8, color=MUTED)


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of (delegation text, model) pairs", color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=9, length=0)
    ax.invert_yaxis()


def stage1_outcomes() -> Path:
    rows = [r for r in read_jsonl(DERIVED / "pairs.jsonl") if r["clarity"] == "ambiguous"]
    texts = {t["text_id"]: t for t in delegation_texts()}
    forced = forced_choice(K)

    def counts(variant, model, family=None):
        sub = [r for r in rows if r["variant"] == variant and r["executor_model"] == model
               and (family is None or r["family"] == f"{family}_ambiguous")]
        return {key: sum(r["category"] == key for r in sub) for key, _, _ in SEGMENTS}

    def forced_counts(model):
        amb = [tid for tid, t in texts.items() if t["clarity"] == "ambiguous"]
        mismatch = sum(forced[(tid, model)] != Proposal.from_dict(
            DOMAINS[texts[tid]["domain"]]["ambiguous"]["intended"]) for tid in amb)
        return {"executed_mismatch": mismatch, "executed_intended": len(amb) - mismatch}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1, 1]})

    ax, y, ticks, labels = axes[0], 0.0, [], []
    for model in MODELS:
        ax.text(-0.02, y - 0.55, model, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, color=INK, fontweight="bold")
        for condition, name in CONDITIONS:
            _bar(ax, y, forced_counts(model) if condition == "forced" else counts(condition, model))
            ticks.append(y)
            labels.append(name)
            y += 0.7
        y += 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    _style(ax, "(a) Ambiguous delegations, by Executor Agent model")

    ax, y, ticks, labels = axes[1], 0.0, [], []
    for family in FAMILIES:
        ax.text(-0.02, y - 0.55, FAMILY_NAMES[family], transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=9.5, color=INK, fontweight="bold")
        for model in MODELS:
            _bar(ax, y, counts("neutral", model, family))
            ticks.append(y)
            labels.append(model)
            y += 0.7
        y += 0.45
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    _style(ax, "(b) Option to ask, by source delegation family")

    fig.legend(handles=[Patch(facecolor=c, label=name) for _, name, c in SEGMENTS], loc="upper center",
               ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    fig.text(0.01, -0.035,
             "Executor Agent may ask the Planner Agent instead of proposing; stage 1, 80 canonical delegation texts, "
             f"{K} samples per pair. A pair asks if most of its samples ask.\n"
             "Primary analysis A (format-valid samples; gpt-4o-mini with explicit cue: 39 scorable pairs). "
             "Post-hoc analysis B (first attempts) differs by at most one pair per segment.\n"
             "Clear delegations: no pair asked under either prompt (0/80), 1/80 semantic mismatch. "
             "Forced choice: modal of the first 5 canonical samples.",
             fontsize=8, color=MUTED, ha="left", va="top")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0.03, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return OUT


def format_failures() -> Path:
    """First-attempt output-format failures, by model, delegation type and prompt."""
    units = read_jsonl(RAW) + (read_jsonl(PARSE_FAILURES) if PARSE_FAILURES.exists() else [])
    clarity = {t["text_id"]: t["clarity"] for t in delegation_texts()}
    parts = (("Valid on a retry", "#2a78d6", lambda u: u["decision"] is not None),
             ("No valid attempt", "#eb6834", lambda u: u["decision"] is None))
    rows = (("ambiguous", "neutral", "ambiguous, option to ask"), ("ambiguous", "explicit", "ambiguous, explicit cue"),
            ("clear", "neutral", "clear, option to ask"), ("clear", "explicit", "clear, explicit cue"))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y, ticks, labels = 0.0, [], []
    for model in MODELS:
        ax.text(-0.01, y - 0.5, model, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, color=INK, fontweight="bold")
        for clar, variant, name in rows:
            us = [u for u in units if u["model"] == model and u["variant"] == variant and clarity[u["text_id"]] == clar]
            failed = [u for u in us if u["parse_errors"]]
            left = 0.0
            for _, color, test in parts:
                share = sum(map(test, failed)) / len(us)
                if share:
                    ax.barh(y, share, left=left, height=BAR, color=color, edgecolor=SURFACE, linewidth=0.8)
                left += share
            never = sum(u["decision"] is None for u in failed)
            ax.text(left + 0.004, y, f"{len(failed)}/{len(us)}" + (f"  ({never} no valid attempt)" if never else ""),
                    ha="left", va="center", fontsize=8, color=INK)
            ticks.append(y)
            labels.append(name)
            y += 0.6
        y += 0.5
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.35)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["0%", "10%", "20%", "30%"])
    ax.set_xlabel("share of units whose first attempt failed format validation", color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=9, length=0)
    ax.set_title("Output-format failures with the option to ask, by model, delegation type and prompt", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold", pad=26)
    ax.legend(handles=[Patch(facecolor=c, label=name) for name, c, _ in parts], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, handlelength=1.2)
    attempts = {m: sum(len(u["parse_errors"]) for u in units if u["model"] == m) for m in MODELS}
    fig.text(0.01, -0.03,
             f"A unit is one planned sample (200 per bar) with up to 4 attempts. Failed attempts in total: "
             + ", ".join(f"{m} {n}" for m, n in attempts.items())
             + "; every one wrote an action name into \"decision\".\nThe canonical forced-choice prompt had no "
             "format failures for these models. Units with no valid attempt are unscored in analysis A and read "
             "from their first attempt in analysis B.", fontsize=8, color=MUTED, ha="left", va="top")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    out = ROOT / "figures" / "clarification_option_format_failures.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    for path in (stage1_outcomes(), format_failures()):
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
