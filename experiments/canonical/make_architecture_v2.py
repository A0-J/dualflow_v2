"""DualFlow architecture, readback form (PNG only).

The same verification as figures/dualflow_architecture.png, drawn as the sequence it runs in:
the Planner Agent seals its intent when it delegates, the Executor Agent proposes without
executing, the verifier on the Planner side checks the proposal, a tool gateway runs it only with a
permit, and a rejection returns the fields that differ so that the Executor Agent can revise them,
for at most k rounds. The original figure is kept unchanged.

    PYTHONPATH=src python experiments/canonical/make_architecture_v2.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import ROOT
from make_architecture import AGENT, AUTH, EDGE, GATE, MUTED, SEM, box

REGION = "#9a9893"


def arrow(ax, a, b, text="", at=None, ha="center", style="arc3", color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color, linewidth=1.3,
                                 connectionstyle=style))
    if text:
        x, y = at or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.1)
        ax.text(x, y, text, ha=ha, va="bottom", fontsize=8.6, color=MUTED, linespacing=1.3)


def region(ax, x, y, w, h, label, right=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor="none",
                                edgecolor=REGION, linewidth=1.1, linestyle=(0, (5, 3))))
    ax.text(x + w - 0.15 if right else x + 0.15, y + h - 0.12, label, ha="right" if right else "left",
            va="top", fontsize=8.6, color=MUTED, style="italic")


def main() -> int:
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    # inputs from outside
    box(ax, 0.3, 6.45, 2.9, 0.6, "#ffffff", EDGE["gate"], "User or assigning agent")
    box(ax, 5.35, 6.25, 3.6, 0.9, "#ffffff", EDGE["auth"], "Authorization store (external)",
        "policy $\\pi$ (administrators), approvals $\\sigma$ (records)")

    # Planner side
    region(ax, 0.15, 2.55, 12.75, 3.3, "Planner side (trusted; outside the Executor Agent's control)", right=True)
    box(ax, 0.3, 2.7, 2.9, 2.85, AGENT, EDGE["agent"], "Planner Agent",
        "keeps task context $C_{task}$\n(goal, constraints; never sent)\n\n(1) writes delegation message $d$\n"
        "(2) restates its intent $N$ times\nfrom $C_{task}$ and $d$ and seals the\nmode $\\hat{P}_{intent}$ before any proposal")
    region(ax, 4.0, 2.7, 6.05, 2.85, "DualFlow verifier")
    box(ax, 4.25, 4.05, 5.6, 1.15, AUTH, EDGE["auth"], "Authorization Flow",
        "$P_{exec}$ checked against $\\pi$ and $\\sigma$ by fixed rules (no LLM)\n$V_{auth} = 1$ iff every rule holds; else the rule that fails")
    box(ax, 4.25, 2.82, 5.6, 1.1, SEM, EDGE["sem"], "Semantic Flow",
        "$P_{exec}$ vs $\\hat{P}_{intent}$, field by field: $a$, $r$, $s$\n$V_{sem} = 1$ iff all three match; else the fields that differ")
    box(ax, 10.35, 3.2, 2.3, 1.75, GATE, EDGE["gate"], "Decision", "\n$D = V_{sem} \\wedge V_{auth}$")

    # outside the Planner side
    box(ax, 13.45, 3.2, 2.4, 1.75, "#ffffff", EDGE["sem"], "Tool gateway",
        "(4) $D = 1$: runs $P_{exec}$\n(nothing runs\nwithout a permit)")
    box(ax, 0.3, 0.2, 2.9, 2.0, AGENT, EDGE["agent"], "Executor Agent",
        "sees only $d$ and the\nruntime context\n(3) proposes $P_{exec} = (a, r, s)$\nand waits; (5) revises only\nthe flagged fields")
    box(ax, 5.6, 0.45, 4.6, 1.3, "#ffffff", EDGE["auth"], "Reject with reasons",
        "(4) $D = 0$: which field ($a$, $r$ or $s$) differs\nand why, or which authorization rule fails")
    box(ax, 11.2, 0.45, 4.65, 1.3, "#ffffff", EDGE["gate"], "Stop after $k$ rounds",
        "$k = 3$ by default: nothing is run,\nand the Planner Agent is told why")

    arrow(ax, (1.75, 6.45), (1.75, 5.55), "user request", at=(1.87, 5.95), ha="left")
    arrow(ax, (7.15, 6.25), (7.15, 5.2), "$\\pi$, $\\sigma$", at=(7.27, 5.62), ha="left")
    arrow(ax, (1.75, 2.7), (1.75, 2.2), "$d$", at=(1.87, 2.33), ha="left")
    arrow(ax, (3.2, 3.37), (4.25, 3.37), "(2)\n$\\hat{P}_{intent}$", at=(3.6, 3.45))
    arrow(ax, (3.2, 1.9), (4.6, 2.82), "(3) readback: $P_{exec}$\n(proposal, not executed)", at=(4.72, 2.02),
          ha="left", style="angle,angleA=0,angleB=90,rad=0")
    arrow(ax, (9.85, 4.62), (10.35, 4.3))
    arrow(ax, (9.85, 3.37), (10.35, 3.85))
    arrow(ax, (12.65, 4.07), (13.45, 4.07), "permit", at=(13.05, 4.15))
    arrow(ax, (11.5, 3.2), (9.6, 1.75), "reject", at=(10.5, 2.15), ha="left")
    arrow(ax, (5.6, 1.0), (3.2, 1.0), "(5) feedback:\nrevise the flagged fields", at=(4.4, 1.07))
    arrow(ax, (10.2, 1.0), (11.2, 1.0), "$k$-th reject", at=(10.7, 1.07))

    out = ROOT / "figures" / "dualflow_architecture_v2.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
