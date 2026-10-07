"""DualFlow architecture diagram (PNG only)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import ROOT

INK, MUTED = "#0b0b0b", "#52514e"
AGENT, SEM, AUTH, GATE = "#e8f0fb", "#e6f6ef", "#fdeee6", "#f3f3f0"
EDGE = {"agent": "#2a78d6", "sem": "#1baf7a", "auth": "#eb6834", "gate": "#898781"}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def box(ax, x, y, w, h, fill, edge, title, body=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.6, color=MUTED, linespacing=1.4)


def arrow(ax, a, b, text="", offset=(0, 0.12), color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.2))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text,
                ha="center", va="bottom", fontsize=8.4, color=MUTED)


def main() -> int:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    box(ax, 0.2, 2.6, 2.4, 1.7, AGENT, EDGE["agent"], "Planner Agent",
        "private task context g\n(never shown to the\nExecutor)")
    box(ax, 0.2, 0.3, 2.4, 1.7, AGENT, EDGE["agent"], "Executor Agent",
        "sees delegation d\nand runtime context c")
    arrow(ax, (1.4, 2.6), (1.4, 2.0), "delegation d", offset=(0.75, -0.12))

    box(ax, 3.6, 2.6, 3.5, 1.7, SEM, EDGE["sem"], "Semantic Flow",
        "Planner restates intent N times\n→ modal o_P = (a, r, s)\npass iff o_E = o_P")
    box(ax, 3.6, 0.3, 3.5, 1.7, AUTH, EDGE["auth"], "Authorization Flow",
        "policy π: actions, resources, scopes\nauthorization state σ: granted approvals\npass iff π allows o_E, and σ holds\na granted approval when required")

    arrow(ax, (2.6, 3.45), (3.6, 3.45), "restatements")
    arrow(ax, (2.6, 1.15), (3.6, 1.15), "proposal o_E", offset=(0, -0.38))
    arrow(ax, (3.1, 1.15), (3.75, 2.6), "", color=MUTED)

    box(ax, 8.0, 1.45, 1.6, 1.7, GATE, EDGE["gate"], "AND", "execute only if\nboth flows pass")
    arrow(ax, (7.1, 3.45), (8.0, 2.75))
    arrow(ax, (7.1, 1.15), (8.0, 1.85))

    box(ax, 10.2, 2.6, 1.6, 1.0, "#ffffff", EDGE["sem"], "EXECUTE")
    box(ax, 10.2, 0.85, 1.6, 1.0, "#ffffff", EDGE["auth"], "BLOCK", "reason recorded")
    arrow(ax, (9.6, 2.6), (10.2, 3.1))
    arrow(ax, (9.6, 2.0), (10.2, 1.35))

    ax.text(5.35, 2.33, "independent: different sources of truth", ha="center", va="center",
            fontsize=8.4, color=MUTED, style="italic")

    out = ROOT / "figures" / "dualflow_architecture.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
