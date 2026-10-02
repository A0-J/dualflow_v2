"""Saves E3's cross-model comparison as CSV + a static bar figure
(PNG/PDF/SVG) -- 0 new API calls, reads the already-collected derived
results for all four replicates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE.parent.parent / "figures"
_DERIVED = _HERE.parent.parent / "results" / "e2" / "derived"
_RAW = _HERE.parent.parent / "results" / "e2" / "raw"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

REPLICATES = [
    ("gpt-4o-mini\n(original)", _DERIVED / "full_results.jsonl", None),
    ("gpt-4o-mini\n(repeat)", _DERIVED / "gpt4omini_repeat_results.jsonl", None),
    ("gpt-4.1-mini", _DERIVED / "gpt41mini_results.jsonl", None),
    ("gpt-4.1", _DERIVED / "gpt41_results.jsonl", None),
]

P2_IDS = {"exfil_summarize_p2__r00", "exfil_summarize_p2__r01", "exfil_summarize_p2__r02",
         "exfil_send_external_p2__r00", "exfil_send_external_p2__r01", "exfil_send_external_p2__r02"}


def _count(path: Path, arm: str = "grounded") -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if path.name == "full_results.jsonl":
        rows = [r for r in rows if r["episode_id"] in P2_IDS]
    unsafe = sum(r["arms"][arm]["unsafe_execution"] for r in rows)
    fr = sum(r["arms"][arm]["false_reject"] for r in rows)
    ex = sum(r["arms"][arm]["execute"] for r in rows)
    return {"n_episodes": len(rows), "unsafe": unsafe, "false_reject": fr, "execute": ex}


def main() -> int:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels, stats = [], []
    for label, path, _ in REPLICATES:
        labels.append(label)
        stats.append(_count(path))

    csv_path = _HERE / "cross_model_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["replicate", "n_episodes", "unsafe_execution", "false_reject", "execute"])
        for label, s in zip(labels, stats):
            w.writerow([label.replace("\n", " "), s["n_episodes"], s["unsafe"], s["false_reject"], s["execute"]])
    print(f"saved {csv_path}")

    execute_counts = [s["execute"] for s in stats]
    n = stats[0]["n_episodes"]

    # Single color (not 4 distinct categorical hues): the finding IS that all
    # four replicates are identical, so one consistent color -- matching
    # "EXECUTE" in the E1/E2 figures -- communicates sameness; 4 different
    # hues would visually imply a distinction that isn't there.
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(labels, execute_counts, color=AQUA, width=0.55)
    for bar, count in zip(bars, execute_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, str(count),
               ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")

    ax.set_ylabel(f"EXECUTE count (of {n} episodes)", color=INK, fontsize=10)
    ax.set_ylim(0, max(execute_counts) + 2)
    ax.set_title("E3 cross-model: Grounded arm, P2-paraphrase test subset\n"
                "(Delegate frozen; only Principal's 20-sample bank regenerated per model)",
                fontsize=10.5, color=INK, pad=12)
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
        out = _FIGURES_DIR / f"e3_cross_model.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else None, facecolor="white", bbox_inches="tight")
        print(f"saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
