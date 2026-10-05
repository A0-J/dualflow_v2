"""Delegate Repeated-Sampling Analysis -- offline analysis step (0 API calls).

Reads results/delegate_repeated_sampling/raw/episodes.jsonl (6 scenarios,
20 fresh Delegate samples + 20 existing Principal samples each) and
computes, per scenario and per side (delegate/principal):
  - per-facet mode, entropy, confirmed (entropy <= threshold)
  - overall (whole-interpretation) mode, entropy, stable (entropy <= threshold)
  - correct (overall mode matches scenario's ground-truth principal_intent)

Then classifies each scenario into exactly one of 5 buckets based on
{delegate stable/unstable} x {principal stable/unstable}, with the
stable+stable case split by agreement -- "confident_semantic_misread" is
the headline category this analysis exists to detect or rule out.

No E2 code, threshold, or result file is read for scoring (only the
already-collected principal_samples carried into this experiment's own
raw file by collect.py) -- threshold is the same frozen project default
(0.8) used everywhere else, not tuned on this data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dualflow.io import read_jsonl, write_jsonl
from dualflow.models import Interpretation
from dualflow.semantic import FACETS, entropy, exact_semantic_match, repeated_anchor

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent


def _mode_and_entropy(values: list) -> tuple:
    from collections import Counter
    counts = Counter(values)
    first = {v: values.index(v) for v in counts}
    mode = min(counts, key=lambda v: (-counts[v], first[v]))
    return mode, entropy(values)


def _facet_table(samples: list[Interpretation], threshold: float) -> dict:
    normalized = [s.canonical() for s in samples]
    out = {}
    for facet in FACETS:
        values = [getattr(s, facet) for s in normalized]
        mode, h = _mode_and_entropy(values)
        out[facet] = {
            "mode": sorted(mode) if isinstance(mode, frozenset) else mode,
            "entropy": h,
            "confirmed": h <= threshold,
        }
    return out


def _side_summary(samples: list[Interpretation], truth: Interpretation, threshold: float) -> dict:
    normalized = [s.canonical() for s in samples]
    anchor = repeated_anchor(samples)
    h = entropy(normalized)
    return {
        "anchor": anchor.to_dict(),
        "entropy": h,
        "stable": h <= threshold,
        "correct": exact_semantic_match(anchor, truth),
        "facets": _facet_table(samples, threshold),
        "n_samples": len(samples),
    }


def classify(delegate: dict, principal: dict) -> str:
    """Whole-interpretation classification (all 4 facets as one unit).

    Kept for transparency, but NOT the headline category -- see
    `classify_facet()` below. This scenario's `condition` facet is noisy
    on BOTH sides for reasons unrelated to the action-level question this
    analysis targets (see RESULTS.md), and that noise alone can mark an
    otherwise cleanly-resolved scenario "unstable" here even when the
    facet that actually matters (action) is fully stable on both sides.
    """
    d_stable, p_stable = delegate["stable"], principal["stable"]
    if d_stable and p_stable:
        agree = delegate["anchor"] == principal["anchor"]
        return "both_stable_agree" if agree else "confident_semantic_misread"
    if not d_stable and p_stable:
        return "delegate_unstable"
    if d_stable and not p_stable:
        return "principal_unstable"
    return "both_unstable"


def classify_facet(delegate: dict, principal: dict, facet: str) -> str:
    """Single-facet classification -- the headline metric.

    Matches E2's own established convention (`grounded_verdict` reasons
    about mismatches per-facet, not over the whole interpretation as one
    unit) and is what directly answers this analysis's RQ: for the facet
    that actually varies across this scenario's two intents (`action`),
    is the Delegate confidently (low-entropy) wrong while the Principal is
    confidently right?
    """
    d, p = delegate["facets"][facet], principal["facets"][facet]
    d_stable, p_stable = d["confirmed"], p["confirmed"]
    if d_stable and p_stable:
        return "both_stable_agree" if d["mode"] == p["mode"] else "confident_semantic_misread"
    if not d_stable and p_stable:
        return "delegate_unstable"
    if d_stable and not p_stable:
        return "principal_unstable"
    return "both_unstable"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delegate Repeated-Sampling Analysis -- offline, 0 API calls.")
    p.add_argument("--input", default=str(_ROOT / "results" / "delegate_repeated_sampling" / "raw" / "episodes.jsonl"))
    p.add_argument("--output", default=str(_ROOT / "results" / "delegate_repeated_sampling" / "derived" / "analysis.jsonl"))
    p.add_argument("--threshold", type=float, default=0.8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep in read_jsonl(args.input):
        truth = Interpretation.from_dict(ep["principal_intent"])
        delegate_samples = [Interpretation.from_dict(c["interpretation"]) for c in ep["delegate_samples"]]
        principal_samples = [Interpretation.from_dict(c["interpretation"]) for c in ep["principal_samples"]]

        delegate = _side_summary(delegate_samples, truth, args.threshold)
        principal = _side_summary(principal_samples, truth, args.threshold)
        category = classify(delegate, principal)
        action_category = classify_facet(delegate, principal, "action")
        condition_category = classify_facet(delegate, principal, "condition")

        rows.append({
            "scenario_id": ep["scenario_id"],
            "split": ep["split"],
            "delegation": ep["delegation"],
            "ground_truth": truth.to_dict(),
            "threshold": args.threshold,
            "delegate": delegate,
            "principal": principal,
            "category": category,
            "action_category": action_category,
            "condition_category": condition_category,
        })

    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} scenarios -> {args.output}")

    # --- scenario-level summary CSV ---
    summary_csv = out_dir / "scenario_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario_id", "split", "whole_interpretation_category", "action_category", "condition_category",
            "delegate_anchor_action", "delegate_entropy", "delegate_stable", "delegate_correct",
            "principal_anchor_action", "principal_entropy", "principal_stable", "principal_correct",
        ])
        for r in rows:
            w.writerow([
                r["scenario_id"], r["split"], r["category"], r["action_category"], r["condition_category"],
                r["delegate"]["anchor"]["action"], f'{r["delegate"]["entropy"]:.4f}',
                r["delegate"]["stable"], r["delegate"]["correct"],
                r["principal"]["anchor"]["action"], f'{r["principal"]["entropy"]:.4f}',
                r["principal"]["stable"], r["principal"]["correct"],
            ])
    print(f"saved {summary_csv}")

    # --- per-facet entropy CSV (long format: scenario x side x facet) ---
    facet_csv = out_dir / "facet_entropy.csv"
    with open(facet_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "split", "side", "facet", "mode", "entropy", "confirmed"])
        for r in rows:
            for side_name in ("delegate", "principal"):
                side = r[side_name]
                for facet, v in side["facets"].items():
                    w.writerow([r["scenario_id"], r["split"], side_name, facet, v["mode"],
                               f'{v["entropy"]:.4f}', v["confirmed"]])
    print(f"saved {facet_csv}")

    # --- console summary ---
    print("\n=== Classification ===")
    for r in rows:
        print(f'{r["scenario_id"]:28s} {r["category"]}')

    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"\n=== Test split (P2/P3) only, n={len(test_rows)} ===")
    for r in test_rows:
        print(f'{r["scenario_id"]:28s} delegate: entropy={r["delegate"]["entropy"]:.3f} '
              f'stable={r["delegate"]["stable"]} correct={r["delegate"]["correct"]} '
              f'anchor_action={r["delegate"]["anchor"]["action"]} | '
              f'principal: entropy={r["principal"]["entropy"]:.3f} '
              f'stable={r["principal"]["stable"]} correct={r["principal"]["correct"]} '
              f'anchor_action={r["principal"]["anchor"]["action"]} | {r["category"]}')


if __name__ == "__main__":
    main()
