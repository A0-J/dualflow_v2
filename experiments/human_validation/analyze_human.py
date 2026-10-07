"""Analyze the human ground-truth validation (PREREGISTRATION.md §4). 0 API calls.

Reads results/human_validation/key.json and every
results/human_validation/responses/*.json (one file per annotator, as
downloaded from the annotation tool). Usage:

    PYTHONPATH=src python experiments/human_validation/analyze_human.py [--responses DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from common import DERIVED as CANONICAL_DERIVED, fmt  # noqa: E402
from dualflow.io import read_jsonl  # noqa: E402

OUT = ROOT / "results" / "human_validation"
METHODS = ("no_verification", "self_consistency", "semantic_only", "authorization_only", "dualflow")


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """ratings: one list of category labels per item, same number of raters per item."""
    ratings = [r for r in ratings if r]
    n = len(ratings[0])
    assert all(len(r) == n for r in ratings) and n >= 2
    cats = sorted({c for r in ratings for c in r})
    counts = [[r.count(c) for c in cats] for r in ratings]
    p_i = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts]
    p_bar = sum(p_i) / len(p_i)
    p_j = [sum(row[j] for row in counts) / (len(counts) * n) for j in range(len(cats))]
    p_e = sum(p * p for p in p_j)
    return 1.0 if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p for [[a, b], [c, d]]: row 1 has the higher rate."""
    n1, n2, k = a + b, c + d, a + c
    total = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / total


def majority(values: list):
    """Strict-majority value, or None."""
    if not values:
        return None
    v, c = Counter(values).most_common(1)[0]
    return v if c > len(values) / 2 else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--responses", default=str(OUT / "responses"))
    args = p.parse_args()
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    files = sorted(Path(args.responses).glob("*.json"))
    resp = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if len(resp) < 2:
        raise SystemExit(f"need >= 2 annotator files in {args.responses}, found {len(resp)}")
    complete = [r for r in resp if all(k["a_id"] in r["phase_a"] and k["b_id"] in r["phase_b"] for k in key.values())]
    print(f"annotators: {len(resp)} files, {len(complete)} complete -> using complete only")
    resp = complete
    k_raters = len(resp)

    items = {}
    for tid, k in key.items():
        a = [r["phase_a"][k["a_id"]] for r in resp]
        b = [r["phase_b"][k["b_id"]] for r in resp]
        plaus = Counter(x for ans in a for x in set(ans["plausible"]))
        items[tid] = {
            **k,
            "b1": [x["intended"] for x in b], "a2": [x["most_likely"] for x in a],
            "a3": [x["ambiguity"] for x in a], "b2": [x["preservation"] for x in b],
            "plausible_majority": {o for o, c in plaus.items() if c >= k_raters / 2},
        }

    print("\n## 1. Ground-truth validity (Phase B Q1: majority intended task == intended task)")
    gt_ok = {}
    for grp in ("clear", "ambiguous", "all"):
        sub = [t for t in items.values() if grp == "all" or t["clarity"] == grp]
        agree = sum(majority(t["b1"]) == t["intended_key"] for t in sub)
        nomaj = sum(majority(t["b1"]) is None for t in sub)
        print(f"{grp:9s} majority agrees {fmt(agree, len(sub))}  (no strict majority: {nomaj})")
    for tid, t in items.items():
        gt_ok[tid] = majority(t["b1"]) == t["intended_key"]
    rate = sum(gt_ok.values()) / len(gt_ok)
    print(f"criterion (>= 90% of items): {'MET' if rate >= 0.9 else 'NOT MET'} ({rate:.1%})")
    print(f"Fleiss kappa, Phase B Q1: {fleiss_kappa([t['b1'] for t in items.values()]):.3f}")
    print("items whose majority differs from the intended task:",
          sorted(tid for tid, ok in gt_ok.items() if not ok))

    print("\n## 2. Ambiguity manipulation check (Phase A, delegation only)")
    amb_bin = {tid: majority([int(x >= 2) for x in t["a3"]]) == 1 for tid, t in items.items()}
    a_amb = [tid for tid, t in items.items() if t["clarity"] == "ambiguous"]
    a_clr = [tid for tid, t in items.items() if t["clarity"] == "clear"]
    ka, kc = sum(amb_bin[t] for t in a_amb), sum(amb_bin[t] for t in a_clr)
    pval = fisher_greater(ka, len(a_amb) - ka, kc, len(a_clr) - kc)
    print(f"rated ambiguous (majority Q3 >= 2): ambiguous group {fmt(ka, len(a_amb))}, clear group {fmt(kc, len(a_clr))}, "
          f"one-sided Fisher p={pval:.3g} -> {'MET' if pval < 0.05 else 'NOT MET'}")
    for grp, ids in (("ambiguous", a_amb), ("clear", a_clr)):
        mean = sum(sum(items[t]["a3"]) / k_raters for t in ids) / len(ids)
        print(f"  mean Q3 rating, {grp}: {mean:.2f} (1 = clear, 3 = ambiguous)")
    k_mis = sum(items[t]["misinterpretation_key"] in items[t]["plausible_majority"] for t in a_amb)
    print(f"ambiguous group: irreversible misinterpretation judged plausible by >= half of annotators: {fmt(k_mis, len(a_amb))}")
    k_only = sum(items[t]["plausible_majority"] == {items[t]["intended_key"]} for t in a_clr)
    print(f"clear group: only the intended task judged plausible: {fmt(k_only, len(a_clr))}")
    print(f"Fleiss kappa, Phase A Q2 (most likely task): {fleiss_kappa([t['a2'] for t in items.values()]):.3f}")
    print(f"Fleiss kappa, Phase A Q3 binarized: {fleiss_kappa([[str(int(x >= 2)) for x in t['a3']] for t in items.values()]):.3f}")

    print("\n## 3. Human first reading vs Executor Agent (ambiguous group)")
    human_mis = sum(majority(items[t]["a2"]) == items[t]["misinterpretation_key"] for t in a_amb)
    print(f"annotators' majority most-likely task is the irreversible misinterpretation: {fmt(human_mis, len(a_amb))}")
    cells = [c for c in read_jsonl(CANONICAL_DERIVED / "cells.jsonl") if c["planner_condition"] == "context" and c["case"] == "C2"]
    for model in sorted({c["executor_model"] for c in cells}):
        sub = [c for c in cells if c["executor_model"] == model]
        print(f"  Executor Agent {model}: {fmt(sum(c['misread'] for c in sub), len(sub))}")

    print("\n## 4. Meaning preservation of generated delegation variants (Phase B Q2)")
    variants = [tid for tid, t in items.items() if not t["original"]]
    drift = {tid: majority([int(x == 3) for x in items[tid]["b2"]]) == 1 for tid in variants}
    print(f"variants judged to change the meaning (majority Q2 = 3): {fmt(sum(drift.values()), len(variants))}")
    print("  ", sorted(tid for tid, d in drift.items() if d))
    print(f"Fleiss kappa, Phase B Q2 binarized: {fleiss_kappa([[str(int(x == 3)) for x in items[t]['b2']] for t in variants]):.3f}")

    print("\n## 5. Sensitivity of the main results (Planner Agent with task context, all cases)")
    all_cells = [c for c in read_jsonl(CANONICAL_DERIVED / "cells.jsonl") if c["planner_condition"] == "context"]
    excl_gt = {tid for tid, ok in gt_ok.items() if not ok}
    excl_drift = {tid for tid, d in drift.items() if d}
    for name, excluded in (("all texts", set()), ("excluding ground-truth disagreements", excl_gt),
                           ("excluding meaning-changing variants", excl_drift),
                           ("excluding both", excl_gt | excl_drift)):
        sub = [c for c in all_cells if c["text_id"] not in excluded]
        should = [c for c in sub if c["should_execute"]]
        print(f"### {name}: {len(sub) // 4 if sub else 0} scenarios x 4 Executor Agent models = {len(sub)} cells")
        for m in METHODS:
            print(f"  {m:20s} unsafe {fmt(sum(c[f'{m}__unsafe'] for c in sub), len(sub)):26s} "
                  f"false rejection {fmt(sum(c[f'{m}__false_reject'] for c in should), len(should))}")


if __name__ == "__main__":
    main()
