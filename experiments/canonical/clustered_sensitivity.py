"""Post-hoc dependence sensitivity for the canonical C1-C4 evaluation.

The pre-registered cell-level analysis (analyze.py) is unchanged. Its cells
are nested:

    8 source delegations (one delegation variant family each)
      -> 80 delegation texts
        -> 2 authorization configurations
          -> 4 Executor Agent models

This script re-expresses the results at two coarser levels.

Level 1, delegation text (80 units). The authorization configurations and
models of each text are collapsed into one value per text. Texts derived
from the same source delegation are not independent, so this level is
descriptive only: it reports direction counts and incidence, never a
p-value or a confidence interval. It answers whether an effect repeats
across the wordings of a family.

Level 2, source delegation (8 units). This is the most conservative
independent unit. It reports exact family-incidence intervals and exact
one-sided sign tests, with only 4 families per case.

Rates are always computed per (source family, case). C1 and C3, or C2 and
C4, are never pooled into one rate.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

from common import DERIVED, EXECUTOR_MODELS, clopper_pearson
from dualflow.io import read_jsonl
from scenarios import DOMAINS

METHODS = ("no_verification", "self_consistency", "semantic_only",
           "authorization_only", "dualflow")
CASES_BY_CLARITY = {"clear": ("C1", "C3"), "ambiguous": ("C2", "C4")}
# name, case, comparator (method, Planner Agent condition), reference (method, condition)
COMPARISONS = (
    ("H3a_semantic_flow_only_vs_dualflow", "C3", ("semantic_only", "context"), ("dualflow", "context")),
    ("H3b_authorization_flow_only_vs_dualflow", "C2", ("authorization_only", "context"), ("dualflow", "context")),
    ("H5_dualflow_no_context_vs_context", "C2", ("dualflow", "no_context"), ("dualflow", "context")),
)


def group_key(row: dict) -> str:
    """Source delegation (variant family) of a cell, e.g. 'audit_ambiguous'."""
    text_id = row["text_id"]
    group, separator, suffix = text_id.rpartition("_")
    if not separator or not suffix.isdigit() or not group.startswith(row["domain"] + "_"):
        raise ValueError(f"Unexpected canonical text_id: {text_id!r}")
    return group


def text_key(row: dict) -> str:
    return row["text_id"]


def sign_test_greater(positive: int, negative: int) -> float:
    """One-sided exact sign-test p-value. Ties must be excluded by the caller."""
    if positive < 0 or negative < 0:
        raise ValueError("Sign counts cannot be negative")
    n = positive + negative
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(positive, n + 1)) / 2**n


def validate_complete_design(rows: list[dict]) -> None:
    """Every family must have all 10 texts x its 2 cases x 4 models, once, per condition."""
    expected_groups = {f"{d}_{c}" for d in DOMAINS for c in CASES_BY_CLARITY}
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["planner_condition"] not in {"context", "no_context"}:
            raise ValueError(f"Unexpected Planner condition: {row['planner_condition']!r}")
        by_group[group_key(row)].append(row)
    if set(by_group) != expected_groups:
        raise ValueError(f"Expected all 8 source delegations; found {sorted(by_group)}")
    for group, group_rows in by_group.items():
        cases = CASES_BY_CLARITY[group.rsplit("_", 1)[1]]
        expected = {(f"{group}_{i:02d}", case, model)
                    for i in range(10) for case in cases for model in EXECUTOR_MODELS}
        for condition in ("context", "no_context"):
            observed = [(r["text_id"], r["case"], r["executor_model"])
                        for r in group_rows if r["planner_condition"] == condition]
            if len(observed) != len(set(observed)) or set(observed) != expected:
                raise ValueError(f"Incomplete or duplicate {condition} cells in {group}: "
                                 f"expected {len(expected)}, found {len(observed)}")


def family_case_rates(rows: list[dict]) -> list[dict]:
    """Rate per (method, metric, source family, case); Planner Agent with task context."""
    cells = defaultdict(list)
    for row in rows:
        if row["planner_condition"] == "context":
            cells[(group_key(row), row["case"])].append(row)
    out = []
    for method in METHODS:
        for metric, eligible in (("unsafe", lambda r: True),
                                 ("false_reject", lambda r: r["should_execute"]),
                                 ("correct", lambda r: True)):
            for (group, case) in sorted(cells):
                den = [r for r in cells[(group, case)] if eligible(r)]
                num = sum(bool(r[f"{method}__{metric}"]) for r in den)
                out.append({"method": method, "metric": metric, "source_family": group, "case": case,
                            "numerator": num, "denominator": len(den),
                            "rate": num / len(den) if den else ""})
    return out


def incidence(rows: list[dict], unit, *, with_interval: bool) -> list[dict]:
    """Units (texts or families) with at least one unsafe execution / false rejection.

    A unit counts toward a metric only if it has an eligible cell. The exact
    interval is reported only for independent units (source families)."""
    by_unit = defaultdict(list)
    for row in rows:
        if row["planner_condition"] == "context":
            by_unit[unit(row)].append(row)
    out = []
    for method in METHODS:
        for metric, eligible in (("unsafe", lambda r: True), ("false_reject", lambda r: r["should_execute"])):
            units = [[r for r in v if eligible(r)] for v in by_unit.values()]
            units = [u for u in units if u]
            events = sum(any(r[f"{method}__{metric}"] for r in u) for u in units)
            row = {"method": method, "metric": metric, "units": len(units), "units_with_event": events,
                   "exact_95_upper": ""}
            if with_interval:
                row["exact_95_upper"] = clopper_pearson(events, len(units))[1]
            out.append(row)
    return out


def _unsafe_rate(rows: list[dict], method: str, condition: str) -> float:
    selected = [r for r in rows if r["planner_condition"] == condition]
    if not selected:
        raise ValueError(f"No {condition} cells for a paired unit")
    return sum(bool(r[f"{method}__unsafe"]) for r in selected) / len(selected)


def paired_differences(rows: list[dict], unit) -> list[dict]:
    """Per-unit unsafe-rate difference (comparator - reference) for each comparison."""
    out = []
    for name, case, (c_method, c_cond), (r_method, r_cond) in COMPARISONS:
        by_unit = defaultdict(list)
        for row in rows:
            if row["case"] == case:
                by_unit[unit(row)].append(row)
        for key in sorted(by_unit):
            a = _unsafe_rate(by_unit[key], c_method, c_cond)
            b = _unsafe_rate(by_unit[key], r_method, r_cond)
            out.append({"comparison": name, "case": case, "unit": key, "comparator_rate": a,
                        "reference_rate": b, "difference": a - b,
                        "direction": "comparator_higher" if a > b else "reference_higher" if a < b else "tie"})
    return out


def direction_counts(details: list[dict], *, sign_test: bool) -> list[dict]:
    out = []
    for name, case, *_ in COMPARISONS:
        d = [x for x in details if x["comparison"] == name]
        higher = sum(x["direction"] == "comparator_higher" for x in d)
        lower = sum(x["direction"] == "reference_higher" for x in d)
        row = {"comparison": name, "case": case, "units": len(d), "comparator_higher": higher,
               "reference_higher": lower, "ties": len(d) - higher - lower}
        if sign_test:
            row["exact_one_sided_sign_test_p"] = sign_test_greater(higher, lower)
        out.append(row)
    return out


def _write_csv(path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cells = read_jsonl(DERIVED / "cells.jsonl")
    if not cells:
        raise SystemExit(f"No cells found at {DERIVED / 'cells.jsonl'}; run analyze.py first.")
    validate_complete_design(cells)

    text_details = paired_differences(cells, text_key)
    family_details = paired_differences(cells, group_key)
    text_directions = direction_counts(text_details, sign_test=False)
    family_tests = direction_counts(family_details, sign_test=True)
    text_incidence = incidence(cells, text_key, with_interval=False)
    family_incidence = incidence(cells, group_key, with_interval=True)

    _write_csv(DERIVED / "clustered_family_case_rates.csv", family_case_rates(cells))
    _write_csv(DERIVED / "clustered_text_robustness.csv", text_directions)
    _write_csv(DERIVED / "clustered_text_incidence.csv", text_incidence)
    _write_csv(DERIVED / "clustered_family_incidence.csv", family_incidence)
    _write_csv(DERIVED / "clustered_paired_tests.csv", family_tests)
    _write_csv(DERIVED / "clustered_paired_group_differences.csv", family_details)

    def dualflow(rows, metric):
        return next(r for r in rows if r["method"] == "dualflow" and r["metric"] == metric)

    print("Post-hoc dependence sensitivity (does not replace the pre-registered cell-level analysis)")
    print("\nLevel 1 - delegation text (80), descriptive only: texts within a family are not independent")
    for metric in ("unsafe", "false_reject"):
        r = dualflow(text_incidence, metric)
        print(f"  DualFlow {metric}: {r['units_with_event']}/{r['units']} texts with any event")
    for r in text_directions:
        print(f"  {r['comparison']}: comparator higher in {r['comparator_higher']}/{r['units']} texts, "
              f"reference higher in {r['reference_higher']}, ties {r['ties']}")
    print("\nLevel 2 - source delegation (8), the most conservative independent unit")
    for metric in ("unsafe", "false_reject"):
        r = dualflow(family_incidence, metric)
        print(f"  DualFlow {metric}: {r['units_with_event']}/{r['units']} families with any event, "
              f"exact 95% upper {r['exact_95_upper']:.3f}")
    for r in family_tests:
        print(f"  {r['comparison']}: comparator higher in {r['comparator_higher']}/{r['units']} families, "
              f"reference higher in {r['reference_higher']}, ties {r['ties']}, "
              f"one-sided sign test p={r['exact_one_sided_sign_test_p']:.4f}")
    print(f"\nWrote six tables (clustered_*.csv) to {DERIVED}.")


if __name__ == "__main__":
    main()
