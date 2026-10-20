"""Evaluation set, phase C: analysis of the d-contract checker on the evaluation tasks. 0 API calls.

Inputs (under results/schema_free/main_eval/): planner.jsonl (phase A), the checker runs on the
natural contracts and on the mutations (check_consistency.py with --contracts and --out), and the
hand labels of the delegations (labels/main_delegations.json, written after phase A and before any
checker run).

Two references are kept apart:
  intent preservation  the contract against the task's ground truth (what the Planner Agent intends)
  local consistency    the contract against the effects the delegation d itself asks for (hand label)
They differ only where d does not preserve the intent; such tasks are reported separately.

Mutations measure detection by error type; natural contracts measure how often errors occur and
what the checker does with them. Rates are given per contract with a Wilson 95% interval and a
task-cluster bootstrap 95% interval (contracts of one task are not independent), and per task and
per domain.

    PYTHONPATH=src python experiments/schema_free/eval_consistency.py [--checks natural_v2 mutations_v2]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dualflow.effect_contracts import IRREVERSIBLE_KINDS, Effect, IntentContract  # noqa: E402
from dualflow.io import read_jsonl  # noqa: E402
from dualflow.schema_free import Task, gate, modal_contract, parse_contract  # noqa: E402
from classify_contracts import classify  # noqa: E402
from mutate_contracts import CONTROLS, form_of  # noqa: E402
from tasks import MAIN_TASKS, domain_of  # noqa: E402

EVAL = ROOT / "results" / "schema_free" / "main_eval"
LABELS = Path(__file__).resolve().parent / "labels" / "main_delegations.json"
TASKS = {t.task_id: t for t in MAIN_TASKS}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_ci(rows: list[tuple[str, bool]], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap over tasks: resample tasks with replacement, pool their contracts."""
    by_task = defaultdict(list)
    for tid, x in rows:
        by_task[tid].append(x)
    ids = sorted(by_task)
    if not ids:
        return (math.nan, math.nan)
    rng, stats = random.Random(seed), []
    for _ in range(reps):
        pick = [x for t in (rng.choice(ids) for _ in ids) for x in by_task[t]]
        stats.append(sum(pick) / len(pick))
    stats.sort()
    return (stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1])


def rate(name: str, rows: list[tuple[str, bool]]) -> str:
    k, n = sum(x for _, x in rows), len(rows)
    if n == 0:
        return f"  {name:34s} -"
    lo, hi = wilson(k, n)
    clo, chi = cluster_ci(rows)
    return (f"  {name:34s} {k:3d}/{n:<3d} = {k / n:5.1%}  Wilson [{lo:.1%}, {hi:.1%}]  "
            f"task bootstrap [{clo:.1%}, {chi:.1%}]  ({len({t for t, _ in rows})} tasks)")


def effect(text: str) -> Effect:
    parts = text.split()
    return Effect(parts[0], parts[1], parts[2] if len(parts) > 2 else "")


def goal(c: IntentContract) -> tuple[frozenset, frozenset]:
    """What the gate uses: required effects without reads, and the allowed irreversible effects."""
    return (frozenset(e for e in c.required if e.kind != "read"),
            frozenset(e for e in c.required | c.permitted if e.kind in IRREVERSIBLE_KINDS))


def intended_plan(t: Task) -> frozenset[Effect]:
    """What a plan carrying out exactly the intended task declares: its effects plus the reads they need
    (the tools for anonymized, archive and in-place writes are not built yet, so no tool plan is used)."""
    reads = {Effect("read", form_of(e.resource)[1]) for e in t.truth.required}
    reads |= {Effect("read", e.resource) for e in t.truth.required if e.kind == "external_send"}
    return frozenset(t.truth.required) | reads


def load_labels() -> dict[str, dict]:
    if not LABELS.exists():
        return {}
    raw = json.loads(LABELS.read_text(encoding="utf-8"))
    out = {}
    for tid, v in raw.items():
        t = TASKS[tid]
        implied = t.truth if v.get("same_as_truth") else IntentContract(
            frozenset(map(effect, v["implied_required"])), frozenset(), frozenset(map(effect, v.get("implied_permitted", []))))
        out[tid] = {"implied": implied, "preserves_intent": goal(implied) == goal(t.truth), "note": v.get("note", "")}
    return out


def checks(name: str) -> list[dict]:
    path = EVAL / name / "checks.jsonl"
    rows = [r for r in read_jsonl(path) if r["type"] == "check"] if path.exists() else []
    for r in rows:
        r["_contract"] = parse_contract(json.dumps(r["contract"]))
        r["_held"] = r["decision"] != "pass"
    return rows


def mutation_report(rows: list[dict], labels: dict[str, dict]) -> None:
    print(f"\n### Mutations: detection by error type ({len(rows)} contracts, "
          f"{len({r['task_id'] for r in rows})} tasks)")
    keep = {tid for tid, v in labels.items() if v["preserves_intent"]} if labels else set(TASKS)
    for title, subset in (("tasks whose d preserves the intent", [r for r in rows if r["task_id"] in keep]),
                          ("all tasks", rows)):
        print(f"\n[{title}]  held = not passed (held by schema, by the checker, or unparseable)")
        types = sorted({r["mutation"] for r in subset}, key=lambda m: (m not in CONTROLS, m))
        for m in types:
            label = "false alarm" if m in CONTROLS else "detected"
            print(rate(f"{m} ({label})", [(r["task_id"], r["_held"]) for r in subset if r["mutation"] == m]))
        errors = [r for r in subset if r["mutation"] not in CONTROLS]
        print(rate("ALL ERRORS (detected)", [(r["task_id"], r["_held"]) for r in errors]))
        print(rate("ALL CONTROLS (false alarm)", [(r["task_id"], r["_held"]) for r in subset if r["mutation"] in CONTROLS]))
        unsafe = [r for r in errors if classify(r["_contract"], TASKS[r["task_id"]])["admits_unintended"]]
        print(rate("unsafe contracts PASSED", [(r["task_id"], not r["_held"]) for r in unsafe]))
    print("\nby domain (all tasks): detected errors | false alarms on controls")
    for dom in ("sales", "crm", "ops", "reports"):
        rs = [r for r in rows if domain_of(TASKS[r["task_id"]]) == dom]
        e = [r["_held"] for r in rs if r["mutation"] not in CONTROLS]
        c = [r["_held"] for r in rs if r["mutation"] in CONTROLS]
        print(f"  {dom:8s} {sum(e)}/{len(e)} | {sum(c)}/{len(c)}")
    print("\nper task: errors detected, controls held  (* d does not preserve the intent)")
    for tid in TASKS:
        rs = [r for r in rows if r["task_id"] == tid]
        if rs:
            e = [r for r in rs if r["mutation"] not in CONTROLS]
            c = [r for r in rs if r["mutation"] in CONTROLS]
            missed = sorted({r["mutation"] for r in e if not r["_held"]})
            star = "*" if labels and not labels[tid]["preserves_intent"] else " "
            print(f" {star}{tid:28s} {sum(r['_held'] for r in e):2d}/{len(e):<2d}  controls held "
                  f"{sum(r['_held'] for r in c)}/{len(c)}" + (f"  missed: {', '.join(missed)}" if missed else ""))


def natural_report(rows: list[dict], labels: dict[str, dict]) -> None:
    print(f"\n### Natural contracts ({len(rows)} contracts, {len({r['task_id'] for r in rows})} tasks)")
    for r in rows:
        t = TASKS[r["task_id"]]
        r["_true"] = goal(r["_contract"]) == goal(t.truth)
        r["_local"] = goal(r["_contract"]) == goal(labels[r["task_id"]]["implied"]) if labels else None
        r["_kind"] = classify(r["_contract"], t)["kind"]
    print(rate("contracts equal to the truth", [(r["task_id"], r["_true"]) for r in rows]))
    print(rate("held, among contracts = truth", [(r["task_id"], r["_held"]) for r in rows if r["_true"]]))
    print(rate("held, among contracts != truth", [(r["task_id"], r["_held"]) for r in rows if not r["_true"]]))
    if labels:
        print(rate("contracts equal to d's effects", [(r["task_id"], r["_local"]) for r in rows]))
        print(rate("held, among contracts = d", [(r["task_id"], r["_held"]) for r in rows if r["_local"]]))
        print(rate("held, among contracts != d", [(r["task_id"], r["_held"]) for r in rows if not r["_local"]]))
    unsafe = [r for r in rows if classify(r["_contract"], TASKS[r["task_id"]])["admits_unintended"]]
    print(rate("unsafe contracts PASSED", [(r["task_id"], not r["_held"]) for r in unsafe]))
    kinds = defaultdict(list)
    for r in rows:
        kinds[r["_kind"]].append(r)
    print("  error types vs truth (held by schema / by checker / passed): " + "; ".join(
        f"{k} {sum(r['decision'] == 'held by schema' for r in v)}/{sum(r['decision'] == 'hold' for r in v)}/"
        f"{sum(r['decision'] == 'pass' for r in v)}" for k, v in sorted(kinds.items())))

    print("\nper task (independent samples): reference = modal of the samples that pass each filter;"
          "\nno sample left = explicit stop. The intended plan's effects are gated against the reference.")
    filters = {"none": lambda r: True, "schema": lambda r: not r["schema_problems"],
               "schema + d-contract": lambda r: not r["_held"]}
    for name, keep in filters.items():
        done = stopped = exposed = 0
        for tid, t in TASKS.items():
            pool = [r["_contract"] for r in rows if r["task_id"] == tid and r["role"] == "independent" and keep(r)]
            if not [r for r in rows if r["task_id"] == tid]:
                continue
            if not pool:
                stopped += 1
                continue
            ref = modal_contract(pool)
            done += gate(ref, intended_plan(t), t.policy, t.state).execute
            exposed += bool(classify(ref, t)["admits_unintended"])
        print(f"  filter {name:20s} intended plan executes in {done}/{len(TASKS)} tasks; stopped {stopped}; "
              f"reference admits an unintended irreversible effect in {exposed}")
    print("\nper task: natural contracts = truth / held  (* d does not preserve the intent)")
    for tid in TASKS:
        rs = [r for r in rows if r["task_id"] == tid]
        if rs:
            star = "*" if labels and not labels[tid]["preserves_intent"] else " "
            wrong = [r for r in rs if not r["_true"]]
            print(f" {star}{tid:28s} = truth {sum(r['_true'] for r in rs)}/{len(rs)}; held among = truth "
                  f"{sum(r['_held'] for r in rs if r['_true'])}, among != truth {sum(r['_held'] for r in wrong)}/{len(wrong)}"
                  + (f"  [{', '.join(sorted({r['_kind'] for r in wrong}))}]" if wrong else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checks", nargs=2, default=["natural_v2", "mutations_v2"], metavar=("NATURAL", "MUTATIONS"))
    args = p.parse_args()
    labels = load_labels()
    print(f"delegation labels: {len(labels)} tasks" + (f", d preserves the intent in "
          f"{sum(v['preserves_intent'] for v in labels.values())}" if labels else " (none yet: local consistency skipped)"))
    for tid, v in labels.items():
        if not v["preserves_intent"]:
            print(f"  d does not preserve the intent: {tid}: {v['note']}")
    natural, mutated = checks(args.checks[0]), checks(args.checks[1])
    if natural:
        natural_report(natural, labels)
    if mutated:
        mutation_report(mutated, labels)


if __name__ == "__main__":
    main()
