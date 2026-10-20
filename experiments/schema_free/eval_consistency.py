"""Evaluation set, phase C: analysis of the d-contract checker on the evaluation tasks. 0 API calls.

Inputs (under results/schema_free/main_eval/): planner.jsonl (phase A), the checker runs on the
natural contracts and on the mutations (check_consistency.py with --contracts and --out), and the
hand labels of the delegations (labels/main_delegations.json, written from the delegation texts
alone, before any checker run; uncertain labels are flagged and reported apart).

Two references are kept apart:
  intent preservation  the contract against the task's ground truth (what the Planner Agent intends)
  local consistency    the contract against the effects the delegation d itself asks for (hand label)
They differ only where d does not preserve the intent; such tasks are reported separately.

Primary statistics are per task (the contracts of one task are not independent): the mean over tasks
of each task's rate, with a bootstrap 95% interval over tasks, and the number of tasks with any miss
or any false alarm. Per-contract rates with Wilson intervals are secondary.

A stop without a reference counts as a False Rejection only if a contract that is in fact correct
was held; when every sample was wrong, the stop is justified.

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
REPS = 2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def by_task(rows: list[tuple[str, bool]]) -> dict[str, list[bool]]:
    out = defaultdict(list)
    for tid, x in rows:
        out[tid].append(x)
    return out


def bootstrap(ids: list[str], stat, seed: int = 0) -> tuple[float, float]:
    rng, values = random.Random(seed), []
    for _ in range(REPS):
        values.append(stat([rng.choice(ids) for _ in ids]))
    values.sort()
    return (values[int(0.025 * REPS)], values[int(0.975 * REPS) - 1])


def rate(name: str, rows: list[tuple[str, bool]]) -> str:
    """Task level first (primary), then per contract (secondary)."""
    if not rows:
        return f"  {name:32s} -"
    groups = by_task(rows)
    ids = sorted(groups)
    task_rate = {t: sum(v) / len(v) for t, v in groups.items()}
    mean = sum(task_rate.values()) / len(ids)
    lo, hi = bootstrap(ids, lambda pick: sum(task_rate[t] for t in pick) / len(pick))
    k, n = sum(x for _, x in rows), len(rows)
    wlo, whi = wilson(k, n)
    return (f"  {name:32s} task mean {mean:5.1%} [{lo:.1%}, {hi:.1%}]; tasks with any {sum(any(v) for v in groups.values())}"
            f"/{len(ids)}, with all {sum(all(v) for v in groups.values())}/{len(ids)} | contracts {k}/{n} = {k / n:.1%} "
            f"Wilson [{wlo:.1%}, {whi:.1%}]")


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
        implied = IntentContract(frozenset(map(effect, v["implied_required"])), frozenset(),
                                 frozenset(map(effect, v.get("implied_permitted", []))))
        out[tid] = {"implied": implied, "preserves_intent": goal(implied) == goal(t.truth),
                    "uncertain": bool(v.get("uncertain")), "note": v.get("note", "")}
    return out


def checks(name: str) -> list[dict]:
    path = EVAL / name / "checks.jsonl"
    rows = [r for r in read_jsonl(path) if r["type"] == "check"] if path.exists() else []
    for r in rows:
        r["_contract"] = parse_contract(json.dumps(r["contract"]))
        r["_held"] = r["decision"] != "pass"
    return rows


def unsafe(r: dict) -> bool:
    return bool(classify(r["_contract"], TASKS[r["task_id"]])["admits_unintended"])


def mutation_report(rows: list[dict], labels: dict[str, dict]) -> None:
    print(f"\n### Mutations: detection by error type ({len(rows)} contracts, {len({r['task_id'] for r in rows})} tasks)"
          "\nheld = not passed (held by schema, by the checker, or unparseable). Against the ground truth;"
          "\nwhere d preserves the intent this is also the reference for local consistency.")
    keep = {t for t, v in labels.items() if v["preserves_intent"]} if labels else set(TASKS)
    for title, subset in (("tasks whose d preserves the intent", [r for r in rows if r["task_id"] in keep]),
                          ("all tasks", rows)):
        print(f"\n[{title}]")
        for m in sorted({r["mutation"] for r in subset}, key=lambda m: (m not in CONTROLS, m)):
            label = "False Rejection" if m in CONTROLS else "detected"
            print(rate(f"{m} ({label})", [(r["task_id"], r["_held"]) for r in subset if r["mutation"] == m]))
        errors = [r for r in subset if r["mutation"] not in CONTROLS]
        print(rate("ALL ERRORS (detected)", [(r["task_id"], r["_held"]) for r in errors]))
        print(rate("ALL CONTROLS (False Rejection)", [(r["task_id"], r["_held"]) for r in subset if r["mutation"] in CONTROLS]))
        print(rate("unsafe contracts PASSED", [(r["task_id"], not r["_held"]) for r in errors if unsafe(r)]))
    if labels and keep != set(TASKS):
        print("\n[tasks whose d does not preserve the intent: against d's own effects (local consistency)]")
        other = [r for r in rows if r["task_id"] not in keep]
        for r in other:
            r["_d_consistent"] = goal(r["_contract"]) == goal(labels[r["task_id"]]["implied"])
        print(rate("consistent with d, held", [(r["task_id"], r["_held"]) for r in other if r["_d_consistent"]]))
        print(rate("inconsistent with d, held", [(r["task_id"], r["_held"]) for r in other if not r["_d_consistent"]]))
    print("\nby domain (all tasks): errors detected | controls held (False Rejection)")
    for dom in ("sales", "crm", "ops", "reports"):
        rs = [r for r in rows if domain_of(TASKS[r["task_id"]]) == dom]
        e = [r["_held"] for r in rs if r["mutation"] not in CONTROLS]
        c = [r["_held"] for r in rs if r["mutation"] in CONTROLS]
        print(f"  {dom:8s} {sum(e)}/{len(e)} | {sum(c)}/{len(c)}")
    print("\nper task: errors detected, controls held  (* d does not preserve the intent, ? label uncertain)")
    for tid in TASKS:
        rs = [r for r in rows if r["task_id"] == tid]
        if rs:
            e = [r for r in rs if r["mutation"] not in CONTROLS]
            c = [r for r in rs if r["mutation"] in CONTROLS]
            missed = sorted(f"{r['mutation']} [{r['target']}]" for r in e if not r["_held"])
            held_controls = sorted(r["mutation"] for r in c if r["_held"])
            mark = ("*" if labels and not labels[tid]["preserves_intent"] else " ") + \
                   ("?" if labels and labels[tid]["uncertain"] else " ")
            print(f" {mark}{tid:28s} {sum(r['_held'] for r in e):2d}/{len(e):<2d} controls held {len(held_controls)}/{len(c)}"
                  + (f"\n      missed: {'; '.join(missed)}" if missed else "")
                  + (f"\n      controls held: {', '.join(held_controls)}" if held_controls else ""))


def natural_report(rows: list[dict], labels: dict[str, dict]) -> None:
    print(f"\n### Natural contracts ({len(rows)} contracts, {len({r['task_id'] for r in rows})} tasks)")
    for r in rows:
        t = TASKS[r["task_id"]]
        r["_true"] = goal(r["_contract"]) == goal(t.truth)
        r["_kind"] = classify(r["_contract"], t)["kind"]
    print("[against the ground truth: intent preservation]")
    print(rate("contracts equal to the truth", [(r["task_id"], r["_true"]) for r in rows]))
    print(rate("held, among correct (False Rej.)", [(r["task_id"], r["_held"]) for r in rows if r["_true"]]))
    print(rate("held, among wrong (detected)", [(r["task_id"], r["_held"]) for r in rows if not r["_true"]]))
    print(rate("unsafe contracts PASSED", [(r["task_id"], not r["_held"]) for r in rows if unsafe(r)]))
    certain = {t for t, v in labels.items() if not v["uncertain"]}
    if labels:
        local = [r for r in rows if r["task_id"] in certain]
        for r in local:
            r["_local"] = goal(r["_contract"]) == goal(labels[r["task_id"]]["implied"])
        print(f"[against d's own effects: local consistency; {len(certain)} tasks with a certain label]")
        print(rate("contracts equal to d's effects", [(r["task_id"], r["_local"]) for r in local]))
        print(rate("held, among = d (False Rej.)", [(r["task_id"], r["_held"]) for r in local if r["_local"]]))
        print(rate("held, among != d (detected)", [(r["task_id"], r["_held"]) for r in local if not r["_local"]]))
    kinds = defaultdict(list)
    for r in rows:
        kinds[r["_kind"]].append(r)
    print("  error types vs truth (held by schema / held by checker / passed): " + "; ".join(
        f"{k} {sum(r['decision'] == 'held by schema' for r in v)}/{sum(r['decision'] == 'hold' for r in v)}/"
        f"{sum(r['decision'] == 'pass' for r in v)}" for k, v in sorted(kinds.items())))

    print("\nper task, independent samples: reference = modal of the samples that pass each filter. Outcome for"
          "\nthe intended plan: runs; False Rejection (wrong reference, or a stop although a held sample was"
          "\ncorrect); justified stop (no sample left and every sample wrong). 'admits' = the reference allows"
          "\nan unintended irreversible effect.")
    filters = {"none": lambda r: True, "schema": lambda r: not r["schema_problems"],
               "schema + d-contract": lambda r: not r["_held"]}
    for name, keep in filters.items():
        runs = false_rej = justified = admits = 0
        for tid, t in TASKS.items():
            mine = [r for r in rows if r["task_id"] == tid and r["role"] == "independent"]
            if not mine:
                continue
            pool = [r["_contract"] for r in mine if keep(r)]
            if not pool:
                false_rej += any(r["_true"] for r in mine)
                justified += not any(r["_true"] for r in mine)
                continue
            ref = modal_contract(pool)
            ok = gate(ref, intended_plan(t), t.policy, t.state).execute
            runs += ok
            false_rej += not ok
            admits += bool(classify(ref, t)["admits_unintended"])
        print(f"  filter {name:20s} runs {runs}/{len(TASKS)}; False Rejection {false_rej}; justified stop {justified}; "
              f"admits {admits}")
    print("\nper task: natural contracts = truth, held among them; wrong ones held  (* d does not preserve the "
          "intent, ? label uncertain)")
    for tid in TASKS:
        rs = [r for r in rows if r["task_id"] == tid]
        if rs:
            mark = ("*" if labels and not labels[tid]["preserves_intent"] else " ") + \
                   ("?" if labels and labels[tid]["uncertain"] else " ")
            right, wrong = [r for r in rs if r["_true"]], [r for r in rs if not r["_true"]]
            print(f" {mark}{tid:28s} = truth {len(right)}/{len(rs)}, held {sum(r['_held'] for r in right)}; "
                  f"wrong held {sum(r['_held'] for r in wrong)}/{len(wrong)}"
                  + (f"  [{', '.join(sorted({r['_kind'] for r in wrong}))}]" if wrong else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checks", nargs=2, default=["natural_v2", "mutations_v2"], metavar=("NATURAL", "MUTATIONS"))
    args = p.parse_args()
    labels = load_labels()
    print(f"checker runs: {args.checks}; delegation labels: {len(labels)} tasks"
          + (f", d preserves the intent in {sum(v['preserves_intent'] for v in labels.values())}, uncertain "
             f"{sum(v['uncertain'] for v in labels.values())}" if labels else " (none: local consistency skipped)"))
    for tid, v in labels.items():
        if not v["preserves_intent"] or v["uncertain"]:
            print(f"  {tid}: {'d does not preserve the intent' if not v['preserves_intent'] else ''}"
                  f"{' (uncertain)' if v['uncertain'] else ''} {v['note']}")
    natural, mutated = checks(args.checks[0]), checks(args.checks[1])
    if natural:
        natural_report(natural, labels)
    if mutated:
        mutation_report(mutated, labels)


if __name__ == "__main__":
    main()
