"""Pilot of the schema-free E2E experiment (src/dualflow/schema_free.py).

For each task: the planner outputs once (shared). For each Executor Agent kind
(granular, composite) and repetition: the two shared plans, then every
condition A, B, C, C*, D, E in its own fresh workspace, scored against the
task's ground truth. One JSONL record per episode; the shared calls are
recorded once, in records of type "planner" and "plans".

The pilot checks that the environment produces comparable results:
that safe and unsafe executions both occur, that C vs C* vs D separate the gate
reference from what is sent, and that completion and unsafe effects are
measured independently. It does not test a hypothesis.

    PYTHONPATH=src python experiments/schema_free/run_pilot.py --dry-run
    PYTHONPATH=src python experiments/schema_free/run_pilot.py
    PYTHONPATH=src python experiments/schema_free/run_pilot.py --summarize-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.io import append_jsonl, read_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402
from dualflow.schema_free import (  # noqa: E402
    CONDITIONS, EXECUTOR_TOOLS, MAX_ATTEMPTS, MAX_ROUNDS, Episode, PlannerOutputs, SchemaFreePlanner, Task,
    ToolExecutor, contract_to_json, planner_outputs, run_episode, score, shared_plans,
)
from tasks import tasks  # noqa: E402

from common import sampling_for  # noqa: E402

OUT = ROOT / "results" / "schema_free" / "pilot"
N_INDEPENDENT = 5
REPEATS = 3


def calls_upper_bound(n_tasks: int, n_executors: int, repeats: int, n_independent: int) -> tuple[int, int]:
    """(calls without retries, calls if every call needed all retries)."""
    planner = 2 + n_independent
    per_rep = 2 + 2 * (MAX_ROUNDS - 1)  # two shared plans; E: clarify + re-plan per extra round
    base = n_tasks * (planner + n_executors * repeats * per_rep)
    return base, base * MAX_ATTEMPTS


def _effects(es) -> list[str]:
    return sorted(" ".join(x for x in (e.kind, e.resource, e.recipient) if x) for e in es or ())


def episode_record(ep: Episode, task: Task, repeat: int) -> dict:
    rounds = []
    for r in ep.rounds:
        rounds.append({"received": r.received, "plan": [c.to_dict() for c in r.plan] if r.plan else None,
                       "plan_error": r.plan_error, "declared": _effects(r.declared),
                       "verdict": asdict(r.verdict) if r.verdict else None,
                       "steps": [{"call": s.call.to_dict(), "declared": _effects(s.declared),
                                  "actual": _effects(s.actual), "error": s.error} for s in r.steps],
                       "responses": [x.to_dict() for x in r.responses]})
    return {"type": "episode", "task_id": task.task_id, "kind": task.kind, "executor": ep.executor,
            "repeat": repeat, "condition": ep.condition, "stop_reason": ep.stop_reason, "rounds": rounds,
            "actual_effects": _effects(ep.actual_effects), "score": asdict(score(ep, task))}


def planner_record(task: Task, out: PlannerOutputs) -> dict:
    texts = [out.delegation, out.sent_contract, *out.independent]
    return {"type": "planner", "task_id": task.task_id, "delegation": out.delegation.value,
            "sent_contract": contract_to_json(out.sent_contract.value),
            "independent_contracts": [contract_to_json(a.value) for a in out.independent],
            "independent_contract": contract_to_json(out.independent_contract),
            "failed_attempts": [list(a.failed_attempts) for a in texts],
            "responses": [a.response.to_dict() for a in texts]}


def summarize(records: list[dict]) -> str:
    eps = [r for r in records if r["type"] == "episode"]
    lines = [f"episodes={len(eps)}  (unit: episode = task x executor x repetition x condition; tasks are dev tasks)"]
    lines.append(f"{'condition':9s} {'executor':10s} {'n':>4s} {'unsafe':>7s} {'completed':>10s} {'safe+done':>10s} "
                 f"{'unneeded rej.':>13s} {'mean rounds':>11s}")
    groups = defaultdict(list)
    for e in eps:
        groups[(e["condition"], e["executor"])].append(e["score"])
    for (cond, ex), ss in sorted(groups.items(), key=lambda kv: (CONDITIONS.index(kv[0][0]), kv[0][1])):
        n = len(ss)
        lines.append(f"{cond:9s} {ex:10s} {n:4d} {sum(s['unsafe'] for s in ss):7d} {sum(s['completed'] for s in ss):10d} "
                     f"{sum(s['safe_completion'] for s in ss):10d} {sum(s['unnecessary_rejections'] for s in ss):13d} "
                     f"{sum(s['rounds'] for s in ss) / n:11.2f}")
    lines.append("\nby task kind: unsafe / safe completion per condition (both executors)")
    kinds = sorted({e["kind"] for e in eps})
    for kind in kinds:
        cells = []
        for cond in CONDITIONS:
            ss = [e["score"] for e in eps if e["kind"] == kind and e["condition"] == cond]
            cells.append(f"{cond} {sum(s['unsafe'] for s in ss)}/{sum(s['safe_completion'] for s in ss)}")
        lines.append(f"  {kind:16s} " + "  ".join(cells))
    stops = defaultdict(int)
    for e in eps:
        stops[e["stop_reason"]] += 1
    lines.append(f"\nstop reasons: {dict(stops)}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--planner-model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--executor-model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--independent", type=int, default=N_INDEPENDENT)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    records_path = out / "episodes.jsonl"

    if args.summarize_only:
        print(summarize(read_jsonl(records_path)))
        return
    chosen = tasks(args.split)
    base, worst = calls_upper_bound(len(chosen), len(EXECUTOR_TOOLS), args.repeats, args.independent)
    print(f"split={args.split} tasks={len(chosen)} executors={list(EXECUTOR_TOOLS)} repeats={args.repeats} "
          f"independent contracts={args.independent}\nLLM calls: at most {base} without retries "
          f"(at most {worst} if every call needed all {MAX_ATTEMPTS} attempts)")
    if args.dry_run:
        return
    if records_path.exists():
        raise SystemExit(f"{records_path} exists; move it away before a new run")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    sdk = OpenAI(max_retries=8, timeout=120)
    base_model = lambda m: "gpt-4o-mini" if m.startswith("gpt-4o-mini") else "gpt-4.1" if m.startswith("gpt-4.1") else m  # noqa: E731
    planner = SchemaFreePlanner(OpenAILLMClient(sdk, args.planner_model, **sampling_for(base_model(args.planner_model))))
    executor_llm = OpenAILLMClient(sdk, args.executor_model, **sampling_for(base_model(args.executor_model)))
    work = out / "workspaces"
    for i, task in enumerate(chosen, 1):
        outputs = planner_outputs(planner, task, args.independent)
        append_jsonl(records_path, planner_record(task, outputs))
        for ex_name, tools in EXECUTOR_TOOLS.items():
            executor = ToolExecutor(executor_llm, tools)
            for rep in range(args.repeats):
                plans = shared_plans(executor, task, outputs)
                append_jsonl(records_path, {
                    "type": "plans", "task_id": task.task_id, "executor": ex_name, "repeat": rep,
                    **{k: ({"error": str(v)} if isinstance(v, Exception) else
                           {"plan": [c.to_dict() for c in v.value], "failed_attempts": list(v.failed_attempts),
                            "response": v.response.to_dict()}) for k, v in plans.items()}})
                for cond in CONDITIONS:
                    root = work / task.task_id / ex_name / str(rep) / cond.replace("*", "_star")
                    if root.exists():
                        shutil.rmtree(root)
                    ep = run_episode(cond, task, planner, outputs, ex_name, executor, plans, root)
                    append_jsonl(records_path, episode_record(ep, task, rep))
        print(f"{i}/{len(chosen)} {task.task_id}", flush=True)
    summary = summarize(read_jsonl(records_path))
    print(summary)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
