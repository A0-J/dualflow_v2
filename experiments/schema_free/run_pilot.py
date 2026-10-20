"""Pilot of the schema-free E2E experiment (src/dualflow/schema_free.py).

For each task: the planner outputs once (shared). For each Executor Agent kind
(granular, composite) and repetition: the shared plans (one per delegation
source, one from the sent contract), then every condition in its own fresh
workspace: A, D and E once per delegation source, B, C and C* once. Every
episode is scored against the task's ground truth. One JSONL record per
episode; shared calls are recorded once, in "planner" and "plans" records, and
a "run" record holds the settings and the code version.

A hard cap counts every model call, retries included, and stops the run when
it is reached; the summary then covers the episodes completed so far.

The pilot checks that the environment produces comparable results. It does not
test a hypothesis, and its dev tasks are not part of the main analysis.

    PYTHONPATH=src python experiments/schema_free/run_pilot.py --dry-run
    PYTHONPATH=src python experiments/schema_free/run_pilot.py
    PYTHONPATH=src python experiments/schema_free/run_pilot.py --summarize-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.io import append_jsonl, read_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402
from dualflow.schema_free import (  # noqa: E402
    CONDITIONS, DELEGATION_SOURCES, EXECUTOR_TOOLS, MAX_ATTEMPTS, MAX_ROUNDS, SENDS_CONTRACT, Episode,
    PlannerOutputs, SchemaFreePlanner, Task, ToolExecutor, contract_to_json, parse_contract, planner_outputs,
    run_episode, score, shared_plans,
)
from tasks import tasks  # noqa: E402

from common import sampling_for  # noqa: E402

OUT = ROOT / "results" / "schema_free" / "pilot_v2"
N_INDEPENDENT = 5
REPEATS = 1
CALL_CAP = 516
INTERFACE_VERSION = 2


class CallBudgetExceeded(Exception):
    """Not a ValueError, so retries and episodes do not swallow it."""


class BudgetedLLM:
    def __init__(self, inner, counter: dict) -> None:
        self.inner, self.counter = inner, counter

    def generate(self, **kw):
        if self.counter["calls"] >= self.counter["cap"]:
            raise CallBudgetExceeded(f"call cap {self.counter['cap']} reached")
        self.counter["calls"] += 1
        return self.inner.generate(**kw)


def calls_upper_bound(n_tasks: int, n_executors: int, repeats: int, n_independent: int,
                      n_sources: int = len(DELEGATION_SOURCES)) -> int:
    """Calls without retries if every E episode used all its rounds."""
    planner = 2 + n_independent
    per_rep = (n_sources + 1) + n_sources * 2 * (MAX_ROUNDS - 1)  # shared plans; E: clarify + re-plan per round
    return n_tasks * (planner + n_executors * repeats * per_rep)


def _effects(es) -> list[str]:
    return sorted(" ".join(x for x in (e.kind, e.resource, e.recipient) if x) for e in es or ())


def _path(p: Path) -> str:
    """Relative to the repository when inside it, so verify_pilot.py finds it from any checkout."""
    p = p.resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()


def episode_record(ep: Episode, task: Task, repeat: int, workspace: Path) -> dict:
    rounds = []
    for r in ep.rounds:
        rounds.append({"received": r.received, "plan": [c.to_dict() for c in r.plan] if r.plan else None,
                       "plan_error": r.plan_error, "declared": _effects(r.declared),
                       "verdict": asdict(r.verdict) if r.verdict else None,
                       "steps": [{"call": s.call.to_dict(), "declared": _effects(s.declared),
                                  "actual": _effects(s.actual), "error": s.error} for s in r.steps],
                       "responses": [x.to_dict() for x in r.responses]})
    return {"type": "episode", "task_id": task.task_id, "kind": task.kind, "executor": ep.executor,
            "repeat": repeat, "condition": ep.condition, "source": ep.source, "stop_reason": ep.stop_reason,
            "workspace": _path(workspace), "rounds": rounds,
            "actual_effects": _effects(ep.actual_effects), "score": asdict(score(ep, task))}


def planner_record(task: Task, out: PlannerOutputs) -> dict:
    texts = [out.delegation, out.sent_contract, *out.independent]
    return {"type": "planner", "task_id": task.task_id, "delegation": out.delegation.value,
            "terse_delegation": task.user_request,
            "sent_contract": contract_to_json(out.sent_contract.value),
            "independent_contracts": [contract_to_json(a.value) for a in out.independent],
            "independent_contract": contract_to_json(out.independent_contract),
            "failed_attempts": [list(a.failed_attempts) for a in texts],
            "responses": [a.response.to_dict() for a in texts]}


def _git() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "src", "experiments"))}


def summarize(records: list[dict]) -> str:
    eps = [r for r in records if r["type"] == "episode"]
    run = next((r for r in records if r["type"] == "run"), {})
    lines = [f"interface v{run.get('interface_version', 1)}; commit {run.get('git', {}).get('commit', '?')[:7]}"
             f"{' (dirty)' if run.get('git', {}).get('dirty') else ''}; episodes={len(eps)}"
             f"{'; STOPPED AT THE CALL CAP' if any(r['type'] == 'stopped' for r in records) else ''}",
             "unit: episode = task x executor x repetition x condition (x source for A, D, E); dev tasks",
             f"{'cond':4s} {'source':8s} {'executor':10s} {'n':>3s} {'unsafe':>6s} {'completed':>9s} {'safe+done':>9s} "
             f"{'false rej.':>10s} {'manifest viol.':>14s} {'rounds':>6s}"]
    groups = defaultdict(list)
    for e in eps:
        groups[(e["condition"], e.get("source", "contract" if e["condition"] in SENDS_CONTRACT else "planner"),
                e["executor"])].append(e)
    order = lambda k: (CONDITIONS.index(k[0]), k[1], k[2])  # noqa: E731
    for (cond, src, ex) in sorted(groups, key=order):
        es = groups[(cond, src, ex)]
        ss = [e["score"] for e in es]
        lines.append(f"{cond:4s} {src:8s} {ex:10s} {len(ss):3d} {sum(s['unsafe'] for s in ss):6d} "
                     f"{sum(s['completed'] for s in ss):9d} {sum(s['safe_completion'] for s in ss):9d} "
                     f"{sum(s['unnecessary_rejections'] for s in ss):10d} "
                     f"{sum(e['stop_reason'] == 'manifest_violation' for e in es):14d} "
                     f"{sum(s['rounds'] for s in ss) / len(ss):6.2f}")
    lines.append("\nby task kind: unsafe / safe completion, both executors (A, D, E split by source)")
    for kind in sorted({e["kind"] for e in eps}):
        cells = []
        for (cond, src) in sorted({(c, s) for c, s, _ in groups}, key=lambda k: (CONDITIONS.index(k[0]), k[1])):
            ss = [e["score"] for e in eps if e["kind"] == kind and e["condition"] == cond
                  and e.get("source", "contract" if cond in SENDS_CONTRACT else "planner") == src]
            tag = cond if src == "contract" else f"{cond}/{src[0]}"
            cells.append(f"{tag} {sum(s['unsafe'] for s in ss)}/{sum(s['safe_completion'] for s in ss)}")
        lines.append(f"  {kind:16s} " + "  ".join(cells))
    planners = [r for r in records if r["type"] == "planner"]
    if planners:
        from tasks import DEV_TASKS, MAIN_TASKS
        truth = {t.task_id: t.truth.required for t in DEV_TASKS + MAIN_TASKS}
        correct = lambda r, key: ({x for x in parse_contract(json.dumps(r[key])).required if x.kind != "read"}  # noqa: E731
                                  == truth[r["task_id"]])
        lines.append("")
        for key in ("sent_contract", "independent_contract"):
            ok = [r["task_id"] for r in planners if correct(r, key)]
            lines.append(f"{key} equals the ground truth (reads ignored): {len(ok)}/{len(planners)}"
                         f"; wrong: {sorted(set(r['task_id'] for r in planners) - set(ok)) or 'none'}")
    stops = defaultdict(int)
    for e in eps:
        stops[e["stop_reason"]] += 1
    lines.append(f"stop reasons: {dict(stops)}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--planner-model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--executor-model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--independent", type=int, default=N_INDEPENDENT)
    p.add_argument("--cap", type=int, default=CALL_CAP)
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
    bound = calls_upper_bound(len(chosen), len(EXECUTOR_TOOLS), args.repeats, args.independent)
    print(f"split={args.split} tasks={len(chosen)} executors={list(EXECUTOR_TOOLS)} sources={list(DELEGATION_SOURCES)} "
          f"repeats={args.repeats} independent contracts={args.independent}\n"
          f"LLM calls: at most {bound} without retries; hard cap {args.cap} calls including retries "
          f"(up to {MAX_ATTEMPTS} attempts per call)")
    if args.dry_run:
        return
    if bound > args.cap:
        raise SystemExit(f"upper bound {bound} exceeds the cap {args.cap}; reduce repeats or tasks")
    if records_path.exists():
        raise SystemExit(f"{records_path} exists; use a new --out for a new run")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    sdk = OpenAI(max_retries=8, timeout=120)
    counter = {"calls": 0, "cap": args.cap}
    base_model = lambda m: "gpt-4o-mini" if m.startswith("gpt-4o-mini") else "gpt-4.1" if m.startswith("gpt-4.1") else m  # noqa: E731
    planner = SchemaFreePlanner(BudgetedLLM(OpenAILLMClient(
        sdk, args.planner_model, **sampling_for(base_model(args.planner_model))), counter))
    executor_llm = BudgetedLLM(OpenAILLMClient(
        sdk, args.executor_model, **sampling_for(base_model(args.executor_model))), counter)
    append_jsonl(records_path, {"type": "run", "interface_version": INTERFACE_VERSION, "git": _git(),
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "split": args.split,
                                "planner_model": args.planner_model, "executor_model": args.executor_model,
                                "repeats": args.repeats, "independent": args.independent, "cap": args.cap,
                                "sources": list(DELEGATION_SOURCES), "upper_bound": bound})
    work = out / "workspaces"
    try:
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
                        for src in (("planner",) if cond in SENDS_CONTRACT else DELEGATION_SOURCES):
                            name = cond.replace("*", "_star") + ("" if cond in SENDS_CONTRACT else f"_{src}")
                            root = work / task.task_id / ex_name / str(rep) / name
                            if root.exists():
                                shutil.rmtree(root)
                            ep = run_episode(cond, task, planner, outputs, ex_name, executor, plans, root, source=src)
                            append_jsonl(records_path, episode_record(ep, task, rep, root))
            print(f"{i}/{len(chosen)} {task.task_id} calls={counter['calls']}", flush=True)
    except CallBudgetExceeded as e:
        append_jsonl(records_path, {"type": "stopped", "reason": str(e), "calls": counter["calls"]})
        print(f"STOPPED: {e}")
    summary = summarize(read_jsonl(records_path)) + f"\ncalls used (retries included): {counter['calls']}"
    print(summary)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
