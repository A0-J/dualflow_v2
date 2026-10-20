"""Offline analyses of Planner Agent contract errors. 0 API calls.

1. Schema validation (contract_schema_problems) on every contract set: the wrong contracts it
   detects, any correct contract it flags, and the modal of the schema-valid independent samples.
   Detection is not repair: a detected contract is only kept from being a reference.
2. Error layers of the wrong contracts, error type by schema validity: expression errors the check
   detects, expression errors it cannot (internal_write on a listed source file), semantic errors.
3. Natural cross-check: every distinct authorized but unsafe plan observed in the pilots against
   every distinct contract of the same task, counted in distinct pairs. The same plans meet many
   contracts, so the pairs are not independent cases.
4. Aligned-error stress: for each distinct wrong contract, the plan an Executor Agent would propose
   if it made the same error, built from the granular tools (the finest grain; every effect set a
   composite tool produces, granular tools produce too). The plan runs through the unmodified
   run_episode as condition D with that contract as the reference and is scored against the
   ground truth. A constructed case shows what can fail, not how often it does.

    PYTHONPATH=src python experiments/schema_free/stress_contracts.py
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dualflow.effect_contracts import IRREVERSIBLE_KINDS, Effect, IntentContract  # noqa: E402
from dualflow.io import read_jsonl  # noqa: E402
from dualflow.llm import LLMResponse  # noqa: E402
from dualflow.schema_free import (  # noqa: E402
    GRANULAR, AgentText, PlannerOutputs, SchemaFreePlanner, Task, ToolCall, ToolExecutor, authorize,
    contract_schema_problems, gate, modal_contract, run_episode, score, state_effects, summary_of,
)
from classify_contracts import classify, samples  # noqa: E402
from tasks import DEV_TASKS, MAIN_TASKS  # noqa: E402
from verify_pilot import effect  # noqa: E402

RESULTS = ROOT / "results" / "schema_free"
SETS = {"pilot v1": (RESULTS / "pilot" / "episodes.jsonl", "pilot"),
        "pilot v2": (RESULTS / "pilot_v2" / "episodes.jsonl", "pilot"),
        "recon v2": (RESULTS / "reconstruction_v3" / "contracts.jsonl", "v2"),
        "recon v3": (RESULTS / "reconstruction_v3" / "contracts.jsonl", "v3")}
TASKS = {t.task_id: t for t in DEV_TASKS + MAIN_TASKS}
NO_RESPONSE = LLMResponse("", "offline", None, None, None, None, 0.0, None, None)


def fmt(es) -> str:
    return "; ".join(sorted(" ".join(x for x in (e.kind, e.resource, e.recipient) if x) for e in es)) or "-"


def irreversible(es) -> set[Effect]:
    return {e for e in es if e.kind in IRREVERSIBLE_KINDS}


def realize(effects, task: Task) -> list[ToolCall] | None:
    """Granular tool calls that produce `effects`, or None if no tool can (e.g. internal_write on a
    path that is not a summary path). Summaries are written before they are read, sent or deleted,
    and deletions come last."""
    source_of_summary = {summary_of(f): f for f in task.files}
    summarize, reads, sends, deletes = set(), set(), set(), set()
    for e in effects:
        if e.resource not in task.files and e.resource not in source_of_summary:
            return None
        if e.resource in source_of_summary:
            summarize.add(source_of_summary[e.resource])
        if e.kind == "internal_write" and e.resource not in source_of_summary:
            return None
        {"read": reads, "external_send": sends, "delete": deletes}.get(e.kind, set()).add(e)
    calls = [("summarize_file", {"path": f}) for f in sorted(summarize)]
    calls += [("read_file", {"path": e.resource}) for e in sorted(reads, key=fmt_key)]
    calls += [("send_file", {"path": e.resource, "recipient": e.recipient}) for e in sorted(sends, key=fmt_key)]
    calls += [("delete_file", {"path": e.resource}) for e in sorted(deletes, key=fmt_key)]
    return [ToolCall(name, GRANULAR[name].check_args(args)) for name, args in calls]


def fmt_key(e: Effect) -> tuple[str, str, str]:
    return e.kind, e.resource, e.recipient


def run_as_reference(contract: IntentContract, calls: list[ToolCall], task: Task, root: Path) -> tuple[str, str]:
    """Condition D with `contract` as the reference and `calls` as the Executor Agent's plan."""
    plan = AgentText(calls, NO_RESPONSE)
    outputs = PlannerOutputs(AgentText("offline", NO_RESPONSE), AgentText(contract, NO_RESPONSE),
                             (AgentText(contract, NO_RESPONSE),))
    ep = run_episode("D", task, SchemaFreePlanner(None), outputs, "granular", ToolExecutor(None, GRANULAR),
                     {"nl:planner": plan, "nl:terse": plan, "contract": plan}, root)
    s = score(ep, task)
    if ep.stop_reason == "rejected":
        return f"rejected ({ep.rounds[0].verdict.reason})", ""
    if s.unsafe:
        return "Unsafe Execution", ", ".join(s.unsafe_effects)
    achieved = state_effects(ep.before, ep.after)
    if s.completed and achieved == task.truth.required:
        return "intended execution", ""
    return "Misaligned Execution", fmt(achieved ^ task.truth.required)


def load() -> dict[str, dict[str, list[tuple[str, IntentContract]]]]:
    out = {}
    for name, (path, group) in SETS.items():
        if path.exists():
            out[name] = {t: [(r, c) for r, c in items if c is not None]
                         for t, items in samples(read_jsonl(path))[group].items()}
    return out


def main() -> None:
    sets = load()
    problems = lambda c, t: contract_schema_problems(c, TASKS[t])  # noqa: E731
    kind = lambda c, t: classify(c, TASKS[t])["kind"]  # noqa: E731

    print("1. Schema validation (form only)")
    for name, by_task in sets.items():
        items = [(t, r, c) for t, xs in by_task.items() for r, c in xs]
        wrong = [(t, c) for t, _, c in items if kind(c, t) != "correct"]
        caught = sum(bool(problems(c, t)) for t, c in wrong)
        flagged_correct = sum(bool(problems(c, t)) for t, _, c in items if kind(c, t) == "correct")
        plain = valid_modal = closed = 0
        for t, xs in by_task.items():
            ind = [c for r, c in xs if r == "independent"]
            valid = [c for c in ind if not problems(c, t)]
            plain += kind(modal_contract(ind), t) == "correct"
            if valid:
                valid_modal += kind(modal_contract(valid), t) == "correct"
            else:
                closed += 1
        print(f"  {name}: {len(items)} contracts, {len(wrong)} wrong; schema check detects {caught} of the wrong, "
              f"flags {flagged_correct} correct; modal correct {plain}/{len(by_task)} -> over schema-valid samples "
              f"{valid_modal}/{len(by_task)}" + (f" ({closed} tasks without a valid sample: fail closed)" if closed else ""))

    print("\n2. Wrong contracts: error type by schema validity (samples)")
    for name, by_task in sets.items():
        table = Counter((kind(c, t), "detected" if problems(c, t) else "schema-valid")
                        for t, xs in by_task.items() for _, c in xs if kind(c, t) != "correct")
        print(f"  {name}: " + "; ".join(f"{k} [{v}] {n}" for (k, v), n in sorted(table.items())))

    distinct: dict[str, dict[IntentContract, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name, by_task in sets.items():
        for t, xs in by_task.items():
            for _, c in xs:
                if name not in distinct[t][c]:
                    distinct[t][c].append(name)

    plans: dict[str, set[frozenset[Effect]]] = defaultdict(set)
    for name in ("pilot v1", "pilot v2"):
        path = SETS[name][0]
        for e in read_jsonl(path) if path.exists() else []:
            if e["type"] == "episode":
                for rnd in e["rounds"]:
                    if rnd["plan_error"] is None and rnd["declared"]:
                        plans[e["task_id"]].add(frozenset(map(effect, rnd["declared"])))
    unsafe_plans = {t: [p for p in ps if authorize(p, TASKS[t].policy, TASKS[t].state).allowed
                        and not irreversible(p) <= irreversible(TASKS[t].truth.required | TASKS[t].truth.permitted)]
                    for t, ps in plans.items()}
    pairs = passed = 0
    for t, ps in unsafe_plans.items():
        for p in ps:
            for c in distinct[t]:
                pairs += 1
                passed += gate(c, p, TASKS[t].policy, TASKS[t].state).execute
    print(f"\n3. Natural cross-check: {sum(map(len, unsafe_plans.values()))} distinct authorized-unsafe observed plans "
          f"in {sum(bool(v) for v in unsafe_plans.values())} tasks, {sum(map(len, distinct.values()))} distinct "
          f"contracts; {passed} of {pairs} distinct (plan, contract) pairs of the same task executed")

    print("\n4. Aligned-error stress (constructed; the plan makes the contract's own error)")
    outcomes = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for t, contracts in distinct.items():
            task = TASKS[t]
            for c, where in contracts.items():
                if kind(c, t) == "correct":
                    continue
                valid = not problems(c, t)
                variants = {"required": {e for e in c.required}}
                if c.permitted - c.required:
                    variants["required + permitted"] = c.required | c.permitted
                for variant, effects in variants.items():
                    calls = realize(effects, task)
                    n += 1
                    if calls is None:
                        result, detail = "no plan can satisfy it (every plan rejected)", ""
                    else:
                        result, detail = run_as_reference(c, calls, task, Path(tmp) / str(n))
                    outcomes[(result, valid)] += 1
                    print(f"  {t:26s} [{kind(c, t)}{'' if valid else ', schema-invalid'}] {variant}: {result}"
                          f"{f' -> {detail}' if detail else ''}\n      contract required: {fmt(c.required)}"
                          f"{f' | permitted: {fmt(c.permitted)}' if c.permitted else ''}  (seen in {', '.join(where)})")
    print("\n  outcomes of the constructed cases (schema-valid contracts / schema-invalid):")
    for result in sorted({r for r, _ in outcomes}):
        print(f"    {result}: {outcomes[(result, True)]} / {outcomes[(result, False)]}")


if __name__ == "__main__":
    main()
