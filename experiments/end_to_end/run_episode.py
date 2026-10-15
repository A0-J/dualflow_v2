"""Run one end-to-end DualFlow episode (src/dualflow/runtime.py).

The Planner Agent writes the delegation message d from the user request,
and the Executor Agent's tool call runs in a sandbox only if the Execution
Gate permits it. After a rejection, the chosen recovery policy decides
whether the Planner Agent writes a new d.

This is a smoke test of the runtime, not an evaluation. The task is built
from a canonical C1-C4 scenario (experiments/canonical/scenarios.py):

  user request  the canonical delegation sentence of the case's clarity
  C_task        the Planner Agent's private task context for that delegation
  π, σ          the case's authorization configuration
  o*            the intended operation (evaluation only, never shown)

Usage:
    PYTHONPATH=src python experiments/end_to_end/run_episode.py --domain audit --case C2 --dry-run
    PYTHONPATH=src python experiments/end_to_end/run_episode.py --domain audit --case C2 --recovery clarify
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.flows import AuthorizationPolicy, AuthorizationState, Proposal  # noqa: E402
from dualflow.io import append_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402
from dualflow.roles import ExecutorAgent, PlannerAgent  # noqa: E402
from dualflow.runtime import Task, clarify_on_semantic_rejection, evaluate_episode, no_recovery, run_episode  # noqa: E402
from dualflow.sandbox import Sandbox  # noqa: E402

from common import N_SAMPLES, PLANNER_MODEL, sampling_for  # noqa: E402
from scenarios import CASES, DOMAINS, IRREVERSIBLE, authorization_config  # noqa: E402

OUT = ROOT / "results" / "end_to_end" / "smoke"
RECOVERY = {"none": no_recovery, "clarify": clarify_on_semantic_rejection}
FILE_LINE = re.compile(r"^- ([\w.-]+/[\w./-]+): (.+)$", re.MULTILINE)


def canonical_task(domain: str, case: str) -> tuple[Task, Proposal, dict[str, str]]:
    d, clarity = DOMAINS[domain], CASES[case]
    policy, state = authorization_config(domain, case)
    task = Task(user_request=d[clarity]["delegation"], task_context=d[clarity]["private_context"],
                runtime_context=d["context"], policy=AuthorizationPolicy.from_dict(policy),
                state=AuthorizationState.from_dict(state))
    files = {path: f"{desc}\n" for path, desc in FILE_LINE.findall(d["context"])}
    return task, Proposal.from_dict(d[clarity]["intended"]), files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=sorted(DOMAINS), default="audit")
    p.add_argument("--case", choices=sorted(CASES), default="C2")
    p.add_argument("--recovery", choices=sorted(RECOVERY), default="none")
    p.add_argument("--planner-model", default=PLANNER_MODEL)
    p.add_argument("--executor-model", default="gpt-4o-mini")
    p.add_argument("--restatements", type=int, default=N_SAMPLES)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="print the task; no API calls, no files")
    args = p.parse_args()

    task, intended, files = canonical_task(args.domain, args.case)
    if args.dry_run:
        print(f"user request: {task.user_request}\nC_task: {task.task_context}\nfiles: {sorted(files)}\n"
              f"policy: {task.policy}\nstate: {task.state}\nintended (o*): {intended.to_dict()}")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    sdk = OpenAI(max_retries=8, timeout=120)
    planner = PlannerAgent(OpenAILLMClient(sdk, args.planner_model, **sampling_for(args.planner_model)))
    executor = ExecutorAgent(OpenAILLMClient(sdk, args.executor_model, **sampling_for(args.executor_model)))
    run_id = f"{args.domain}_{args.case}_{args.recovery}_{time.strftime('%Y%m%d-%H%M%S')}"
    sandbox = Sandbox(OUT / "sandboxes" / run_id, files)

    episode = run_episode(task, planner, executor, sandbox, n_restatements=args.restatements,
                          max_rounds=args.max_rounds, recovery=RECOVERY[args.recovery])
    outcome = evaluate_episode(episode, task, intended, IRREVERSIBLE)
    append_jsonl(OUT / "episodes.jsonl", {
        "run_id": run_id, "domain": args.domain, "case": args.case, "recovery": args.recovery,
        "planner_model": args.planner_model, "executor_model": args.executor_model,
        "intended": intended.to_dict(), "round_outcomes": [o.value for o in outcome.round_outcomes],
        "task_completed": outcome.task_completed, "unsafe_execution": outcome.unsafe_execution,
        **episode.to_dict()})
    for i, r in enumerate(episode.rounds, 1):
        print(f"round {i}: d={r.delegation!r}\n  P_exec={r.proposal.to_dict()} V_sem={r.decision.semantic.passed} "
              f"V_auth={r.decision.authorization.allowed} -> {outcome.round_outcomes[i - 1].value}"
              f"{'' if r.result is None else f' | {r.result.effect} {r.result.path}'}")
    print(f"stop: {episode.stop_reason}, task completed: {outcome.task_completed}, sandbox: {sandbox.root}")


if __name__ == "__main__":
    main()
