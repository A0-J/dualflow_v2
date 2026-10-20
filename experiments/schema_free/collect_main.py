"""Evaluation set, phase A: the Planner Agent's outputs for each evaluation task.

Per task: one delegation message d, then one "sent" and N "independent" contracts with effect
vocabulary v3, all from the user request, the private context and the shared catalog, as in the
pilot (planner_outputs). No Executor Agent and no checker run here. Unparseable outputs are kept in
full. The delegations are then labelled by hand (intended effects of each d, and whether d preserves
the task's intent) before the checker sees anything: see eval_consistency.py.

    PYTHONPATH=src python experiments/schema_free/collect_main.py --dry-run
    PYTHONPATH=src python experiments/schema_free/collect_main.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.io import append_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402
from dualflow.schema_free import (  # noqa: E402
    EFFECT_VOCABULARIES, MAX_ATTEMPTS, SchemaFreePlanner, contract_to_json,
)
from run_pilot import BudgetedLLM, CallBudgetExceeded, _git  # noqa: E402
from tasks import tasks  # noqa: E402

from common import sampling_for  # noqa: E402

OUT = ROOT / "results" / "schema_free" / "main_eval" / "planner.jsonl"
N_INDEPENDENT = 5
VOCABULARY = "v3"


class Recording:
    """Keeps the text of the last response, so an output that fails to parse is kept in full."""

    def __init__(self, inner) -> None:
        self.inner, self.last = inner, None

    def generate(self, **kw):
        self.last = self.inner.generate(**kw)
        return self.last


def attempt(call, recorder: Recording) -> dict:
    failed = []
    for _ in range(MAX_ATTEMPTS):
        try:
            a = call()
            return {"value": a.value, "failed_attempts": failed, "response": a.response.to_dict()}
        except ValueError as e:
            failed.append({"error": str(e)[:300], "text": recorder.last.text if recorder.last else None})
    return {"value": None, "failed_attempts": failed, "response": None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="main")
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--independent", type=int, default=N_INDEPENDENT)
    p.add_argument("--cap", type=int, default=1000)  # runaway guard only
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    chosen, out = tasks(args.split), Path(args.out)
    print(f"split={args.split} tasks={len(chosen)}: per task 1 delegation + {1 + args.independent} contracts "
          f"(vocabulary {VOCABULARY}) = {len(chosen) * (2 + args.independent)} calls without retries")
    if args.dry_run:
        return
    if out.exists():
        raise SystemExit(f"{out} exists; use a new --out for a new run")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    counter = {"calls": 0, "cap": args.cap}
    base = "gpt-4o-mini" if args.model.startswith("gpt-4o-mini") else args.model
    recorder = Recording(BudgetedLLM(OpenAILLMClient(OpenAI(max_retries=8, timeout=120), args.model,
                                                     **sampling_for(base)), counter))
    planner = SchemaFreePlanner(recorder, vocabulary=EFFECT_VOCABULARIES[VOCABULARY])
    tasks_file = Path(__file__).resolve().parent / "tasks.py"
    append_jsonl(out, {"type": "run", "git": _git(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "split": args.split, "planner_model": args.model, "sampling": sampling_for(base),
                       "independent": args.independent, "vocabulary": EFFECT_VOCABULARIES[VOCABULARY],
                       "tasks_sha256": hashlib.sha256(tasks_file.read_bytes()).hexdigest()})
    try:
        for i, task in enumerate(chosen, 1):
            kw = {"user_request": task.user_request, "private_context": task.private_context, "catalog": task.catalog()}
            d = attempt(lambda: planner.delegate(**kw), recorder)
            append_jsonl(out, {"type": "planner", "task_id": task.task_id, "delegation": d["value"],
                               "failed_attempts": d["failed_attempts"], "response": d["response"]})
            for k in range(1 + args.independent):
                c = attempt(lambda: planner.contract(**kw), recorder)
                append_jsonl(out, {"type": "contract", "task_id": task.task_id, "vocabulary": VOCABULARY, "index": k,
                                   "role": "sent" if k == 0 else "independent",
                                   "contract": contract_to_json(c["value"]) if c["value"] is not None else None,
                                   "failed_attempts": c["failed_attempts"], "response": c["response"]})
            print(f"{i}/{len(chosen)} {task.task_id} calls={counter['calls']}", flush=True)
    except CallBudgetExceeded as e:
        append_jsonl(out, {"type": "stopped", "reason": str(e), "calls": counter["calls"]})
        print(f"STOPPED: {e}")
    print(f"calls used (retries included): {counter['calls']}")


if __name__ == "__main__":
    main()
