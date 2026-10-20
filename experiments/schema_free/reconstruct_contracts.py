"""Contract reconstruction diagnostic: effect vocabulary v2 vs v3, on the same tasks, in one run.

Only the Planner Agent's contract step runs; no Executor Agent, no gate. Per task and vocabulary it
draws what the pilot draws, one "sent" and N "independent" contracts with the same prompt, so the
per-sample rates compare with pilot v2. The two vocabularies alternate call by call, so both are
drawn under the same conditions; only the vocabulary text differs. Every call counts against a hard
cap, retries included. Classification: classify_contracts.py.

    PYTHONPATH=src python experiments/schema_free/reconstruct_contracts.py --dry-run
    PYTHONPATH=src python experiments/schema_free/reconstruct_contracts.py
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

from dualflow.io import append_jsonl, read_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402
from dualflow.schema_free import (  # noqa: E402
    EFFECT_VOCABULARIES, MAX_ATTEMPTS, SchemaFreePlanner, contract_to_json, with_retries,
)
from run_pilot import CallBudgetExceeded, BudgetedLLM, _git  # noqa: E402
from tasks import tasks  # noqa: E402

from classify_contracts import report, samples  # noqa: E402
from common import sampling_for  # noqa: E402

OUT = ROOT / "results" / "schema_free" / "reconstruction_v3"
N_INDEPENDENT = 5
CALL_CAP = 200


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--planner-model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--independent", type=int, default=N_INDEPENDENT)
    p.add_argument("--cap", type=int, default=CALL_CAP)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    path = out / "contracts.jsonl"
    chosen = tasks(args.split)
    vocabularies = sorted(EFFECT_VOCABULARIES)
    bound = len(chosen) * (1 + args.independent) * len(vocabularies)
    print(f"split={args.split} tasks={len(chosen)} vocabularies={vocabularies} samples per task and vocabulary="
          f"{1 + args.independent}\nLLM calls: {bound} without retries; hard cap {args.cap} including retries "
          f"(up to {MAX_ATTEMPTS} attempts per call)")
    if args.dry_run:
        return
    if bound > args.cap:
        raise SystemExit(f"{bound} calls exceed the cap {args.cap}")
    if path.exists():
        raise SystemExit(f"{path} exists; use a new --out for a new run")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    counter = {"calls": 0, "cap": args.cap}
    base = "gpt-4o-mini" if args.planner_model.startswith("gpt-4o-mini") else args.planner_model
    llm = BudgetedLLM(OpenAILLMClient(OpenAI(max_retries=8, timeout=120), args.planner_model, **sampling_for(base)),
                      counter)
    planners = {v: SchemaFreePlanner(llm, vocabulary=EFFECT_VOCABULARIES[v]) for v in vocabularies}
    code = hashlib.sha256((ROOT / "src" / "dualflow" / "schema_free.py").read_bytes()).hexdigest()
    append_jsonl(path, {"type": "run", "git": _git(), "schema_free_sha256": code,
                        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "split": args.split,
                        "planner_model": args.planner_model, "sampling": sampling_for(base),
                        "independent": args.independent, "cap": args.cap, "vocabularies": EFFECT_VOCABULARIES})
    try:
        for i, task in enumerate(chosen, 1):
            for k in range(1 + args.independent):
                for v in (vocabularies if k % 2 == 0 else vocabularies[::-1]):
                    rec = {"type": "contract", "task_id": task.task_id, "vocabulary": v, "index": k,
                           "role": "sent" if k == 0 else "independent"}
                    try:
                        a = with_retries(lambda: planners[v].contract(
                            user_request=task.user_request, private_context=task.private_context,
                            catalog=task.catalog()))
                        rec.update(contract=contract_to_json(a.value), failed_attempts=list(a.failed_attempts),
                                   response=a.response.to_dict())
                    except ValueError as e:
                        rec.update(contract=None, error=str(e)[:300])
                    append_jsonl(path, rec)
            print(f"{i}/{len(chosen)} {task.task_id} calls={counter['calls']}", flush=True)
    except CallBudgetExceeded as e:
        append_jsonl(path, {"type": "stopped", "reason": str(e), "calls": counter["calls"]})
        print(f"STOPPED: {e}")
    print(f"calls used (retries included): {counter['calls']}")
    from tasks import DEV_TASKS, MAIN_TASKS
    by_id = {t.task_id: t for t in DEV_TASKS + MAIN_TASKS}
    groups = samples(read_jsonl(path))
    for v in vocabularies:
        report(f"{out.name} / {v}", groups[v], by_id)


if __name__ == "__main__":
    main()
