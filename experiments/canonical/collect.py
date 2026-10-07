"""Collect Executor proposals and Planner restatements for every delegation
text. One JSONL line per sample, appended as it completes; re-running the
same command resumes (collected unit keys are skipped). Unparseable output
is retried up to 3 more times and the parse errors are recorded.
"""

from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from dualflow.io import append_jsonl, read_jsonl
from dualflow.llm import OpenAILLMClient
from dualflow.roles import ExecutorAgent, PlannerAgent

from common import (
    EXECUTOR_MODELS, N_SAMPLES, PLANNER_CONDITIONS, PLANNER_MODEL, SAMPLES,
    delegation_texts, sampling_for,
)
from scenarios import DOMAINS

MAX_ATTEMPTS = 4


def unit_key(role, text_id, model, condition, i):
    return f"{role}|{text_id}|{model}|{condition or '-'}|{i}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output", default=str(SAMPLES))
    p.add_argument("--limit-texts", type=int, default=None, help="smoke tests only")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    texts = delegation_texts()[: args.limit_texts] if args.limit_texts else delegation_texts()
    out = Path(args.output)
    done = {r["unit_key"] for r in read_jsonl(out)} if out.exists() else set()
    units = []
    for t in texts:
        units += [("executor", t, m, None, i) for m in EXECUTOR_MODELS for i in range(args.samples)]
        units += [("planner", t, PLANNER_MODEL, c, i) for c in PLANNER_CONDITIONS for i in range(args.samples)]
    units = [u for u in units if unit_key(u[0], u[1]["text_id"], u[2], u[3], u[4]) not in done]
    print(f"texts={len(texts)} already_done={len(done)} to_run={len(units)}")
    if args.dry_run or not units:
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    sdk = OpenAI(max_retries=8, timeout=120)
    clients = {m: OpenAILLMClient(sdk, m, **sampling_for(m)) for m in set(EXECUTOR_MODELS) | {PLANNER_MODEL}}
    lock = threading.Lock()
    counters = {"ok": 0, "failed": 0, "in": 0, "out": 0}

    def run(unit):
        role, t, model, condition, i = unit
        d = DOMAINS[t["domain"]]
        errors = []
        for _ in range(MAX_ATTEMPTS):
            try:
                if role == "executor":
                    call = ExecutorAgent(clients[model]).propose(delegation=t["delegation"], context=d["context"])
                else:
                    private = d[t["clarity"]]["private_context"] if condition == "context" else ""
                    call = PlannerAgent(clients[model]).restate(
                        delegation=t["delegation"], private_context=private, context=d["context"])
                return unit, call, errors
            except ValueError as e:
                errors.append(str(e)[:300])
        return unit, None, errors

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, u) for u in units]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                (role, t, model, condition, i), call, errors = fut.result()
            except Exception as e:  # API error after SDK retries: left for a resume run
                with lock:
                    counters["failed"] += 1
                print(f"API error (unit left for resume): {type(e).__name__}: {str(e)[:200]}")
                continue
            record = {
                "unit_key": unit_key(role, t["text_id"], model, condition, i),
                "role": role, "text_id": t["text_id"], "model": model,
                "planner_condition": condition, "sample_index": i,
                "proposal": call.proposal.to_dict() if call else None,
                "response": call.response.to_dict() if call else None,
                "parse_errors": errors,
            }
            with lock:
                if call:
                    append_jsonl(out, record)
                    counters["ok"] += 1
                    counters["in"] += call.response.input_tokens or 0
                    counters["out"] += call.response.output_tokens or 0
                else:
                    counters["failed"] += 1
                    append_jsonl(out.with_name(out.stem + "_parse_failures.jsonl"), record)
            if n % 500 == 0 or n == len(futures):
                print(f"{n}/{len(futures)} {counters}", flush=True)
    print(f"done: {counters}")


if __name__ == "__main__":
    main()
