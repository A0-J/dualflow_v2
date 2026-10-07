"""E4 step 2: collect Delegate and Principal sample banks.

One JSONL line per sample, appended as it completes, so an interrupted run
resumes by re-running the same command (already-collected unit keys are
skipped). Data that already exists is reused, not re-collected
(PREREGISTRATION.md §4): for the 6 original scenarios, the gpt-4o-mini
Delegate bank and the L2 Principal bank come from
results/delegate_repeated_sampling/raw/episodes.jsonl.

A sample whose output does not parse is retried up to 3 more times; the
parse errors are recorded on the sample that finally succeeds.
"""

from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from dualflow.agents import DelegateAgent, PrincipalAgent
from dualflow.io import append_jsonl, read_jsonl

from common import (
    DELEGATE_MODELS, LEVELS, N_SAMPLES, PRINCIPAL_MODEL, SAMPLES, TEMPERATURE,
    TOP_P, SamplingClient, load_scenarios, private_goal,
)

MAX_ATTEMPTS = 4


def unit_key(role: str, scenario_id: str, model: str, level: str | None, i: int) -> str:
    return f"{role}|{scenario_id}|{model}|{level or '-'}|{i}"


def plan_units(scenarios, *, roles, models, levels, sources, samples):
    units = []
    for s in scenarios:
        if s["source"] not in sources:
            continue
        if "delegate" in roles:
            for m in models:
                if s["source"] == "original" and m == "gpt-4o-mini":
                    continue  # reused
                units += [("delegate", s, m, None, i) for i in range(samples)]
        if "principal" in roles:
            for lvl in levels:
                if s["source"] == "original" and lvl == "L2":
                    continue  # reused
                units += [("principal", s, PRINCIPAL_MODEL, lvl, i) for i in range(samples)]
    return units


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roles", nargs="+", default=["delegate", "principal"])
    p.add_argument("--models", nargs="+", default=list(DELEGATE_MODELS))
    p.add_argument("--levels", nargs="+", default=list(LEVELS))
    p.add_argument("--sources", nargs="+", default=["original", "generated"])
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--limit-scenarios", type=int, default=None, help="smoke tests only")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", default=str(SAMPLES))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    scenarios = load_scenarios()
    if args.limit_scenarios:
        scenarios = scenarios[: args.limit_scenarios]
    out = Path(args.output)
    done = {r["unit_key"] for r in read_jsonl(out)} if out.exists() else set()
    units = [u for u in plan_units(scenarios, roles=args.roles, models=args.models,
                                   levels=args.levels, sources=args.sources, samples=args.samples)
             if unit_key(u[0], u[1]["scenario_id"], u[2], u[3], u[4]) not in done]
    by_model: dict[str, int] = {}
    for u in units:
        by_model[f"{u[0]}:{u[2]}"] = by_model.get(f"{u[0]}:{u[2]}", 0) + 1
    print(f"scenarios={len(scenarios)} already_done={len(done)} to_run={len(units)} {by_model}")
    if args.dry_run or not units:
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    sdk = OpenAI(max_retries=8, timeout=120)
    clients = {m: SamplingClient(sdk, m, temperature=TEMPERATURE, top_p=TOP_P)
               for m in set(args.models) | {PRINCIPAL_MODEL}}
    lock = threading.Lock()
    counters = {"ok": 0, "failed": 0, "in": 0, "out": 0}

    def run(unit):
        role, s, model, level, i = unit
        errors = []
        for _ in range(MAX_ATTEMPTS):
            try:
                if role == "delegate":
                    call = DelegateAgent(clients[model]).propose(
                        delegation=s["delegation"], context=s["context"])
                else:
                    call = PrincipalAgent(clients[model]).restate_intent(
                        delegation=s["delegation"], private_goal=private_goal(s, level),
                        context=s["context"])
                return unit, call, errors
            except ValueError as e:  # unparseable model output
                errors.append(str(e)[:300])
        return unit, None, errors

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, u) for u in units]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                (role, s, model, level, i), call, errors = fut.result()
            except Exception as e:  # API error after SDK retries: leave unit for a resume run
                with lock:
                    counters["failed"] += 1
                print(f"API error (unit left for resume): {type(e).__name__}: {str(e)[:200]}")
                continue
            record = {
                "unit_key": unit_key(role, s["scenario_id"], model, level, i),
                "role": role, "scenario_id": s["scenario_id"], "source": s["source"],
                "intent": s["intent"], "model": model, "level": level, "sample_index": i,
                "interpretation": call.interpretation.to_dict() if call else None,
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
            if n % 250 == 0 or n == len(futures):
                print(f"{n}/{len(futures)} {counters}", flush=True)

    print(f"done: {counters}")


if __name__ == "__main__":
    main()
