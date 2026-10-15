"""Collect Executor Agent turns with the option to ask (PLAN.md).

The 80 canonical delegation texts and runtime contexts are reused unchanged.
Each sample is one call to `ExecutorAgent.propose_or_ask`: the Executor Agent
either proposes one operation or asks the Planner Agent one question. No
Planner Agent call is made; nothing is answered or executed.

One JSONL line per sample, appended as it completes; re-running the same
command resumes. Unparseable output is retried up to 3 more times and the
parse errors are recorded.

    PYTHONPATH=src python experiments/clarification_option/collect.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.io import append_jsonl, read_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402
from dualflow.roles import ExecutorAgent  # noqa: E402

from common import EXECUTOR_MODELS, delegation_texts, sampling_for  # noqa: E402
from scenarios import DOMAINS  # noqa: E402

RAW = ROOT / "results" / "clarification_option" / "raw"
SAMPLES = RAW / "samples.jsonl"
N_SAMPLES = 5
MAX_ATTEMPTS = 4

# Model ID requested per Executor Agent model: the served_model that the API
# returned for it in the canonical collection (results/canonical/raw/samples.jsonl).
# The dated IDs request that snapshot directly; gpt-6.1-sol was served without a
# date, so its version cannot be pinned (PLAN.md section 2).
MODEL_IDS = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5.5": "gpt-5.5-2026-04-23",
    "gpt-6.1-sol": "gpt-6.1-sol",
}


def unit_key(text_id, model, variant, i):
    return f"executor_option|{text_id}|{model}|{variant}|{i}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--variants", nargs="+", choices=("neutral", "explicit"), default=["neutral"])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--output", default=str(SAMPLES))
    p.add_argument("--limit-texts", type=int, default=None, help="smoke tests only")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    texts = delegation_texts()[: args.limit_texts] if args.limit_texts else delegation_texts()
    out = Path(args.output)
    done = {r["unit_key"] for r in read_jsonl(out)} if out.exists() else set()
    units = [(t, m, v, i) for t in texts for m in EXECUTOR_MODELS for v in args.variants for i in range(args.samples)]
    units = [u for u in units if unit_key(u[0]["text_id"], u[1], u[2], u[3]) not in done]
    print(f"texts={len(texts)} models={len(EXECUTOR_MODELS)} variants={args.variants} samples={args.samples} "
          f"already_done={len(done)} to_run={len(units)} calls")
    if args.dry_run or not units:
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    sdk = OpenAI(max_retries=8, timeout=120)
    clients = {m: OpenAILLMClient(sdk, MODEL_IDS[m], **sampling_for(m)) for m in EXECUTOR_MODELS}
    lock = threading.Lock()
    counters = {"ok": 0, "failed": 0, "asked": 0}

    def run(unit):
        t, model, variant, i = unit
        errors = []
        for _ in range(MAX_ATTEMPTS):
            try:
                turn = ExecutorAgent(clients[model]).propose_or_ask(
                    delegation=t["delegation"], context=DOMAINS[t["domain"]]["context"],
                    explicit=variant == "explicit")
                return unit, turn, errors
            except ValueError as e:
                errors.append(str(e)[:300])
        return unit, None, errors

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, u) for u in units]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                (t, model, variant, i), turn, errors = fut.result()
            except Exception as e:  # API error after SDK retries: left for a resume run
                with lock:
                    counters["failed"] += 1
                print(f"API error (unit left for resume): {type(e).__name__}: {str(e)[:200]}")
                continue
            record = {
                "unit_key": unit_key(t["text_id"], model, variant, i),
                "text_id": t["text_id"], "model": model, "variant": variant, "sample_index": i,
                "decision": None if turn is None else ("ask" if turn.asked else "execute"),
                "proposal": turn.proposal.to_dict() if turn and turn.proposal else None,
                "question": turn.question if turn else None,
                "unclear": list(turn.unclear) if turn else None,
                "response": turn.response.to_dict() if turn else None,
                "parse_errors": errors,
            }
            with lock:
                if turn:
                    append_jsonl(out, record)
                    counters["ok"] += 1
                    counters["asked"] += turn.asked
                else:
                    counters["failed"] += 1
                    append_jsonl(out.with_name(out.stem + "_parse_failures.jsonl"), record)
            if n % 200 == 0 or n == len(futures):
                print(f"{n}/{len(futures)} {counters}", flush=True)
    print(f"done: {counters}")


if __name__ == "__main__":
    main()
