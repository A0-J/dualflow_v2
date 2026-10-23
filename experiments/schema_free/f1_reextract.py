"""F1 deviation fix: re-read every delegation's recipient with the catalog's recipient descriptions.

The F1 run (e00efc9) gave the extractor only the recipient ids, not their descriptions, although
the protocol says it sees d and the catalog's recipients. With neutral ids (contact_1, contact_2)
it could not tell which contact was which. This re-runs only the extraction, with ids and
descriptions and the same prompt, over the stored delegations; the Planner Agent's outputs are
unchanged. The first extraction is kept in each record as "extracted_ids_only".

    PYTHONPATH=src python experiments/schema_free/f1_reextract.py
    PYTHONPATH=src python experiments/schema_free/f1_run.py --analyze-only --out results/schema_free/f1/runs_reextracted.jsonl
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.io import append_jsonl, read_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402
from f1_run import EXTRACT, all_tasks, ask, parse_extract  # noqa: E402
from run_pilot import BudgetedLLM, _git  # noqa: E402

from common import sampling_for  # noqa: E402

SRC = ROOT / "results" / "schema_free" / "f1" / "runs.jsonl"
OUT = ROOT / "results" / "schema_free" / "f1" / "runs_reextracted.jsonl"
MODEL = "gpt-4.1-2025-04-14"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    tasks = {t.task_id: t for t in all_tasks()}
    counter = {"calls": 0, "cap": 2000}
    llm = BudgetedLLM(OpenAILLMClient(OpenAI(max_retries=8, timeout=120), MODEL, **sampling_for("gpt-4.1")), counter)
    for r in read_jsonl(SRC):
        if r["type"] == "run":
            append_jsonl(OUT, {**r, "reextracted": {"git": _git(), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                    "model": MODEL, "recipients": "ids with descriptions"}})
            continue
        if r["type"] == "output" and r["stage"] in ("pipeline", "connected"):
            t = tasks[r["task_id"]]
            d = (r["delegation"] if r["stage"] == "pipeline" else r["output"])["value"]
            d = d["delegation"] if d else None
            r["extracted_ids_only"] = r["extracted"]
            r["extracted"] = (ask(llm, EXTRACT, {"delegation_message": d, "recipients": dict(t.recipients)},
                                  lambda x: parse_extract(x, list(t.recipients))) if d else {"value": None})
        append_jsonl(OUT, r)
    print(f"calls used (retries included): {counter['calls']}")


if __name__ == "__main__":
    main()
