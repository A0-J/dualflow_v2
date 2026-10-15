"""Check, before collection, which model serves each requested Executor Agent ID (PLAN.md section 2).

  (default)  free: is each pinned ID in the account's model list?
  --call     one minimal call per pinned ID and per canonical alias (7 calls), with the
             canonical sampling settings; records the served model

The result is appended to results/clarification_option/raw/model_check.jsonl,
so the check is part of the record.

    PYTHONPATH=src python experiments/clarification_option/check_models.py [--call]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# This experiment's collect.py first: it puts experiments/canonical (which has its
# own collect.py) on the path for common and scenarios.
from collect import MODEL_IDS, RAW  # noqa: E402
from common import EXECUTOR_MODELS, sampling_for  # noqa: E402
from dualflow.io import append_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--call", action="store_true", help="make one minimal call per ID (costs 7 calls)")
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    sdk = OpenAI(max_retries=3, timeout=120)
    available = {m.id for m in sdk.models.list()}
    record = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "models": []}
    for label in EXECUTOR_MODELS:
        for requested in dict.fromkeys((MODEL_IDS[label], label)):  # pinned ID, then the canonical alias
            row = {"executor_model": label, "requested": requested, "pinned": requested == MODEL_IDS[label],
                   "in_model_list": requested in available}
            if args.call:
                try:
                    response = OpenAILLMClient(sdk, requested, **sampling_for(label)).generate(
                        instructions="Reply with the single word OK.", input_text="OK")
                    row["served_model"] = response.served_model
                except Exception as e:  # recorded, not raised: the point is to see what fails
                    row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            record["models"].append(row)
            print(f"{label:12s} {requested:24s} in list: {row['in_model_list']!s:5s} "
                  f"served: {row.get('served_model', row.get('error', '-'))}")
    append_jsonl(RAW / "model_check.jsonl", record)
    print(f"appended to {RAW / 'model_check.jsonl'}")


if __name__ == "__main__":
    main()
