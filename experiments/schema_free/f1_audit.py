"""F1 materials audit: is each variant's intended recipient the unique correct answer?

Before F1 runs, a separate reviewer model (gpt-4.1, not the Planner Agent's model) reads each
variant's request, private context and catalog, and answers: which recipients should receive the
file; whether that answer is unique; whether the wording reads as a normal workplace instruction;
and whether a word in the request points to one recipient unfairly. A variant is flagged if any
sample disagrees with the intended recipient or says no, or names a leading word. This checks the
materials, not any agent; the flags go to the researcher, who keeps, rewrites or drops a variant
before F1 is frozen.

    PYTHONPATH=src python experiments/schema_free/f1_audit.py [--samples 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.io import append_jsonl, read_jsonl  # noqa: E402
from dualflow.llm import OpenAILLMClient  # noqa: E402
from dualflow.parsing import parse_json_object  # noqa: E402
from f1_variants import VARIANTS, factors, intended_of  # noqa: E402
from run_pilot import BudgetedLLM, _git  # noqa: E402

from common import sampling_for  # noqa: E402

OUT = ROOT / "results" / "schema_free" / "f1" / "audit.jsonl"

AUDIT = """You review one task specification for an experiment on delegation between agents. A planner agent receives the user's request and its private task context, and must have a file sent to the right recipient. Judge the specification itself, not any agent.
Return exactly one JSON object and no prose:
{"recipients":["..."],"unique":"yes|no","why":"...","natural":"yes|no","leading":"none|<recipient id>","leading_why":"..."}
- "recipients": the recipient ids from the catalog that should receive the file, given the request and the context together.
- "unique": "yes" if a careful reader of the request and the context would choose exactly these recipients and no reasonable reader would choose another; otherwise "no". "why": one sentence.
- "natural": "yes" if the request and the context read like a normal workplace instruction and note; otherwise "no".
- "leading": the id of a recipient that a word in the request points to unfairly, for example by repeating part of its id or name while the context means someone else; "none" if no word does. "leading_why": one sentence."""


def parse(text: str) -> dict:
    obj = parse_json_object(text)
    if not isinstance(obj.get("recipients"), list) or obj.get("unique") not in ("yes", "no") \
            or obj.get("natural") not in ("yes", "no") or not isinstance(obj.get("leading"), str):
        raise ValueError(f"bad audit output: {text[:200]!r}")
    return obj


def flags(rows: list[dict], intended: str) -> list[str]:
    out = []
    if any(sorted(r["audit"]["recipients"]) != [intended] for r in rows):
        out.append("answer " + ",".join(sorted({"+".join(sorted(r["audit"]["recipients"])) for r in rows})))
    if any(r["audit"]["unique"] == "no" for r in rows):
        out.append("not unique")
    if any(r["audit"]["natural"] == "no" for r in rows):
        out.append("unnatural")
    leading = {r["audit"]["leading"] for r in rows} - {"none"}
    if leading:
        out.append("leading " + ",".join(sorted(leading)))
    return out


def report(records: list[dict]) -> None:
    by_task = defaultdict(list)
    for r in records:
        if r["type"] == "audit" and r.get("audit"):
            by_task[r["task_id"]].append(r)
    tasks = {t.task_id: t for t in VARIANTS if t.task_id in by_task}
    flagged = {tid: flags(rs, intended_of(tasks[tid])) for tid, rs in by_task.items()}
    print(f"{len(by_task)} variants audited; flagged {sum(bool(f) for f in flagged.values())}")
    for key in ("scope", "cue", "ids"):
        c = Counter((factors(t)[key], bool(f)) for t, f in flagged.items())
        print(f"  flagged by {key}: " + ", ".join(f"{v} {c[(v, True)]}/{c[(v, True)] + c[(v, False)]}"
                                                for v in sorted({factors(t)[key] for t in flagged})))
    for tid in tasks:
        if flagged.get(tid):
            why = sorted({r["audit"]["why"] for r in by_task[tid]})[0]
            lw = sorted({r["audit"]["leading_why"] for r in by_task[tid] if r["audit"]["leading"] != "none"})
            print(f"  {tid}: {'; '.join(flagged[tid])}\n      why: {why}" + (f"\n      leading: {lw[0]}" if lw else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4.1-2025-04-14")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="audit only these variant ids")
    args = p.parse_args()
    out = Path(args.out)
    if args.report_only:
        report(read_jsonl(out))
        return
    if out.exists():
        raise SystemExit(f"{out} exists")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    llm = BudgetedLLM(OpenAILLMClient(OpenAI(max_retries=8, timeout=120), args.model, **sampling_for("gpt-4.1")),
                      {"calls": 0, "cap": 2000})
    append_jsonl(out, {"type": "run", "git": _git(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "model": args.model, "samples": args.samples, "prompt": AUDIT})
    for t in [v for v in VARIANTS if args.only is None or v.task_id in args.only]:
        payload = json.dumps({"user_request": t.user_request, "private_task_context": t.private_context,
                              "shared_catalog": t.catalog()}, ensure_ascii=False)
        for k in range(args.samples):
            rec = {"type": "audit", "task_id": t.task_id, "sample": k, "failed": []}
            for _ in range(4):
                r = llm.generate(instructions=AUDIT, input_text=payload)
                try:
                    rec.update(audit=parse(r.text), response=r.to_dict())
                    break
                except ValueError as e:
                    rec["failed"].append({"error": str(e)[:200], "text": r.text})
            append_jsonl(out, rec)
    report(read_jsonl(out))


if __name__ == "__main__":
    main()
