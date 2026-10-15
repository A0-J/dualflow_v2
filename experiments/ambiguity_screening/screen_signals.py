"""Post-hoc screening of ambiguity signals on the canonical samples. 0 API calls.

Question: can signals computed from data that already exists tell the
ambiguous canonical delegations from the clear ones, before any detection
method is designed?

Unit: delegation text (80; 40 clear, 40 ambiguous). Labels are the design
labels of the canonical scenarios, not human judgements. The texts derive
from 8 source delegations, so results are close to all-or-nothing per family
and the effective sample is 4 ambiguous against 4 clear families. These 80
texts are the development set for detection signals; nothing here is
evidence about new delegations.

Signals per delegation text:
  self_entropy_mean     mean over Executor Agent models of the entropy (bits) of
                        the 20 sampled (a, r, s); the Consistency-based Gate's signal
  cross_model_distinct  number of distinct modal proposals across the 4 models
  reader_disagreement   share of models whose modal proposal differs from the
                        Planner Agent's modal restatement without task context
  context_shift_tv      total variation distance between the Planner Agent's
                        restatements with and without task context (needs C_task)
  context_shift_modal   1 if the Planner Agent's modal restatement changes with C_task

AUROC is threshold-free. The flag rules printed next to it are fixed,
illustrative cut points, not tuned values.

    PYTHONPATH=src python experiments/ambiguity_screening/screen_signals.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.flows import Proposal, modal  # noqa: E402
from dualflow.io import read_jsonl  # noqa: E402
from dualflow.stats import entropy  # noqa: E402

from common import EXECUTOR_MODELS, SAMPLES  # noqa: E402
from scenarios import DOMAINS  # noqa: E402

OUT = ROOT / "results" / "ambiguity_screening" / "derived"
PLANNER_NO_CONTEXT = "planner_no_context"


def total_variation(a: list, b: list) -> float:
    ca, cb = Counter(a), Counter(b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def auroc(positives: list[float], negatives: list[float]) -> float:
    """Probability that a random positive scores above a random negative; ties count half."""
    return sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))


def load_banks():
    """Samples in sample_index order, so modal ties go to the earliest sample, as in analyze.py."""
    executor, planner = defaultdict(list), defaultdict(list)
    for r in sorted(read_jsonl(SAMPLES), key=lambda r: r["sample_index"]):
        p = Proposal.from_dict(r["proposal"])
        if r["role"] == "executor":
            executor[(r["text_id"], r["model"])].append(p)
        else:
            planner[(r["text_id"], r["planner_condition"])].append(p)
    return executor, planner


def text_rows(executor, planner) -> list[dict]:
    rows = []
    for text_id in sorted({t for t, _ in executor}):
        domain, clarity = text_id.split("_")[:2]
        intended = Proposal.from_dict(DOMAINS[domain][clarity]["intended"])
        p_exec = {m: modal(executor[(text_id, m)]) for m in EXECUTOR_MODELS}
        no_ctx, ctx = planner[(text_id, "no_context")], planner[(text_id, "context")]
        rows.append({
            "text_id": text_id, "family": text_id.rsplit("_", 1)[0], "ambiguous": clarity == "ambiguous",
            "models_with_semantic_mismatch": sum(p != intended for p in p_exec.values()),
            "self_entropy_mean": sum(entropy([(p.action, p.resource, p.scope) for p in executor[(text_id, m)]])
                                     for m in EXECUTOR_MODELS) / len(EXECUTOR_MODELS),
            "cross_model_distinct": len(set(p_exec.values())),
            "reader_disagreement": sum(p != modal(no_ctx) for p in p_exec.values()) / len(EXECUTOR_MODELS),
            "context_shift_tv": total_variation(ctx, no_ctx),
            "context_shift_modal": float(modal(ctx) != modal(no_ctx)),
            **{f"p_exec__{m}": p_exec[m] for m in EXECUTOR_MODELS},
            "intended": intended, PLANNER_NO_CONTEXT: modal(no_ctx),
        })
    return rows


def main() -> None:
    lines: list[str] = []

    def say(text=""):
        print(text)
        lines.append(text)

    rows = text_rows(*load_banks())
    amb = [r for r in rows if r["ambiguous"]]
    clr = [r for r in rows if not r["ambiguous"]]
    say(f"delegation texts={len(rows)}; with a semantic mismatch by >=1 Executor Agent model: "
        f"ambiguous {sum(r['models_with_semantic_mismatch'] > 0 for r in amb)}/{len(amb)}, "
        f"clear {sum(r['models_with_semantic_mismatch'] > 0 for r in clr)}/{len(clr)}")

    rules = (("self_entropy_mean", "mean entropy > 0.8 bits", lambda r: r["self_entropy_mean"] > 0.8),
             ("cross_model_distinct", ">= 2 distinct modal proposals", lambda r: r["cross_model_distinct"] >= 2),
             ("reader_disagreement", "any model differs from context-free reading", lambda r: r["reader_disagreement"] > 0),
             ("context_shift_tv", "TV distance >= 0.5", lambda r: r["context_shift_tv"] >= 0.5),
             ("context_shift_modal", "modal restatement changes with C_task", lambda r: r["context_shift_modal"] > 0))
    say(f"\n{'signal':22s} {'AUROC':>5s}  flagged ambiguous | flagged clear  (illustrative rule)")
    for name, desc, rule in rules:
        score = auroc([r[name] for r in amb], [r[name] for r in clr])
        say(f"{name:22s} {score:5.2f}  {sum(map(rule, amb)):>8d}/{len(amb)}     | {sum(map(rule, clr)):>4d}/{len(clr)}"
            f"       ({desc})")

    say("\nper source family (10 texts): cross_model_distinct>=2 | context_shift_modal | texts with a semantic mismatch")
    for family in sorted({r["family"] for r in rows}):
        fs = [r for r in rows if r["family"] == family]
        say(f"  {family:22s} {sum(r['cross_model_distinct'] >= 2 for r in fs):2d} | "
            f"{sum(r['context_shift_modal'] > 0 for r in fs):2d} | {sum(r['models_with_semantic_mismatch'] > 0 for r in fs):2d}")

    say("\ncross-model disagreement without one model (ambiguous / clear texts flagged)")
    for left_out in EXECUTOR_MODELS:
        kept = [m for m in EXECUTOR_MODELS if m != left_out]
        flag = lambda r: len({r[f'p_exec__{m}'] for m in kept}) >= 2  # noqa: E731
        say(f"  without {left_out:12s} {sum(map(flag, amb)):2d}/{len(amb)} | {sum(map(flag, clr)):2d}/{len(clr)}")

    say("\none Executor Agent plus one second reader: its semantic mismatches on ambiguous texts that the "
        "second reader disagrees with")
    for m in EXECUTOR_MODELS:
        mism = [r for r in amb if r[f"p_exec__{m}"] != r["intended"]]
        readers = [o for o in EXECUTOR_MODELS if o != m] + [PLANNER_NO_CONTEXT]
        caught = ", ".join(f"{o}: {sum(r[f'p_exec__{m}'] != r[o if o == PLANNER_NO_CONTEXT else f'p_exec__{o}'] for r in mism)}"
                           for o in readers)
        say(f"  {m:12s} mismatches {len(mism):2d} -> {caught}")

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["text_id", "family", "ambiguous", "models_with_semantic_mismatch"] + [n for n, _, _ in rules]
    with open(OUT / "text_signals.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (OUT / "screening_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
