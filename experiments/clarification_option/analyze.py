"""Analysis of the clarification-option experiment (PLAN.md section 5). 0 API calls.

Unit: one pair = (delegation text, Executor Agent model), 80 x 4 = 320 per
prompt variant. No authorization configuration is involved, so these are
not the canonical cells (scenario x model).

Per pair, over its k samples:
  decision  ask if more than half of the samples ask, else execute
  P_exec    the modal proposal among the execute samples
  category  asked | executed_intended | executed_mismatch
The same pair under forced choice is the modal of the first k canonical
Executor Agent samples (results/canonical/raw/samples.jsonl).

Ambiguity labels are the design labels (clear / ambiguous) of the canonical
delegations, not human judgements.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.flows import Proposal, modal  # noqa: E402
from dualflow.io import read_jsonl, write_jsonl  # noqa: E402
from dualflow.stats import mcnemar_exact  # noqa: E402

from common import EXECUTOR_MODELS, SAMPLES as CANONICAL_SAMPLES, delegation_texts, fmt  # noqa: E402
from scenarios import DOMAINS, IRREVERSIBLE  # noqa: E402

RAW = ROOT / "results" / "clarification_option" / "raw" / "samples.jsonl"
DERIVED = ROOT / "results" / "clarification_option" / "derived"
CATEGORIES = ("asked", "executed_intended", "executed_mismatch")


def score_pair(turns: list[dict], intended: Proposal, irreversible=IRREVERSIBLE) -> dict:
    """turns: the pair's samples, each with 'decision', 'proposal' and 'unclear'."""
    n = len(turns)
    asks = [t for t in turns if t["decision"] == "ask"]
    executes = [Proposal.from_dict(t["proposal"]) for t in turns if t["decision"] == "execute"]
    row = {"n": n, "asks": len(asks), "ask_rate": len(asks) / n,
           "asks_naming_action": sum("action" in (t["unclear"] or []) for t in asks)}
    if len(asks) * 2 > n:
        return {**row, "decision": "ask", "category": "asked", "p_exec": None,
                "irreversible_misinterpretation": False}
    p_exec = modal(executes)
    intended_ok = p_exec == intended.canonical()
    return {**row, "decision": "execute", "p_exec": p_exec.to_dict(),
            "category": "executed_intended" if intended_ok else "executed_mismatch",
            "irreversible_misinterpretation": (not intended_ok) and p_exec.action in irreversible}


def forced_choice(k: int) -> dict[tuple[str, str], Proposal]:
    bank = defaultdict(list)
    for r in read_jsonl(CANONICAL_SAMPLES):
        if r["role"] == "executor" and r["sample_index"] < k:
            bank[(r["text_id"], r["model"])].append((r["sample_index"], Proposal.from_dict(r["proposal"])))
    return {key: modal([p for _, p in sorted(v, key=lambda x: x[0])]) for key, v in bank.items()}


def main() -> None:
    lines: list[str] = []

    def say(text=""):
        print(text)
        lines.append(text)

    samples = defaultdict(list)
    for r in read_jsonl(RAW):
        samples[(r["variant"], r["text_id"], r["model"])].append(r)
    texts = {t["text_id"]: t for t in delegation_texts()}
    intended = {tid: Proposal.from_dict(DOMAINS[t["domain"]][t["clarity"]]["intended"]) for tid, t in texts.items()}
    rows = []
    for (variant, text_id, model), turns in sorted(samples.items()):
        turns = sorted(turns, key=lambda r: r["sample_index"])
        rows.append({"variant": variant, "text_id": text_id, "family": text_id.rsplit("_", 1)[0],
                     "clarity": texts[text_id]["clarity"], "executor_model": model,
                     **score_pair(turns, intended[text_id])})
    DERIVED.mkdir(parents=True, exist_ok=True)
    write_jsonl(DERIVED / "pairs.jsonl", rows)
    k = min(r["n"] for r in rows)
    forced = forced_choice(k)

    for variant in sorted({r["variant"] for r in rows}):
        sub = [r for r in rows if r["variant"] == variant]
        say(f"# Prompt variant: {variant}  (pairs={len(sub)}, samples per pair={k})")
        for clarity in ("ambiguous", "clear"):
            cs = [r for r in sub if r["clarity"] == clarity]
            say(f"\n## {clarity} delegations ({len(cs)} pairs)")
            for c in CATEGORIES:
                say(f"  {c:20s} {fmt(sum(r['category'] == c for r in cs), len(cs))}")
            say(f"  {'  of which irreversible':20s} {fmt(sum(r['irreversible_misinterpretation'] for r in cs), len(cs))}")
            say(f"  pairs with mixed samples (0 < ask rate < 1): {sum(0 < r['ask_rate'] < 1 for r in cs)}/{len(cs)}")
            for model in EXECUTOR_MODELS:
                ms = [r for r in cs if r["executor_model"] == model]
                say(f"    {model:12s} " + "  ".join(f"{c} {sum(r['category'] == c for r in ms)}/{len(ms)}" for c in CATEGORIES))

        amb = [r for r in sub if r["clarity"] == "ambiguous"]
        asks = sum(r["asks"] for r in amb)
        say(f"\n## Facet named in questions on ambiguous delegations: 'action' in 'unclear' "
            f"{fmt(sum(r['asks_naming_action'] for r in amb), asks)} (unit: ask samples)")

        say("\n## Paired with forced choice (same pair, modal of the first "
            f"{k} canonical samples): semantic mismatch executed")
        for clarity in ("ambiguous", "clear"):
            cs = {(r["text_id"], r["executor_model"]): r for r in sub if r["clarity"] == clarity}
            f_mis = {key: forced[key] != intended[key[0]] for key in cs}
            o_mis = {key: r["category"] == "executed_mismatch" for key, r in cs.items()}
            removed = [key for key in cs if f_mis[key] and not o_mis[key]]
            added = sum(o_mis[key] and not f_mis[key] for key in cs)
            by_ask = sum(cs[key]["category"] == "asked" for key in removed)
            say(f"  {clarity:9s} forced {fmt(sum(f_mis.values()), len(cs)):24s} with option {fmt(sum(o_mis.values()), len(cs)):24s} "
                f"discordant {len(removed)} vs {added} (exact McNemar p={mcnemar_exact(len(removed), added):.3g}); "
                f"of the {len(removed)} removed, {by_ask} by asking")

        say("\n## Per source family (10 texts x 4 models): asked | executed_intended | executed_mismatch")
        for family in sorted({r["family"] for r in sub}):
            fs = [r for r in sub if r["family"] == family]
            say(f"  {family:22s} " + " | ".join(f"{sum(r['category'] == c for r in fs):2d}" for c in CATEGORIES))
        say()

    (DERIVED / "analysis_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
