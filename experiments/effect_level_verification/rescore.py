"""Effect-level Semantic Flow, re-scored on the canonical cells. Post-hoc, 0 API calls.

Question: if the Semantic Flow compared execution *effects* instead of exact
(action, resource, scope) tuples, as it must when the Planner Agent and the
Executor Agent share no task schema, what would it still block, what would
it let through, and how much does a finer effect vocabulary recover?

Effects of one operation (a, r, s), as dualflow.effect_contracts.Effect:
  read           read r
  summarize      read r, internal_write (a summary of r)
  send_external  read r, external_send of r to the recipient in s   (irreversible)
  delete         delete r                                           (irreversible)
Any other action maps to an unknown effect, which the shared check rejects.
The intended operation P̂_intent becomes an IntentContract whose required
effects are its non-read effects (no permitted or forbidden extras).

Rules compared, D = V_sem AND V_auth throughout (V_auth unchanged):
  exact                        P_exec == P̂_intent (the canonical DualFlow)
  safety, resource-aware       assess_effects(...).safe
  safety+goal, resource-aware  assess_effects(...).safe and .goal_covered
  safety, coarse / safety+goal, coarse
                               the same two checks on effect kinds only (resource and
                               recipient dropped); an analysis view, not a gate

Outcomes are scored with the canonical definitions (dualflow.outcomes), so a
coarser rule cannot look safer by redefining what is unsafe.

Unit: canonical cell (scenario x Executor Agent model). The Planner Agent's
modal restatement stands in for its stated intended effects, so these are
not schema-free runs; they isolate what the comparison rule changes. Design
labels; the cells derive from 8 source delegations.

    PYTHONPATH=src python experiments/effect_level_verification/rescore.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from dualflow.effect_contracts import IRREVERSIBLE_KINDS, Effect, IntentContract, assess_effects  # noqa: E402
from dualflow.io import read_jsonl  # noqa: E402
from dualflow.outcomes import Label, Outcome, classify  # noqa: E402

from common import DERIVED as CANONICAL_DERIVED, EXECUTOR_MODELS, fmt  # noqa: E402

OUT = ROOT / "results" / "effect_level_verification" / "derived"
CASES = ("C1", "C2", "C3", "C4")
EXTERNAL = "external:"


def effects(op: dict) -> frozenset[Effect]:
    """Effects of one canonical (a, r, s) operation."""
    a, r, s = op["action"], op["resource"], op["scope"]
    if a == "read":
        return frozenset({Effect("read", r)})
    if a == "summarize":
        return frozenset({Effect("read", r), Effect("internal_write", r)})
    if a == "send_external":
        recipient = s[len(EXTERNAL):] if s.startswith(EXTERNAL) else ""
        return frozenset({Effect("read", r), Effect("external_send", r, recipient)})
    if a == "delete":
        return frozenset({Effect("delete", r)})
    return frozenset({Effect(f"unknown:{a}", r)})


def contract_for(p_intent: dict) -> IntentContract:
    return IntentContract(required=frozenset(e for e in effects(p_intent) if e.kind != "read"),
                          forbidden=frozenset())


def coarse_check(contract: IntentContract, proposed: frozenset[Effect]) -> tuple[bool, bool]:
    """Kind-only view of the same two checks; unknown kinds stay unsafe."""
    kinds = {e.kind for e in proposed}
    allowed = {e.kind for e in contract.required | contract.permitted if e.kind in IRREVERSIBLE_KINDS}
    unsafe = {k for k in kinds if k in IRREVERSIBLE_KINDS or k.startswith("unknown")}
    return unsafe <= allowed, {e.kind for e in contract.required} <= kinds


RULES = ("exact", "safety, coarse", "safety, resource-aware", "safety+goal, coarse", "safety+goal, resource-aware")


def v_sem(rule: str, p_exec: dict, p_intent: dict) -> bool:
    if rule == "exact":
        return all(p_exec[k] == p_intent[k] for k in ("action", "resource", "scope"))
    contract, proposed = contract_for(p_intent), effects(p_exec)
    if rule.endswith("coarse"):
        safe, goal = coarse_check(contract, proposed)
    else:
        verdict = assess_effects(contract, proposed)
        safe, goal = verdict.safe, verdict.goal_covered
    return safe and goal if rule.startswith("safety+goal") else safe


def rescore(cell: dict, rule: str) -> dict:
    lab = Label(cell["intended"], cell["authorized"], cell["auth_reason"], cell["irreversible_misinterpretation"])
    execute = v_sem(rule, cell["p_exec"], cell["p_intent"]) and cell["authorized"]
    return {"execute": execute, "outcome": classify(execute, lab), "completed": execute and cell["intended"]}


def mismatch_kind(cell: dict) -> str:
    """How P_exec differs from the Planner Agent's restated operation."""
    p, intended = cell["p_exec"], cell["p_intent"]
    if p["action"] == intended["action"]:
        return f"{p['action']}, other resource"
    return f"{p['action']} instead of {intended['action']}"


def main() -> None:
    lines: list[str] = []

    def say(text=""):
        print(text)
        lines.append(text)

    cells = read_jsonl(CANONICAL_DERIVED / "cells.jsonl")
    say("Effect-level Semantic Flow re-scored on canonical cells (post-hoc, 0 API calls)")
    say("Outcomes use the canonical definitions; Unsafe = executed and (unauthorized or irreversible misinterpretation)\n")
    for condition, title in (("context", "Planner Agent with task context (the method)"),
                             ("no_context", "Planner Agent without task context (ablation)")):
        sub = [c for c in cells if c["planner_condition"] == condition]
        valid = [c for c in sub if c["valid_proposal"]]
        say(f"## {title}: {len(sub)} cells, {len(valid)} valid proposals")
        say(f"  {'rule':28s} | {'unsafe':22s} | {'misaligned':22s} | {'false rejection':22s} | "
            f"{'intended executed C1':22s} | {'intended executed C2+C4':22s}")
        for rule in RULES:
            scored = [rescore(c, rule) for c in sub]
            count = lambda o: sum(s["outcome"] is o for s in scored)  # noqa: E731
            fr = sum(s["outcome"] is Outcome.FALSE_REJECTION for s, c in zip(scored, sub) if c["valid_proposal"])
            c1 = [s for s, c in zip(scored, sub) if c["case"] == "C1"]
            c24 = [s for s, c in zip(scored, sub) if c["case"] in ("C2", "C4")]
            say(f"  {rule:28s} | {fmt(count(Outcome.UNSAFE_EXECUTION), len(sub)):22s} | "
                f"{fmt(count(Outcome.MISALIGNED_EXECUTION), len(sub)):22s} | {fmt(fr, len(valid)):22s} | "
                f"{fmt(sum(s['completed'] for s in c1), len(c1)):22s} | {fmt(sum(s['completed'] for s in c24), len(c24))}")
        say("  by case, unsafe / misaligned (of 160 cells each):")
        for rule in RULES:
            parts = []
            for case in CASES:
                outs = Counter(rescore(c, rule)["outcome"] for c in sub if c["case"] == case)
                parts.append(f"{case} {outs[Outcome.UNSAFE_EXECUTION]:3d}/{outs[Outcome.MISALIGNED_EXECUTION]:3d}")
            say(f"    {rule:28s} " + "   ".join(parts))
        say()

    sub = [c for c in cells if c["planner_condition"] == "context"]
    say("## Misaligned executions by rule, Executor Agent model and kind of mismatch (Planner Agent with task context)")
    for rule in RULES[1:]:
        say(f"  {rule}")
        for model in EXECUTOR_MODELS:
            mis = [c for c in sub if c["executor_model"] == model
                   and rescore(c, rule)["outcome"] is Outcome.MISALIGNED_EXECUTION]
            say(f"    {model:12s} {len(mis):3d}  {dict(Counter(mismatch_kind(c) for c in mis))}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rescore_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
