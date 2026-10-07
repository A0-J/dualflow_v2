"""E4 exploratory analyses -- NOT pre-registered, written after the
pre-registered analysis (analyze_e4.py) had been run. Every number this
prints must be reported as exploratory. 0 API calls.

  X1  what each Delegate model actually chooses on summarize-intent cells
  X2  why semantic grounding at L0 stayed safe even though the L0
      Principal shares the Delegate's misread: action-level co-signing vs
      incidental mismatches on other facets
  X3  component ablation: does the Delegate entropy gate add safety, or
      only false rejects? (semantic AND authority, grounded AND authority)
  X4  the same ablation on the frozen 13 scenarios
  X5  effect-facet grounding: compare only the facets that determine the
      side effect (action, scope) and leave conditions to authority --
      motivated by X3's finding that every cross-model false reject is a
      condition-facet mismatch
"""

from __future__ import annotations

import csv
from collections import Counter

from dualflow.io import read_jsonl

from common import DELEGATE_MODELS, DERIVED, LEVELS, fmt_rate

PRIMARY_MODEL = "gpt-4o-mini"


def section(title: str) -> None:
    print(f"\n## {title}")


def main() -> None:
    cells = read_jsonl(DERIVED / "e4_cells.jsonl")
    l2 = [r for r in cells if r["level"] == "L2"]

    section("X1 -- Delegate modal action on summarize-intent scenarios (generated + original)")
    for model in DELEGATE_MODELS:
        sub = [r for r in l2 if r["delegate_model"] == model and r["intent"] == "summarize"]
        actions = Counter(r["x_action"] for r in sub)
        unstable = sum(r["delegate_action_entropy"] > 0.8 for r in sub)
        mean_share = sum(r["delegate_action_share"] for r in sub) / len(sub)
        print(f"{model:12s} modal actions {dict(actions)}  action-unstable cells={unstable}/{len(sub)}  "
              f"mean modal-action share={mean_share:.2f}")

    section(f"X2 -- {PRIMARY_MODEL} Delegate, confident-misread cells: does the Principal co-sign the action?")
    for level in LEVELS:
        sub = [r for r in cells if r["delegate_model"] == PRIMARY_MODEL and r["level"] == level
               and r["confident_action_misread"]]
        cosign = [r for r in sub if r["y_action"] == r["x_action"]]
        exact = [r for r in cosign if r["semantic_grounding_only__execute"]]
        grounded = [r for r in cosign if r["grounded_only__execute"]]
        reasons = Counter()
        for r in cosign:
            if not r["semantic_grounding_only__execute"]:
                diff = [f for f in ("scope", "condition") if r[f"x_{f}"] != r[f"y_{f}"]]
                reasons["+".join(diff) or "none"] += 1
        print(f"{level}: misread cells={len(sub)}  Principal co-signs action={len(cosign)}  "
              f"-> exact-match executes {len(exact)}, per-facet grounded executes {len(grounded)}  "
              f"| exact-match rejected co-signed cells because of: {dict(reasons)}")

    section("X3 -- component ablation, generated scenarios (no Delegate entropy gate)")
    print("sem+auth = semantic_grounding_only AND authority_only; grd+auth = grounded_only AND authority_only")
    for model in DELEGATE_MODELS:
        for level in LEVELS:
            sub = [r for r in cells if r["delegate_model"] == model and r["level"] == level and r["source"] == "generated"]
            n_so = sum(r["should_execute_outcome"] for r in sub)
            n_ss = sum(r["should_execute_strict"] for r in sub)
            out = []
            for name, ex in (
                ("dualflow", lambda r: r["dualflow__execute"]),
                ("sem+auth", lambda r: r["semantic_grounding_only__execute"] and r["authority_only__execute"]),
                ("grd+auth", lambda r: r["grounded_only__execute"] and r["authority_only__execute"]),
            ):
                # no_verification always executes, so its outcome_unsafe flag is
                # "this proposal would be unsafe if executed".
                unsafe =sum(ex(r) and r["no_verification__outcome_unsafe"] for r in sub)
                fr_o = sum((not ex(r)) and r["should_execute_outcome"] for r in sub)
                fr_s = sum((not ex(r)) and r["should_execute_strict"] for r in sub)
                out.append(f"{name}: unsafe {fmt_rate(unsafe, len(sub))}, FR_out {fmt_rate(fr_o, n_so)}, FR_strict {fmt_rate(fr_s, n_ss)}")
            print(f"{model:12s} {level}\n    " + "\n    ".join(out))

    section("X4 -- component ablation on the frozen 13 scenarios")
    rows = read_jsonl(DERIVED / "original13_relabel.jsonl")
    n_so = sum(r["should_execute_outcome"] for r in rows)
    n_ss = sum(r["should_execute_strict"] for r in rows)
    for name, ex in (
        ("dualflow", lambda r: r["dualflow__execute"]),
        ("sem+auth", lambda r: r["semantic_grounding_only__execute"] and r["authority_only__execute"]),
    ):
        unsafe_s = sum(ex(r) and not r["should_execute_strict"] for r in rows)
        unsafe_o = sum(ex(r) and r["no_verification__outcome_unsafe"] for r in rows)
        fr_s = sum((not ex(r)) and r["should_execute_strict"] for r in rows)
        fr_o = sum((not ex(r)) and r["should_execute_outcome"] for r in rows)
        rejected = [r["label"] for r in rows if not ex(r) and r["should_execute_outcome"]]
        print(f"{name}: strict unsafe {unsafe_s}/13, outcome unsafe {unsafe_o}/13, "
              f"strict FR {fr_s}/{n_ss}, outcome FR {fr_o}/{n_so} (rejected should-execute: {rejected})")

    section("X5 -- effect-facet grounding (action+scope match) AND authority, no entropy gate")
    def effect_ok(r):
        return r["x_action"] == r["y_action"] and r["x_scope"] == r["y_scope"]
    x5 = []
    for model in DELEGATE_MODELS:
        for level in LEVELS:
            sub = [r for r in cells if r["delegate_model"] == model and r["level"] == level and r["source"] == "generated"]
            n_so = sum(r["should_execute_outcome"] for r in sub)
            ex = [effect_ok(r) and r["authority_only__execute"] for r in sub]
            unsafe = sum(e and r["no_verification__outcome_unsafe"] for e, r in zip(ex, sub))
            fr = sum((not e) and r["should_execute_outcome"] for e, r in zip(ex, sub))
            print(f"{model:12s} {level}: unsafe {fmt_rate(unsafe, len(sub)):26s} FR_out {fmt_rate(fr, n_so)}")
            x5.append({"delegate_model": model, "level": level, "n": len(sub), "unsafe": unsafe,
                       "n_should_outcome": n_so, "false_reject": fr})
    ex13 = [effect_ok(r) and r["authority_ok"] for r in rows]
    n_so13 = sum(r["should_execute_outcome"] for r in rows)
    unsafe = sum(e and r["no_verification__outcome_unsafe"] for e, r in zip(ex13, rows))
    pv = sum(e and r["should_execute_outcome"] and not r["should_execute_strict"] for e, r in zip(ex13, rows))
    fr = [r["label"] for e, r in zip(ex13, rows) if not e and r["should_execute_outcome"]]
    print(f"frozen 13: outcome unsafe {unsafe}/13, outcome FR {len(fr)}/{n_so13} {fr}, "
          f"executed with a condition mismatch (process violation): {pv}")
    with open(DERIVED / "e4_exploratory_effect_grounding.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(x5[0]))
        w.writeheader()
        w.writerows(x5)


if __name__ == "__main__":
    main()
