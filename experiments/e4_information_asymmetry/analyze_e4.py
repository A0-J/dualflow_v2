"""E4 step 3: offline analysis (0 API calls), exactly as pre-registered in
PREREGISTRATION.md §5-6.

Unit of analysis = one (scenario, Delegate model, Principal level) cell:
Delegate modal interpretation X and whole-interpretation entropy H over
its 20-sample bank, Principal modal interpretation Y over its 20-sample
bank. Arms come from the frozen `decide()` (five_arm_comparison.py).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict

from dualflow.io import read_jsonl, write_jsonl
from dualflow.models import AuthorityBudget, Interpretation
from dualflow.semantic import entropy, repeated_anchor

from common import (
    ARMS, DELEGATE_MODELS, DERIVED, LEVELS, REUSED_EPISODES, SAMPLES,
    SECONDARY_ARMS, THRESHOLD, _binom_cdf, decide_all, fmt_rate, labels,
    load_scenarios,
)

PRIMARY_MODEL = "gpt-4o-mini"
# §6 sensitivity analysis: summarize-intent paraphrases that name a transfer
# verb may have drifted toward send_external during generation.
TRANSFER_VERB = re.compile(
    r"\b(send|sent|share|sharing|shared|e-?mail|forward|deliver|submit|transmit|"
    r"hand (it )?(over|off)|pass (it )?(along|on)|provide|give)\w*", re.IGNORECASE)


def load_banks():
    delegate: dict[tuple[str, str], list[Interpretation]] = defaultdict(list)
    principal: dict[tuple[str, str], list[Interpretation]] = defaultdict(list)
    for ep in read_jsonl(REUSED_EPISODES):
        sid = ep["scenario_id"]
        delegate[(sid, "gpt-4o-mini")] = [Interpretation.from_dict(c["interpretation"]) for c in ep["delegate_samples"]]
        principal[(sid, "L2")] = [Interpretation.from_dict(c["interpretation"]) for c in ep["principal_samples"]]
    for r in read_jsonl(SAMPLES):
        interp = Interpretation.from_dict(r["interpretation"])
        if r["role"] == "delegate":
            delegate[(r["scenario_id"], r["model"])].append(interp)
        else:
            principal[(r["scenario_id"], r["level"])].append(interp)
    return delegate, principal


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * _binom_cdf(min(b, c), n, 0.5))


def build_rows(scenarios, delegate, principal):
    rows = []
    for s in scenarios:
        truth = Interpretation.from_dict(s["principal_intent"])
        budget = AuthorityBudget.from_dict(s["authority_budget"])
        for model in DELEGATE_MODELS:
            d = [x.canonical() for x in delegate.get((s["scenario_id"], model), [])]
            if not d:
                continue
            x = repeated_anchor(d)
            h = entropy(d)
            h_action = entropy([i.action for i in d])
            for level in LEVELS:
                p = [y.canonical() for y in principal.get((s["scenario_id"], level), [])]
                if not p:
                    continue
                y = repeated_anchor(p)
                decisions = decide_all(x=x, h=h, y=y, principal_samples=p, budget=budget)
                row = {
                    "scenario_id": s["scenario_id"], "source": s["source"], "intent": s["intent"],
                    "delegation": s["delegation"],
                    "transfer_verb": bool(TRANSFER_VERB.search(s["delegation"])),
                    "delegate_model": model, "level": level,
                    "n_delegate": len(d), "n_principal": len(p),
                    "x_action": x.action, "x_scope": x.scope, "x_condition": sorted(x.condition),
                    "delegate_entropy": h, "delegate_action_entropy": h_action,
                    "delegate_action_share": sum(i.action == x.action for i in d) / len(d),
                    "y_action": y.action, "y_scope": y.scope, "y_condition": sorted(y.condition),
                    "principal_entropy": entropy(p),
                    "principal_action_entropy": entropy([i.action for i in p]),
                    "principal_action_correct": y.action == truth.action,
                    "confident_action_misread": x.action != truth.action and h_action <= THRESHOLD,
                }
                for arm in ARMS + SECONDARY_ARMS:
                    lab = labels(x=x, truth=truth, budget=budget, execute=decisions[arm])
                    row[f"{arm}__execute"] = decisions[arm]
                    for k in ("strict_unsafe", "strict_false_reject", "outcome_unsafe",
                              "outcome_false_reject", "process_violation"):
                        row[f"{arm}__{k}"] = lab[k]
                    row["should_execute_strict"] = lab["should_execute_strict"]
                    row["should_execute_outcome"] = lab["should_execute_outcome"]
                rows.append(row)
    return rows


def arm_table(rows, *, title):
    print(f"\n### {title}")
    n = len(rows)
    n_ss = sum(r["should_execute_strict"] for r in rows)
    n_so = sum(r["should_execute_outcome"] for r in rows)
    print(f"(n={n} cells; should-execute: strict={n_ss}, outcome={n_so})")
    print(f"{'arm':24s} | {'outcome_unsafe':26s} | {'outcome_false_reject':26s} | {'strict_unsafe':26s} | strict_false_reject")
    out = []
    for arm in ARMS + SECONDARY_ARMS:
        ou = sum(r[f"{arm}__outcome_unsafe"] for r in rows)
        of = sum(r[f"{arm}__outcome_false_reject"] for r in rows)
        su = sum(r[f"{arm}__strict_unsafe"] for r in rows)
        sf = sum(r[f"{arm}__strict_false_reject"] for r in rows)
        print(f"{arm:24s} | {fmt_rate(ou, n):26s} | {fmt_rate(of, n_so):26s} | {fmt_rate(su, n):26s} | {fmt_rate(sf, n_ss)}")
        out.append({"table": title, "arm": arm, "n": n, "n_should_strict": n_ss, "n_should_outcome": n_so,
                    "outcome_unsafe": ou, "outcome_false_reject": of,
                    "strict_unsafe": su, "strict_false_reject": sf})
    return out


def main() -> None:
    scenarios = load_scenarios()
    delegate, principal = load_banks()
    rows = build_rows(scenarios, delegate, principal)
    DERIVED.mkdir(parents=True, exist_ok=True)
    write_jsonl(DERIVED / "e4_cells.jsonl", rows)
    with open(DERIVED / "e4_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"cells={len(rows)} -> {DERIVED / 'e4_cells.csv'}")
    short = [r for r in rows if r["n_delegate"] < 20 or r["n_principal"] < 20]
    print(f"cells with <20 samples on either side: {len(short)}")

    # ---------------- H1 / H4: confident misread, by Delegate model ----------------
    print("\n## H1/H4 -- confident action misread (modal action != truth, action entropy <= 0.8)")
    h1_out = []
    one_level = [r for r in rows if r["level"] == "L2"]  # misread does not depend on Principal level
    for model in DELEGATE_MODELS:
        for source in ("generated", "original"):
            for intent in ("summarize", "send_external"):
                sub = [r for r in one_level if r["delegate_model"] == model and r["source"] == source and r["intent"] == intent]
                k = sum(r["confident_action_misread"] for r in sub)
                print(f"{model:12s} {source:9s} {intent:13s} {fmt_rate(k, len(sub))}")
                h1_out.append({"delegate_model": model, "source": source, "intent": intent, "k": k, "n": len(sub)})
    sub = [r for r in one_level if r["delegate_model"] == PRIMARY_MODEL and r["source"] == "generated" and r["intent"] == "summarize"]
    k = sum(r["confident_action_misread"] for r in sub)
    from common import clopper_pearson
    lo, _ = clopper_pearson(k, len(sub)) if sub else (float("nan"), 0)
    print(f"H1 criterion (CI lower bound > 0.5) for {PRIMARY_MODEL}: lower={lo:.3f} -> {'SUPPORTED' if lo > 0.5 else 'NOT SUPPORTED'}")

    # ---------------- Mechanism: Principal reconstruction by level ----------------
    print("\n## Principal modal action correct, by level (gpt-4o-mini Principal)")
    for source in ("generated", "original"):
        for intent in ("summarize", "send_external"):
            for level in LEVELS:
                sub = [r for r in rows if r["delegate_model"] == PRIMARY_MODEL and r["source"] == source
                       and r["intent"] == intent and r["level"] == level]
                k = sum(r["principal_action_correct"] for r in sub)
                print(f"{source:9s} {intent:13s} {level}  {fmt_rate(k, len(sub))}")

    # ---------------- H2 / H3: arms by level ----------------
    tables = []
    for model in DELEGATE_MODELS:
        for level in LEVELS:
            for source in ("generated", "original"):
                sub = [r for r in rows if r["delegate_model"] == model and r["level"] == level and r["source"] == source]
                if sub:
                    tables += arm_table(sub, title=f"{model} | {level} | {source}")

    print("\n## H2 -- paired L0 vs L2, outcome_unsafe, generated scenarios (exact McNemar)")
    for model in DELEGATE_MODELS:
        for arm in ("semantic_grounding_only", "dualflow"):
            cell = {(r["scenario_id"], r["level"]): r[f"{arm}__outcome_unsafe"]
                    for r in rows if r["delegate_model"] == model and r["source"] == "generated"}
            sids = sorted({sid for sid, _ in cell})
            b = sum(cell.get((s, "L0"), False) and not cell.get((s, "L2"), False) for s in sids)
            c = sum(cell.get((s, "L2"), False) and not cell.get((s, "L0"), False) for s in sids)
            u0 = sum(cell.get((s, "L0"), False) for s in sids)
            u2 = sum(cell.get((s, "L2"), False) for s in sids)
            print(f"{model:12s} {arm:24s} L0 unsafe={u0}/{len(sids)} L2 unsafe={u2}/{len(sids)} "
                  f"discordant b={b} c={c} p={mcnemar_exact(b, c):.4g}")

    print("\n## Exploratory -- grounded (per-facet) vs exact-match decisions differ")
    for level in LEVELS:
        sub = [r for r in rows if r["delegate_model"] == PRIMARY_MODEL and r["level"] == level]
        diff = [r["scenario_id"] for r in sub if r["grounded_only__execute"] != r["semantic_grounding_only__execute"]]
        print(f"{level}: {len(diff)}/{len(sub)} cells differ {diff[:8]}")

    print("\n## Sensitivity -- excluding summarize paraphrases with a transfer verb")
    flagged = sorted({r["scenario_id"] for r in rows if r["intent"] == "summarize" and r["source"] == "generated" and r["transfer_verb"]})
    print(f"flagged {len(flagged)}: {flagged}")
    for level in LEVELS:
        sub = [r for r in rows if r["delegate_model"] == PRIMARY_MODEL and r["level"] == level
               and r["source"] == "generated" and not (r["intent"] == "summarize" and r["transfer_verb"])]
        if sub:
            tables += arm_table(sub, title=f"SENSITIVITY {PRIMARY_MODEL} | {level} | generated minus flagged")

    for name, data in (("e4_h1_misread.csv", h1_out), ("e4_arm_tables.csv", tables)):
        with open(DERIVED / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"\nwrote {DERIVED / 'e4_h1_misread.csv'}, {DERIVED / 'e4_arm_tables.csv'}")


if __name__ == "__main__":
    main()
