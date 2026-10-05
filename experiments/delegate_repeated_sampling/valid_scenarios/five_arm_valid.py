"""5-arm comparison on the valid (should-execute) scenario subset --
V2-V5 (160 new API calls, see collect.py) + V1 (reused, already in
`results/delegate_repeated_sampling/raw/episodes.jsonl`, 0 new calls).

Purpose: measure false rejection / utility, not safety -- these 5
scenarios are all Semantic PASS, Authority PASS, Expected=EXECUTE. The
9-scenario safety result (S1-S5, A1-A3, V1) is unchanged and not
recomputed here; this is an additive utility check.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DRS_DIR = _HERE.parent  # experiments/delegate_repeated_sampling
_ROOT = _DRS_DIR.parent.parent
sys.path.insert(0, str(_DRS_DIR))

from dualflow.io import read_jsonl, write_jsonl
from dualflow.models import AuthorityBudget, Interpretation

from analyze import _side_summary
from five_arm_comparison import ARMS, THRESHOLD, score_episode

_V2_V5_PATH = _ROOT / "results" / "delegate_repeated_sampling" / "valid_scenarios" / "raw" / "episodes.jsonl"
_V1_PATH = _ROOT / "results" / "delegate_repeated_sampling" / "raw" / "episodes.jsonl"
_OUT_DIR = _ROOT / "results" / "delegate_repeated_sampling" / "valid_scenarios" / "derived"


def _score(ep: dict) -> dict:
    truth = Interpretation.from_dict(ep["principal_intent"])
    budget = AuthorityBudget.from_dict(ep["authority_budget"])
    delegate_samples = [Interpretation.from_dict(c["interpretation"]) for c in ep["delegate_samples"]]
    principal_samples = [Interpretation.from_dict(c["interpretation"]) for c in ep["principal_samples"]]
    delegate = _side_summary(delegate_samples, truth, THRESHOLD)
    principal = _side_summary(principal_samples, truth, THRESHOLD)
    x = Interpretation.from_dict(delegate["anchor"])
    y = Interpretation.from_dict(principal["anchor"])
    scored = score_episode(x=x, h=delegate["entropy"], y=y, budget=budget, truth=truth)
    return {
        "scenario_id": ep["scenario_id"], "label": ep.get("label", "V1"),
        "delegate_modal_action": x.action, "delegate_entropy": delegate["entropy"],
        "principal_modal_action": y.action, **scored,
    }


def main() -> None:
    v2_v5 = read_jsonl(_V2_V5_PATH)
    v1 = [e for e in read_jsonl(_V1_PATH) if e["scenario_id"] == "exfil_send_external_p1"]
    assert len(v1) == 1
    v1[0] = {**v1[0], "label": "V1"}

    rows = [_score(ep) for ep in [*v1, *v2_v5]]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "five_arm_valid.jsonl"
    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} scenarios -> {out_path}")

    csv_path = _OUT_DIR / "five_arm_valid.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["scenario_id", "label", "delegate_modal_action", "delegate_entropy",
                   "principal_modal_action", "true_semantic_match", "true_authority_match"]
        for arm in ARMS:
            header += [f"{arm}_execute", f"{arm}_unsafe", f"{arm}_false_reject", f"{arm}_correct_execute"]
        w.writerow(header)
        for r in rows:
            row = [r["scenario_id"], r["label"], r["delegate_modal_action"], f'{r["delegate_entropy"]:.4f}',
                   r["principal_modal_action"], r["true_semantic_match"], r["true_authority_match"]]
            for arm in ARMS:
                a = r["arms"][arm]
                row += [a["execute"], a["unsafe"], a["false_reject"], a["correct_execute"]]
            w.writerow(row)
    print(f"saved {csv_path}")

    print("\n=== per-scenario check (all should be true_semantic=True, true_authority=True) ===")
    for r in rows:
        print(f'{r["label"]:4s} {r["scenario_id"]:32s} true_semantic={r["true_semantic_match"]} '
              f'true_authority={r["true_authority_match"]}')

    n = len(rows)
    print(f"\n=== valid-scenario subset (n={n}) -- false_reject is the metric that matters here ===")
    print(f'{"arm":28s} {"false_reject":>14s} {"correct_execute":>16s} {"unsafe":>8s}')
    for arm in ARMS:
        fr = sum(r["arms"][arm]["false_reject"] for r in rows)
        ce = sum(r["arms"][arm]["correct_execute"] for r in rows)
        unsafe = sum(r["arms"][arm]["unsafe"] for r in rows)
        print(f'{arm:28s} {fr}/{n:<11d} {ce}/{n:<13d} {unsafe}/{n}')


if __name__ == "__main__":
    main()
