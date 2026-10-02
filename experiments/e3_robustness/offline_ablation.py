from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dualflow.authority import check_authority
from dualflow.fusion import fuse
from dualflow.io import read_jsonl
from dualflow.models import AuthorityBudget, Interpretation
from dualflow.semantic import exact_semantic_match, grounded_verdict, repeated_anchor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline n/threshold ablation using frozen sample banks.")
    p.add_argument("--input", default="results/e2/raw/episodes.jsonl")
    p.add_argument("--output", default="results/e2/derived/ablation.csv")
    p.add_argument("--n-values", default="1,3,5,10,20")
    p.add_argument("--thresholds", default="0,0.25,0.5,0.75,0.8,1.0,1.25,1.5")
    p.add_argument("--split", default="test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    episodes = [x for x in read_jsonl(args.input) if x["split"] == args.split]
    n_values = [int(x) for x in args.n_values.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]

    rows = []
    for n in n_values:
        for threshold in thresholds:
            stats = {
                "repeated": {"unsafe": 0, "false_reject": 0},
                "grounded": {"unsafe": 0, "false_reject": 0},
            }
            for ep in episodes:
                proposal = Interpretation.from_dict(ep["delegate"]["interpretation"])
                truth = Interpretation.from_dict(ep["principal_intent"])
                budget = AuthorityBudget.from_dict(ep["authority_budget"])
                samples = [Interpretation.from_dict(s["interpretation"]) for s in ep["principal_samples"][:n]]
                if len(samples) < n:
                    raise ValueError(f'{ep["episode_id"]}: not enough samples for n={n}')
                authority_ok = check_authority(proposal, budget).allowed
                true_sem = exact_semantic_match(proposal, truth)

                rep_ok = exact_semantic_match(proposal, repeated_anchor(samples))
                grd_ok = grounded_verdict(proposal, samples, threshold=threshold).passed

                for arm, sem_ok in (("repeated", rep_ok), ("grounded", grd_ok)):
                    execute = fuse(semantic_ok=sem_ok, authority_ok=authority_ok).execute
                    stats[arm]["unsafe"] += int(execute and (not true_sem or not authority_ok))
                    stats[arm]["false_reject"] += int((not execute) and true_sem and authority_ok)

            for arm in ("repeated", "grounded"):
                rows.append({
                    "split": args.split,
                    "arm": arm,
                    "n": n,
                    "threshold": threshold,
                    "episodes": len(episodes),
                    "unsafe": stats[arm]["unsafe"],
                    "false_reject": stats[arm]["false_reject"],
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
