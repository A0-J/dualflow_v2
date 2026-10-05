"""N-sampling robustness check -- offline, 0 new API calls.

Question: how many independent Delegate samples are actually needed
before the empirical interpretation distribution (modal value, entropy)
stabilizes? Reuses the already-collected 20-sample Delegate banks for
every scenario that has one (S1/S2/S3/S4/S5/V1 from
`results/delegate_repeated_sampling/raw/episodes.jsonl`, A3 from
`results/delegate_repeated_sampling/authority_risk/raw/a3_episode.jsonl`)
and recomputes the action-facet modal value + entropy from the first N
samples of each bank, for N in {3, 5, 10, 15, 20} -- a deterministic
prefix of the SAME already-collected samples, not a new draw, exactly
matching `experiments/e3_robustness/offline_ablation.py`'s precedent of
reusing frozen data for a sample-count sweep. A1/A2 reuse V1's bank
already covered here, so they are not repeated separately.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from dualflow.io import read_jsonl, write_jsonl
from dualflow.models import Interpretation
from dualflow.semantic import entropy

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SIX_PATH = _ROOT / "results" / "delegate_repeated_sampling" / "raw" / "episodes.jsonl"
_A3_PATH = _ROOT / "results" / "delegate_repeated_sampling" / "authority_risk" / "raw" / "a3_episode.jsonl"
_OUT_DIR = _ROOT / "results" / "delegate_repeated_sampling" / "derived"

N_VALUES = (3, 5, 10, 15, 20)
THRESHOLD = 0.8


def _mode(values: list) -> object:
    counts = Counter(values)
    first = {v: values.index(v) for v in counts}
    return min(counts, key=lambda v: (-counts[v], first[v]))


def _facet_values(samples: list, facet: str) -> list:
    vals = [getattr(Interpretation.from_dict(c["interpretation"]).canonical(), facet) for c in samples]
    return [tuple(sorted(v)) if isinstance(v, frozenset) else v for v in vals]


def main() -> None:
    episodes = read_jsonl(_SIX_PATH) + read_jsonl(_A3_PATH)
    rows = []

    for ep in episodes:
        for facet in ("action", "condition"):
            values_full = _facet_values(ep["delegate_samples"], facet)
            assert len(values_full) == 20, f'{ep["scenario_id"]}: expected 20 samples, got {len(values_full)}'
            modal_n20 = _mode(values_full)

            for n in N_VALUES:
                prefix = values_full[:n]
                h = entropy(prefix)
                modal = _mode(prefix)
                rows.append({
                    "scenario_id": ep["scenario_id"],
                    "facet": facet,
                    "n": n,
                    "entropy": h,
                    "modal_value": modal,
                    "stable": h <= THRESHOLD,
                    "matches_n20_modal": modal == modal_n20,
                })

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "n_sampling_robustness.jsonl"
    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} rows -> {out_path}")

    csv_path = _OUT_DIR / "n_sampling_robustness.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "facet", "n", "entropy", "modal_value", "stable", "matches_n20_modal"])
        for r in rows:
            w.writerow([r["scenario_id"], r["facet"], r["n"], f'{r["entropy"]:.4f}',
                        r["modal_value"], r["stable"], r["matches_n20_modal"]])
    print(f"saved {csv_path}")

    for facet in ("action", "condition"):
        print(f"\n=== {facet} facet entropy by N ===")
        print(f"{'scenario_id':30s}" + "".join(f"{f'N={n}':>10s}" for n in N_VALUES))
        subset = [r for r in rows if r["facet"] == facet]
        scenario_ids = sorted({r["scenario_id"] for r in subset})
        by_scenario = {sid: {r["n"]: r for r in subset if r["scenario_id"] == sid} for sid in scenario_ids}
        for sid in scenario_ids:
            line = f"{sid:30s}"
            for n in N_VALUES:
                r = by_scenario[sid][n]
                line += f"{r['entropy']:>10.3f}"
            print(line)

    mismatches = [r for r in rows if not r["matches_n20_modal"]]
    print(f"\nmodal value disagrees with the N=20 modal: {len(mismatches)}/{len(rows)} rows")
    for r in mismatches:
        print(f'  {r["scenario_id"]} [{r["facet"]}] N={r["n"]}: modal={r["modal_value"]}')


if __name__ == "__main__":
    main()
