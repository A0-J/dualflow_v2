"""Pre-registered secondary (descriptive) analyses, PREREGISTRATION.md §7.
0 API calls. Planner with task context throughout.

  S1  single proposal: each Executor sample scored as if it were the only
      proposal (the deployed setting: one Executor call per operation)
  S2  Planner sample count: DualFlow with the Planner's modal over the
      first N restatements, N in {1, 3, 5, 10, 20}
  S3  self-consistency threshold sweep, theta in {0, 0.5, 0.8, 1.0, 1.5}
"""

from __future__ import annotations

from dualflow.flows import AuthorizationPolicy, AuthorizationState, Proposal, modal
from dualflow.semantic import entropy

from analyze import build_cells, decide, label, load_samples, outcomes
from common import EXECUTOR_MODELS, fmt, scenarios


def main() -> None:
    executor, planner = load_samples()

    print("## S1 -- single Executor proposal (every sample scored individually)")
    print("note: samples within a scenario are not independent; intervals are optimistic")
    for model in EXECUTOR_MODELS + ("pooled",):
        n = unsafe = should = fr = 0
        for s in scenarios():
            policy = AuthorizationPolicy.from_dict(s["policy"])
            state = AuthorizationState.from_dict(s["state"])
            intended = Proposal.from_dict(s["intended"])
            if (s["text_id"], "context") not in planner:
                continue
            y = modal(planner[(s["text_id"], "context")])
            for m in (EXECUTOR_MODELS if model == "pooled" else (model,)):
                for x in executor.get((s["text_id"], m), []):
                    lab = label(x, intended, policy, state)
                    o = outcomes(decide(x, 0.0, y, policy, state)["dualflow"], lab, x)
                    n += 1
                    unsafe += o["unsafe"]
                    should += lab["should_execute"]
                    fr += o["false_reject"]
        print(f"{model:12s} DualFlow unsafe {fmt(unsafe, n):28s} false reject {fmt(fr, should)}")

    print("\n## S2 -- Planner sample count N (Executor modal proposal, pooled cells)")
    for n_planner in (1, 3, 5, 10, 20):
        cells = [r for r in build_cells(executor, planner, planner_n=n_planner) if r["planner_condition"] == "context"]
        should = [r for r in cells if r["should_execute"]]
        print(f"N={n_planner:2d}: DualFlow unsafe {fmt(sum(r['dualflow__unsafe'] for r in cells), len(cells)):26s} "
              f"false reject {fmt(sum(r['dualflow__false_reject'] for r in should), len(should))}")

    print("\n## S3 -- self-consistency threshold sweep (pooled cells)")
    cells = [r for r in build_cells(executor, planner) if r["planner_condition"] == "context"]
    should = [r for r in cells if r["should_execute"]]
    for theta in (0.0, 0.5, 0.8, 1.0, 1.5):
        unsafe = sum(r["executor_entropy"] <= theta and r["no_verification__unsafe"] for r in cells)
        fr = sum(r["executor_entropy"] > theta for r in should)
        print(f"theta={theta:3.1f}: unsafe {fmt(unsafe, len(cells)):26s} false reject {fmt(fr, len(should))}")


if __name__ == "__main__":
    main()
