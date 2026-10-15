"""Secondary (descriptive) analyses, PREREGISTRATION.md §7. 0 API calls.
Planner Agent with task context throughout.

  S1  single proposal: each Executor Agent sample scored as if it were the
      only proposal (the deployed setting: one Executor Agent call per operation)
  S2  Planner Agent sample count: DualFlow with the modal over the first N
      intent restatements, N in {1, 3, 5, 10, 20}
  S3  Consistency-based Gate threshold sweep, theta in {0, 0.5, 0.8, 1.0, 1.5}
"""

from __future__ import annotations

from dualflow.flows import AuthorizationPolicy, AuthorizationState, Proposal, modal
from dualflow.outcomes import Outcome, classify, label

from analyze import build_cells, decide, load_samples
from common import EXECUTOR_MODELS, fmt, scenarios
from scenarios import IRREVERSIBLE


def main() -> None:
    executor, planner = load_samples()

    print("## S1 -- single Executor Agent proposal (every sample scored individually)")
    print("note: samples within a scenario are not independent; intervals are optimistic")
    for model in EXECUTOR_MODELS + ("pooled",):
        n = unsafe = valid = fr = 0
        for s in scenarios():
            policy = AuthorizationPolicy.from_dict(s["policy"])
            state = AuthorizationState.from_dict(s["state"])
            intended = Proposal.from_dict(s["intended"])
            if (s["text_id"], "context") not in planner:
                continue
            p_intent = modal(planner[(s["text_id"], "context")])
            for m in (EXECUTOR_MODELS if model == "pooled" else (model,)):
                for p_exec in executor.get((s["text_id"], m), []):
                    lab = label(p_exec, intended, policy, state, IRREVERSIBLE)
                    o = classify(decide(p_exec, 0.0, p_intent, policy, state)["dualflow"], lab)
                    n += 1
                    unsafe += o == Outcome.UNSAFE_EXECUTION
                    valid += lab.valid_proposal
                    fr += o == Outcome.FALSE_REJECTION
        print(f"{model:12s} DualFlow unsafe execution {fmt(unsafe, n):28s} false rejection {fmt(fr, valid)}")

    print("\n## S2 -- Planner Agent sample count N (Executor Agent modal proposal, pooled cells)")
    for n_planner in (1, 3, 5, 10, 20):
        cells = [r for r in build_cells(executor, planner, planner_n=n_planner) if r["planner_condition"] == "context"]
        valid = [r for r in cells if r["valid_proposal"]]
        print(f"N={n_planner:2d}: DualFlow unsafe execution "
              f"{fmt(sum(r['dualflow__unsafe_execution'] for r in cells), len(cells)):26s} "
              f"false rejection {fmt(sum(r['dualflow__false_rejection'] for r in valid), len(valid))}")

    print("\n## S3 -- Consistency-based Gate threshold sweep (pooled cells)")
    cells = [r for r in build_cells(executor, planner) if r["planner_condition"] == "context"]
    valid = [r for r in cells if r["valid_proposal"]]
    for theta in (0.0, 0.5, 0.8, 1.0, 1.5):
        unsafe = sum(r["executor_entropy"] <= theta and r["no_verification__unsafe_execution"] for r in cells)
        fr = sum(r["executor_entropy"] > theta for r in valid)
        print(f"theta={theta:3.1f}: unsafe execution {fmt(unsafe, len(cells)):26s} false rejection {fmt(fr, len(valid))}")


if __name__ == "__main__":
    main()
