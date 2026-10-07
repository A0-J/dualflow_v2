"""Structured-intent control (post-hoc; added after the pre-registered
analysis, 0 API calls).

Question: what if the Planner Agent sent its intended task as a structured
(action, resource, scope) tuple instead of a natural-language delegation?
The Executor Agent's interpretation step then disappears; the tuple that
executes is the Planner Agent's own statement of its task, and only the
Authorization Flow remains to be checked.

The Planner Agent restatements (with task context) collected for the
Semantic Flow are exactly such structured statements, so they are reused
as the transmitted tuple:
  - modal: the Planner Agent's modal restatement per delegation text
  - single: every individual restatement treated as the transmitted tuple
Units are scenarios (160), not cells: no Executor Agent is involved.
"""

from __future__ import annotations

from dualflow.flows import AuthorizationPolicy, AuthorizationState, Proposal, authorization_flow, modal

from analyze import label, load_samples, outcomes
from common import DERIVED, fmt, scenarios


def main() -> None:
    _, planner = load_samples()
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    for unit in ("modal", "single"):
        by_case = {c: {"n": 0, "unsafe": 0, "should": 0, "fr": 0, "mismatch": 0} for c in ("C1", "C2", "C3", "C4")}
        for s in scenarios():
            policy = AuthorizationPolicy.from_dict(s["policy"])
            state = AuthorizationState.from_dict(s["state"])
            intended = Proposal.from_dict(s["intended"])
            restatements = planner[(s["text_id"], "context")]
            transmitted = [modal(restatements)] if unit == "modal" else restatements
            for o in transmitted:
                lab = label(o, intended, policy, state)
                out = outcomes(authorization_flow(o, policy, state).allowed, lab, o)
                agg = by_case[s["case"]]
                agg["n"] += 1
                agg["unsafe"] += out["unsafe"]
                agg["should"] += lab["should_execute"]
                agg["fr"] += out["false_reject"]
                agg["mismatch"] += not lab["intended"]
        say(f"## Structured intent ({unit} Planner Agent restatement) + Authorization Flow")
        tot = {k: sum(v[k] for v in by_case.values()) for k in ("n", "unsafe", "should", "fr", "mismatch")}
        for case, a in list(by_case.items()) + [("all", tot)]:
            say(f"{case:4s} unsafe {fmt(a['unsafe'], a['n']):24s} false rejection {fmt(a['fr'], a['should']):24s} "
                f"transmitted tuple != intended task: {a['mismatch']}/{a['n']}")
        say()

    DERIVED.mkdir(parents=True, exist_ok=True)
    (DERIVED / "structured_intent_control_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
