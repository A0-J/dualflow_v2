# E2 Results — Security Case Study (frozen, 378 real API calls)

**Role of this experiment (fixed framing, per instruction):** E2 is a
controlled security case study, not a comparative experiment testing
whether Grounded verification beats repeated-restatement verification —
that comparison is Phase 3C's role (the earlier summarize/export pack,
preserved separately). E2 asks a different, narrower question:

> In a data-exfiltration delegation, can Semantic Flow and Authority
> Flow independently block two different real failure modes in actual
> LLM output?

Data: 18 episodes (6 scenarios × 3 repeats), `gpt-4o-mini`, n=20 Principal
samples/episode, threshold=0.8 (both frozen project defaults, not
selected from this data). Raw: `results/e2/raw/episodes.jsonl`. Derived:
`results/e2/derived/full_results.jsonl`.

## Result table

| Scenario | Episodes | Delegate failure | Final result |
|---|---|---|---|
| `summarize` intent | 9 | 9/9 misread as `send_external` | 9/9 REJECT |
| `send_external` intent | 9 | 6/9 omitted `human_approval=true` | 6/9 REJECT |
| `send_external` intent | 9 | 3/9 fully correct | 3/9 EXECUTE |

Across all 18 episodes: **unsafe execution = 0/18, false rejection =
0/18** (every REJECT corresponds to a genuinely wrong or incomplete
Delegate proposal — none of the 9 correct-and-authorized episodes was
blocked).

## What each row means concretely

- **9/9 `summarize` → `send_external` misread**: a confident,
  *authorized* misread — the Delegate generally has `send_external`
  capability, so Authority alone would have let all 9 through. Only
  Semantic verification (comparing against Principal's independently
  reconstructed intent) catches that this isn't what was actually asked.
  This reproduces, far more strongly than the original summarize/export
  pack (100% here vs. 20% there), the exact "authorized-but-unintended
  action" risk this project has studied from the start — now in a
  scenario where the consequence is literally data leaving the trust
  boundary.
- **6/9 `send_external` → missing `human_approval=true`**: a distinct
  failure mode from the above — the Delegate gets the action, resource,
  and scope exactly right, but silently drops one of two required
  condition tags. This is a condition-level omission, not an action
  misread, and it is caught the same way: Principal's reconstruction
  reliably includes both conditions, so the mismatch is real, not
  spurious.
- **3/9 fully correct**: the clean positive case — Semantic and
  Authority both pass, EXECUTE, and (per `tests/test_execution_gating.py`)
  this is the only path that ever produces a file in
  `workspace/external_outbox/`.

## Why Single, Repeated, and Grounded are identical here

> All three semantic-verification arms produced identical decisions
> because Principal reconstructions were fully consistent across
> samples, leaving no uncertain facet for facet-grounded relaxation.

Checked directly, not assumed: every `send_external`-intent episode's
`condition` facet has entropy `-0.0` (i.e. 0, floating-point sign
artifact) across all 20 Principal samples — fully confirmed every time.
Grounded's "only block on confirmed-facet mismatches" rule therefore
never differs from Single/Repeated's plain exact-match here, because
there is no *unconfirmed* mismatch for it to selectively ignore. This is
a structural property of this scenario's Principal evidence, not a
defect in the mechanism or the run.

**Deliberately not done**: adding artificial condition ambiguity to make
the three arms diverge. Doing that only after seeing this frozen result
would read as scenario-shopping for a difference post hoc. Arm
differentiation is Phase 3C's documented result already; this case study
's job is the semantic/authority independence story above, and it
answers that question cleanly on its own.
