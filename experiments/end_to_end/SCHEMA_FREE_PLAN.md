# No-shared-tool-schema E2E scaffold (offline)

Status: implementation scaffold; no API collection, no effectiveness claim.

Question: When Planner and Executor do not share tool names, does comparing
trusted *effects* at the execution boundary improve safety and task completion
over natural-language and structured-goal baselines?

The Planner specifies required and forbidden effects using canonical resource
IDs (which ARE shared). Executors have distinct tool IDs but locally trusted
manifest maps tool calls to the common effect vocabulary.

Factor A: shared tool-operation schema (yes/no).
Factor B: goal channel (natural-language / structured effect contract).
Factor C: gate reference (independent planner restatement / transmitted contract).
Planner correctness and ambiguous delegation are separate stratification factors.

Baselines must use the same Authorization gate, the same set of tools, identical
task data, and an equal API-call budget or report cost as a separate axis.

Pilot tasks:
1. Internal HR summary vs forbidden external export.
2. Wrong file exported to correct recipient.
3. Correct file exported to wrong recipient.
4. Multi-step preparation requiring read and internal write.
5. Authorized but unintended reversible operation.

Primary outcomes: end-to-end unsafe effects, completed intended task,
unnecessary rejection, LLM calls, tokens, and latency. Evaluate at original-task
clusters, not paraphrase sample count.

Threat/trust assumptions:
- Tool-to-effect mappings are installed by the trusted local environment.
- Unknown or incompletely described effects fail closed.
- Manifest-reported effects are predictions, not proof of runtime effects.
- Planner goal interpretation may fail; evaluate it separately.
- Actual execution outcomes require instrumentation and post-state checks.

The module `src/dualflow/effect_contracts.py` intentionally implements only
the simplest required/forbidden effect-set baseline. This cannot establish
goal completion for multi-step tasks. It also does not prove tool manifests
correct, replace the Authorization gate, or handle partial side effects.
Do not use as a drop-in security gate before extending those properties.

## Safety assessment amendment (2026-10-20)

The first scaffold incorrectly treated all effects not explicitly forbidden
as safe. Replaced it with a fail-closed irreversible effect allowlist:
`irreversible(proposed) <= irreversible(required | permitted)`, together
with the forbidden-effect exclusion. Unknown effect kinds, malformed resource
identifiers and recipients are invalid. Unknown tool IDs raise explicit
denial errors. Goal coverage remains a separate result, and callers must
require both safety and goal coverage if enforcing one-step completion.

Regression cases: wrong export resource, wrong recipient, legitimate summary
combined with an unauthorized export, and legitimate summary combined with
an unauthorized delete. An allowlisted external send is also tested.

This is a trusted-predicted-effect prototype only. The complete tool effect
set must be present; undeclared runtime effects cannot be inferred or
prevented by this evaluator. It is not yet part of the runtime gate.

The earlier canonical effect re-scoring code under
`experiments/effect_level_verification/` was not present on this PR
branch at the time of this amendment. Integrate it by importing the single
shared decision rule rather than implementing a separate safety check.
