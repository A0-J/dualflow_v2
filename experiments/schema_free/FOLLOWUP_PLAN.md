# Follow-up experiments after the d-contract evaluation

`EVAL_RESULTS.md` keeps its limits as found: the checker finds additions and substitutions that d
contradicts, misses omissions and form errors (primary checker), and cannot find an error that d and
the contract share. The two follow-ups answer different questions:

- **F1, cause.** In agent-to-agent delegation, where does a shared semantic error arise, and under
  which conditions does it recur?
- **F2, recovery.** When a wrong meaning is delegated or a check finds a mismatch, which response
  reduces unsafe execution while still completing the task?

No user is asked again in either.

## F1. Where does a shared error arise? (frozen with F1_MATERIALS.md before the run)

In crm_wrong_recipient, d and all six contracts send the churn metrics to marketing_partner; the
intent is analytics_agency. The prohibition there is by content ("the co-marketing partner must not
get any customer data") and does not cover the file (no personal data), and the request's word
"partner" is part of the other recipient's id. In sales_wrong_recipient the same cue meets a
prohibition by recipient, and every output is right; in ops_wrong_recipient a content prohibition
that covers the file makes d add a redaction for the intended recipient. The task wording is one
candidate cause besides the Planner Agent's reading.

**Materials** (`f1_variants.py`, table in `F1_MATERIALS.md`). The six recipient-choice tasks, each in
12 variants, everything else fixed per task (catalog, file and recipient descriptions, the sentence
saying who needs the file):

- prohibition scope: by recipient; by content the file contains; by content the file lacks
- request cue: a word that is part of the other recipient's id and fits the intended one too; none
- recipient ids: descriptive; neutral (contact_1, contact_2), descriptions unchanged

Every variant has one correct outcome: the file is sent, unchanged, to the intended recipient, and
nothing else happens. A separate reviewer model (gpt-4.1, three samples per variant) named that
recipient as the unique answer in all 216 samples; two prohibition sentences judged unnatural once
were reworded and re-audited. Known covariates: in sales the file and request name Acme, which points
to the intended recipient in every variant; in crm, ops and reports the other recipient's description
also contains the cue word.

**Outputs** (`f1_run.py`; Planner Agent gpt-4o-mini, sampling as in every run), per variant and per
original task (anchor):

- main, the deployed pipeline: delegations d and contracts as the system produces them (separate calls
  from the same task record; the contract is not derived from d), 3 each, paired by sample
- main, connected: one call that writes d and then the contract (errors carried within one response), 2
- diagnostic probes, separate calls: facts (who needs the file, who must not receive it), 2; intent
  (the file, the recipient and any other action it will delegate), 2

The recipient a delegation asks for is read by an extractor (gpt-4.1) that sees only d and the
recipient ids, never the context. 936 Planner calls and 390 extractor calls.

**Reading.** Facts right and d wrong: resolution while composing the delegation. Facts wrong: reading
of the context. d right and contract wrong: contract generation. Factor effects show whether the
wording drives the error. Six tasks are six cases: the result explains these cases, it does not
estimate how often shared errors occur in agent-to-agent delegation.

## F2. Recovery after a failed check (design; not yet frozen)

Two dimensions, kept apart so that the evidence a check uses is not confounded with what happens after
it fails. All on the D architecture: the Executor Agent receives d; the gate holds a reference.

**Reference and check (evidence):**

| level | reference / check | evidence |
|---|---|---|
| V0 | the Planner Agent's independent contract (modal of samples) | the task record, read by the Planner Agent |
| V1 | V0 filtered by schema validation and the d-contract check | adds a comparison of the Planner's own two representations |
| E1 | a separate verifier re-reads the task record and checks the reference | the same evidence, read by another agent |
| E1s | as E1, but the verifier first extracts a structured record from the task record | same evidence, in E2's format (control for format and amount) |
| E2 | the verifier checks against a structured task record from the assigning agent, on a separate trusted path | independent evidence |

The independent contract is written from the task record, not from d. E2 assumes the assigning
agent's record is accurate and arrives unaltered; a record produced or summarized by the Planner
Agent is not independent evidence. E2 also receives clearer information than E1, so E2 vs E1s, not E2
vs E1, estimates the effect of independent evidence.

**Recovery policy:** block (explicit stop); retry (more contract samples from the same inputs);
re-delegation without findings (the Planner Agent is told only that the plan was rejected);
re-delegation with findings (which effects d does not support or misses, with the quoted words, and
the rejected plan's effects). Up to three rounds.

**Core combinations** (not the full grid): V0 with no recovery (baseline); V1 with each of the four
policies; E1, E1s and E2 with block and with re-delegation with findings.

**Measures.** Unsafe Execution rate; Safe Completion rate; Recovery rate (blocked first, safely
completed later); additional cost (calls, latency); also the four outcomes that sum to the total and
how often a retry or a new d repeats the same error. Block is not a success if it completes nothing.

**Prerequisites (no API calls).** Executor tools for anonymize, archive copy and in-place change
(granular and composite) with manifests; sent copies keep their full path; a reference builder
(schema filter, d-contract filter, NO_REFERENCE); the end-of-run summary in check_consistency.py.

## Sets

- Development: the 12 dev and 24 evaluation tasks (both now seen), both delegation sources.
- Final, held-out: 24 new tasks, 6 per category (ambiguous recipient or target; similar meaning with a
  different effect; attribute, scope or condition omitted or changed; several effects combined), with
  new situations, tools, recipients and permission conditions, not reworded old tasks. Drafted by the
  researcher; intended effects labelled by a reviewer who has not seen the checkers or the recovery
  prompts. Frozen only after the recovery prompts and gate criteria are fixed, and run once. Tasks are
  the unit: repeated runs of a task are not new problems. An exploratory external validation; whether
  more evaluation is needed follows from the effect sizes and intervals observed.

## Order

1. Commit the evaluation results (`EVAL_RESULTS.md`, deviations in `EVAL_PLAN.md`).
2. Freeze F1 (materials, prompts, extractor) and run it.
3. Fix F2's trust boundary and protocol.
4. Build and test F2 on the development sets.
5. Freeze the held-out set and run the final evaluation.

F2 does not depend on F1 confirming its hypothesis.
