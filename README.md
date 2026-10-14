# DualFlow

Code and data for *DualFlow: Semantic and Authorization Verification for
Agent-to-Agent Delegation* (MobiSec 2026), and the runtime for its
end-to-end follow-up.

An Executor Agent turns a natural-language delegation message `d` from a
Planner Agent into an execution proposal `P_exec = (a, r, s)`. Before
anything runs, two independent checks run on it:

- **Semantic Flow.** The Planner Agent restates its intended operation N
  times from its task context `C_task` and `d`, without seeing `P_exec`.
  `V_sem = 1` iff `P_exec` equals the most frequent restatement `P̂_intent`.
- **Authorization Flow.** A deterministic check of `P_exec` against the
  authorization policy `π` and the approval state `σ`, both managed outside
  the agents. This gives `V_auth`.
- **Execution Gate.** `D = V_sem ∧ V_auth`. Execution happens only if `D = 1`.

## Layout

| Path | Contents |
|---|---|
| `src/dualflow/flows.py` | Semantic Flow, Authorization Flow, Execution Gate |
| `src/dualflow/roles.py` | Planner Agent and Executor Agent (LLM calls) |
| `src/dualflow/outcomes.py` | The four outcome categories (see below) |
| `src/dualflow/runtime.py`, `sandbox.py` | End-to-end runtime with sandboxed tool execution |
| `src/dualflow/llm.py`, `parsing.py`, `stats.py`, `io.py` | OpenAI client, JSON parsing, exact statistics, JSONL |
| `experiments/canonical/` | The paper's evaluation (canonical cases C1–C4) |
| `experiments/human_validation/` | Annotation kit and analysis for the intended-task labels (annotations not yet collected) |
| `experiments/end_to_end/` | Smoke runner for the end-to-end runtime |
| `results/canonical/` | Raw samples and derived tables of the paper's evaluation |
| `figures/` | `dualflow_architecture.png`, `canonical_results.png` (Figure 2) |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,figures]'
pytest -q
```

Scripts are run with `PYTHONPATH=src` as shown below. An installed package
makes the prefix unnecessary.

## Terms in code and outputs

Code, tables and figures use the paper's terms.

| Paper | Code / output columns |
|---|---|
| `P_exec`, `P̂_intent` | `p_exec`, `p_intent` (both `Proposal`) |
| `π`, `σ` | `AuthorizationPolicy`, `AuthorizationState` |
| No Verification, Consistency-based Gate, Semantic Flow-Only, Authorization Flow-Only, DualFlow | `no_verification`, `consistency_gate`, `semantic_flow_only`, `authorization_flow_only`, `dualflow` |
| Unsafe Execution, Misaligned Execution, False Rejection, Correct Decision | `unsafe_execution`, `misaligned_execution`, `false_rejection`, `correct_decision` |
| valid proposal (intended and authorized) | `valid_proposal` |
| irreversible misinterpretation | `irreversible_misinterpretation` |
| Intent Match (Table 2) | `intent_match` |

Every decision falls into exactly one of the four outcome categories, so
the four counts always sum to the number of decisions
(`dualflow.outcomes.classify`).

## Reproducing the paper

The evaluation plan was written before data collection:
`experiments/canonical/PREREGISTRATION.md`. It covers 80 delegation texts,
160 scenarios, 640 cells (scenario × Executor Agent model) and 9,600 LLM
calls. The analyses read the frozen raw samples in `results/canonical/raw/`
and make no API calls:

```bash
cd experiments/canonical
PYTHONPATH=../../src python analyze.py      # Table 1, Table 2, hypotheses H1-H5 -> results/canonical/derived/
PYTHONPATH=../../src python secondary.py    # single proposal, Planner Agent sample count, threshold sweep
PYTHONPATH=../../src python structured_intent_control.py   # post-hoc control analysis
PYTHONPATH=../../src python clustered_sensitivity.py       # post-hoc dependence sensitivity
PYTHONPATH=../../src python make_figure.py  # figures/canonical_results.png
```

`analyze.py` prints Table 1 under "all cases". Each row gives Unsafe
Execution, Misaligned Execution, False Rejection and Correct Decision:

| Method | Unsafe | Misaligned | False Rejection | Correct |
|---|---|---|---|---|
| No Verification | 310 | 135 | 0 | 195 |
| Consistency-based Gate | 304 | 111 | 8 | 217 |
| Semantic Flow-Only | 159 | 0 | 0 | 481 |
| Authorization Flow-Only | 75 | 135 | 0 | 430 |
| DualFlow | 0 | 0 | 0 | 640 |

`collect.py` and `generate_paraphrases.py` regenerate the raw data with
real API calls. They are needed only to collect the data again.

**Dependence sensitivity (post-hoc).** The cells are nested: 8 source
delegations → 80 delegation texts → 2 authorization configurations → 4
Executor Agent models. `clustered_sensitivity.py` restates the results at
two coarser levels:

- **Delegation text (80 units).** Descriptive only: direction counts and
  incidence, with no p-values, because variants of one source delegation
  are not independent.
- **Source delegation (8 units).** Exact family-incidence intervals and
  exact one-sided sign tests. With 4 families per case, p cannot fall
  below 1/16.

It does not replace the planned cell-level analysis.

## End-to-end runtime (follow-up work)

The paper evaluates gate decisions on given delegation messages. No tool
runs, and a rejection ends the attempt. `dualflow.runtime.run_episode`
closes the loop:

1. The Planner Agent writes `d` itself from a user request and `C_task`
   (`PlannerAgent.delegate`).
2. The Executor Agent proposes one tool call `P_exec`.
3. The Execution Gate intercepts the call and runs DualFlow on it.
4. If `D = 1`, the tool runs in a `Sandbox`:
   - `read` changes nothing;
   - `summarize` writes an internal summary file;
   - `send_external` copies the file to `outbox/<recipient>/`;
   - `delete` removes the file.

   Nothing leaves the machine. A rejected call never reaches the sandbox.
5. If `D = 0`, a recovery policy decides what happens next:
   - `no_recovery` stops the episode, as the paper's gate does.
   - `clarify_on_semantic_rejection` lets the Planner Agent write a new `d`
     after a Semantic Flow rejection (`PlannerAgent.clarify`). It stops
     when only the Authorization Flow rejected the call, because there
     `P_exec` already equals the Planner Agent's intent.

   Other policies, such as re-planning or approval escalation, plug in
   through the `Recovery` protocol.

`evaluate_episode` scores each round with the four outcome categories. It
also reports whether the intended operation `o*` was actually executed
(`task_completed`). `o*` is ground truth for evaluation only and is never
part of the `Task` given to the agents.

Smoke test on a canonical scenario. The canonical delegation sentence is
used as the user request:

```bash
PYTHONPATH=src python experiments/end_to_end/run_episode.py --domain audit --case C2 --dry-run   # no API calls
PYTHONPATH=src python experiments/end_to_end/run_episode.py --domain audit --case C2 --recovery clarify
```

Episodes go to `results/end_to_end/smoke/episodes.jsonl`. Each one records
every LLM response with token usage, and its sandbox is kept for
inspection.

**Not decided yet.** These must be fixed and documented before any
end-to-end data is collected:

- the end-to-end scenarios and user requests;
- the wording of the `delegate` and `clarify` instructions;
- the recovery policies to compare;
- the round limit;
- the hypotheses.

The `ExecutorAgent.propose` and `PlannerAgent.restate` instructions are the
frozen ones from the paper's evaluation.

## Earlier exploratory experiments

The exploratory experiments that preceded the canonical evaluation (E1–E5
and the repeated-sampling analyses) are not on `main`. They are preserved
at the tag `mobisec2026-submission`, which is the repository as of the
submission:

```bash
git checkout mobisec2026-submission -- experiments/e4_information_asymmetry results/e4
```
