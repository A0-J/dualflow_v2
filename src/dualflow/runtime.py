"""End-to-end DualFlow runtime.

One episode:
  1. The Planner Agent writes the delegation message d from the user request
     and its task context C_task.
  2. The Executor Agent turns d into one tool call, the execution proposal
     P_exec.
  3. The Execution Gate intercepts the tool call before it runs. The Planner
     Agent restates its intended operation N times from C_task and d,
     without seeing P_exec, and DualFlow decides D = V_sem ∧ V_auth
     (flows.py).
  4. If D = 1, the tool runs in the sandbox and the episode ends.
  5. If D = 0, the recovery policy either returns a new delegation message,
     which starts the next round, or ends the episode.

Unparseable model output is retried up to MAX_ATTEMPTS times. After that
the error propagates, and nothing executes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, TypeVar

from .flows import AuthorizationPolicy, AuthorizationState, Decision, Proposal, dualflow
from .llm import LLMResponse
from .outcomes import IRREVERSIBLE_ACTIONS, Outcome, classify, label
from .roles import DelegationCall, ExecutorAgent, PlannerAgent
from .sandbox import Sandbox, ToolResult

MAX_ATTEMPTS = 4
T = TypeVar("T")


def _retry(call: Callable[[], T]) -> T:
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except ValueError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Task:
    """Inputs of one end-to-end task.

    The intended operation o* is deliberately not a field: it is ground
    truth for evaluation only (`evaluate_episode`), never shown to an agent.
    """

    user_request: str
    task_context: str             # C_task, seen only by the Planner Agent
    runtime_context: str          # actions, resources and scopes, seen by both agents
    policy: AuthorizationPolicy   # π
    state: AuthorizationState     # σ


@dataclass(frozen=True)
class Round:
    delegation: str
    proposal: Proposal
    restatements: tuple[Proposal, ...]
    decision: Decision
    result: ToolResult | None
    responses: tuple[LLMResponse, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "delegation": self.delegation,
            "p_exec": self.proposal.to_dict(),
            "p_intent": self.decision.semantic.planner_intent.to_dict(),
            "n_restatements": len(self.restatements),
            "v_sem": self.decision.semantic.passed,
            "v_auth": self.decision.authorization.allowed,
            "auth_reason": self.decision.authorization.reason,
            "execute": self.decision.execute,
            "result": self.result.to_dict() if self.result else None,
            "responses": [r.to_dict() for r in self.responses],
        }


@dataclass(frozen=True)
class Episode:
    rounds: tuple[Round, ...]
    stop_reason: str  # "executed", "recovery_stopped" or "max_rounds"

    def to_dict(self) -> dict[str, object]:
        return {"stop_reason": self.stop_reason, "rounds": [r.to_dict() for r in self.rounds]}


class Recovery(Protocol):
    """Decides what follows a rejection: a new delegation message, or None to stop."""

    def __call__(self, planner: PlannerAgent, task: Task, rounds: Sequence[Round]) -> DelegationCall | None: ...


def no_recovery(planner: PlannerAgent, task: Task, rounds: Sequence[Round]) -> None:
    """The Execution Gate as evaluated in the paper: a rejection ends the episode."""
    return None


def clarify_on_semantic_rejection(planner: PlannerAgent, task: Task,
                                  rounds: Sequence[Round]) -> DelegationCall | None:
    """Ask the Planner Agent for a new d when the Semantic Flow rejected the proposal.

    When only the Authorization Flow rejected it, P_exec equals the Planner
    Agent's own representative intent, so re-delegating cannot help and the
    episode ends. Obtaining a missing approval is outside the runtime.
    """
    last = rounds[-1]
    if last.decision.semantic.passed:
        return None
    return _retry(lambda: planner.clarify(
        user_request=task.user_request, private_context=task.task_context, context=task.runtime_context,
        previous_delegation=last.delegation, rejected_proposal=last.proposal))


def run_episode(task: Task, planner: PlannerAgent, executor: ExecutorAgent, sandbox: Sandbox, *,
                n_restatements: int = 20, max_rounds: int = 3, recovery: Recovery = no_recovery,
                workers: int = 8) -> Episode:
    if n_restatements < 1 or max_rounds < 1:
        raise ValueError("n_restatements and max_rounds must be at least 1")
    call = _retry(lambda: planner.delegate(
        user_request=task.user_request, private_context=task.task_context, context=task.runtime_context))
    rounds: list[Round] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, n_restatements))) as pool:
        while True:
            delegation = call.delegation
            proposed = _retry(lambda: executor.propose(delegation=delegation, context=task.runtime_context))
            restated = list(pool.map(lambda _: _retry(lambda: planner.restate(
                delegation=delegation, private_context=task.task_context, context=task.runtime_context)),
                range(n_restatements)))
            restatements = tuple(r.proposal for r in restated)
            decision = dualflow(proposed.proposal, list(restatements), task.policy, task.state)
            result = _execute(sandbox, proposed.proposal) if decision.execute else None
            rounds.append(Round(delegation, proposed.proposal, restatements, decision, result,
                                (call.response, proposed.response, *(r.response for r in restated))))
            if decision.execute:
                return Episode(tuple(rounds), "executed")
            if len(rounds) >= max_rounds:
                return Episode(tuple(rounds), "max_rounds")
            next_call = recovery(planner, task, rounds)
            if next_call is None:
                return Episode(tuple(rounds), "recovery_stopped")
            call = next_call


def _execute(sandbox: Sandbox, proposal: Proposal) -> ToolResult:
    """A permitted tool call can still fail, e.g. on a file that does not exist."""
    try:
        return sandbox.execute(proposal)
    except (FileNotFoundError, ValueError) as e:
        return ToolResult(proposal.canonical(), "error", "", error=f"{type(e).__name__}: {e}")


@dataclass(frozen=True)
class EpisodeOutcome:
    round_outcomes: tuple[Outcome, ...]   # one decision outcome per round
    task_completed: bool                  # o* was executed without a tool error
    unsafe_execution: bool                # some round was an Unsafe Execution


def evaluate_episode(episode: Episode, task: Task, intended: Proposal,
                     irreversible: frozenset[str] = IRREVERSIBLE_ACTIONS) -> EpisodeOutcome:
    outcomes = tuple(classify(r.decision.execute, label(r.proposal, intended, task.policy, task.state, irreversible))
                     for r in episode.rounds)
    completed = any(r.decision.execute and r.result is not None and r.result.error is None
                    and r.proposal.canonical() == intended.canonical() for r in episode.rounds)
    return EpisodeOutcome(outcomes, completed, Outcome.UNSAFE_EXECUTION in outcomes)
