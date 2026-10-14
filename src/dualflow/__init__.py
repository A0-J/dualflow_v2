"""DualFlow: Semantic Flow and Authorization Flow verification of execution
proposals in agent-to-agent delegation."""

from .flows import (
    AuthorizationPolicy, AuthorizationState, AuthorizationVerdict, Decision, Proposal,
    SemanticVerdict, authorization_flow, dualflow, modal, parse_proposal, semantic_flow,
)
from .outcomes import IRREVERSIBLE_ACTIONS, Label, Outcome, classify, label
from .roles import DelegationCall, ExecutorAgent, PlannerAgent, RoleCall
from .runtime import (
    Episode, EpisodeOutcome, Recovery, Round, Task, clarify_on_semantic_rejection,
    evaluate_episode, no_recovery, run_episode,
)
from .sandbox import Sandbox, ToolResult

__all__ = [
    "AuthorizationPolicy", "AuthorizationState", "AuthorizationVerdict", "Decision", "Proposal",
    "SemanticVerdict", "authorization_flow", "dualflow", "modal", "parse_proposal", "semantic_flow",
    "IRREVERSIBLE_ACTIONS", "Label", "Outcome", "classify", "label",
    "DelegationCall", "ExecutorAgent", "PlannerAgent", "RoleCall",
    "Episode", "EpisodeOutcome", "Recovery", "Round", "Task", "clarify_on_semantic_rejection",
    "evaluate_episode", "no_recovery", "run_episode",
    "Sandbox", "ToolResult",
]
