"""DualFlow verification: Semantic Flow and Authorization Flow.

The Executor Agent proposes one operation (action, resource, scope). Before
anything executes, two independent checks run:

- Semantic Flow: does the proposal match the operation the Planner Agent
  intends? The Planner restates its intended operation N times,
  independently and from its own task context; the proposal must equal
  the Planner's modal restatement on action, resource, and scope.
- Authorization Flow: is the proposal permitted? Deterministic, no LLM:
  the policy grants actions, resource prefixes and scopes, and an action
  that requires approval passes only if the current authorization state
  holds a granted approval for exactly this (action, resource, scope).
  Nothing the Executor says about approvals is consulted.

The operation executes only if both flows pass.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .parsing import _FENCE_RE, _first_json_object


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass(frozen=True)
class Proposal:
    action: str
    resource: str
    scope: str

    def canonical(self) -> "Proposal":
        return Proposal(_norm(self.action), _norm(self.resource), _norm(self.scope))

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action, "resource": self.resource, "scope": self.scope}

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Proposal":
        return cls(obj.get("action", ""), obj.get("resource", ""), obj.get("scope", "")).canonical()


def parse_proposal(text: str) -> Proposal:
    obj = _first_json_object(_FENCE_RE.sub("", text.strip()).strip())
    missing = {"action", "resource", "scope"}.difference(obj)
    if missing:
        raise ValueError(f"Missing proposal fields: {sorted(missing)}")
    return Proposal.from_dict(obj)


def modal(proposals: Iterable[Proposal]) -> Proposal:
    """Most frequent proposal; ties go to the earliest observed."""
    items = [p.canonical() for p in proposals]
    if not items:
        raise ValueError("modal() requires at least one proposal")
    counts = Counter(items)
    return min(counts, key=lambda p: (-counts[p], items.index(p)))


@dataclass(frozen=True)
class SemanticVerdict:
    passed: bool
    planner_intent: Proposal


def semantic_flow(proposal: Proposal, planner_restatements: list[Proposal]) -> SemanticVerdict:
    intent = modal(planner_restatements)
    return SemanticVerdict(proposal.canonical() == intent, intent)


@dataclass(frozen=True)
class AuthorizationPolicy:
    allowed_actions: frozenset[str]
    resource_prefixes: tuple[str, ...]
    allowed_scopes: frozenset[str]
    approval_required: frozenset[str]

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "AuthorizationPolicy":
        return cls(
            allowed_actions=frozenset(_norm(a) for a in obj["allowed_actions"]),
            resource_prefixes=tuple(_norm(p) for p in obj["resource_prefixes"]),
            allowed_scopes=frozenset(_norm(s) for s in obj["allowed_scopes"]),
            approval_required=frozenset(_norm(a) for a in obj.get("approval_required", [])),
        )


@dataclass(frozen=True)
class AuthorizationState:
    """Approvals that have actually been granted, recorded outside any agent."""

    granted: frozenset[Proposal]

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "AuthorizationState":
        return cls(frozenset(Proposal.from_dict(g) for g in obj.get("granted", [])))


@dataclass(frozen=True)
class AuthorizationVerdict:
    allowed: bool
    reason: str


def authorization_flow(proposal: Proposal, policy: AuthorizationPolicy,
                       state: AuthorizationState) -> AuthorizationVerdict:
    p = proposal.canonical()
    if p.action not in policy.allowed_actions:
        return AuthorizationVerdict(False, "action_not_granted")
    if not any(p.resource.startswith(prefix) for prefix in policy.resource_prefixes):
        return AuthorizationVerdict(False, "resource_not_granted")
    if p.scope not in policy.allowed_scopes:
        return AuthorizationVerdict(False, "scope_not_granted")
    if p.action in policy.approval_required and p not in state.granted:
        return AuthorizationVerdict(False, "approval_not_granted")
    return AuthorizationVerdict(True, "authorized")


@dataclass(frozen=True)
class Decision:
    execute: bool
    semantic: SemanticVerdict
    authorization: AuthorizationVerdict


def dualflow(proposal: Proposal, planner_restatements: list[Proposal],
             policy: AuthorizationPolicy, state: AuthorizationState) -> Decision:
    semantic = semantic_flow(proposal, planner_restatements)
    authorization = authorization_flow(proposal, policy, state)
    return Decision(semantic.passed and authorization.allowed, semantic, authorization)
