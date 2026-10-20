"""DualFlow verification: Semantic Flow, Authorization Flow and Execution Gate.

Notation follows the paper. The Executor Agent turns the delegation message
d into one execution proposal P_exec = (a, r, s): action, resource, scope.
Before anything executes, two independent checks run:

- Semantic Flow (V_sem): the Planner Agent generates its intended operation
  P_intent N times from its task context C_task and d, without seeing
  P_exec. The most frequent (a, r, s) is the representative intent
  P̂_intent, and V_sem = 1 iff P_exec = P̂_intent on all three fields.
- Authorization Flow (V_auth): deterministic, no LLM. The authorization
  policy π grants actions, resource prefixes and scopes and lists the
  actions that need approval. The approval state σ records the approvals
  actually granted, per exact (a, r, s). Nothing the Executor Agent says
  about approvals is consulted.
- Execution Gate: D = V_sem ∧ V_auth. The operation executes only if D = 1.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .parsing import parse_json_object


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def safe_resource_path(resource: str) -> bool:
    """Strict portable resource identifier: no aliasing or path traversal.

    Resolve no untrusted filesystem path during authorization: reject all dot
    segments, absolute paths, backslashes, and empty path segments instead.
    """
    return (
        bool(resource)
        and not resource.startswith("/")
        and "\\" not in resource
        and "\x00" not in resource
        and all(segment not in ("", ".", "..") for segment in resource.split("/"))
    )


def resource_under_prefix(resource: str, prefix: str) -> bool:
    """Match whole path segments, never a string-prefix lookalike."""
    root = prefix.rstrip("/")
    return (
        safe_resource_path(resource)
        and safe_resource_path(root)
        and (resource == root or resource.startswith(root + "/"))
    )


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
    obj = parse_json_object(text)
    missing = {"action", "resource", "scope"}.difference(obj)
    if missing:
        raise ValueError(f"Missing proposal fields: {sorted(missing)}")
    return Proposal.from_dict(obj)


def modal(proposals: Iterable[Proposal]) -> Proposal:
    """Most frequent (a, r, s); ties go to the earliest observed."""
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
    """π: what this deployment may do at all, managed outside both agents."""

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
    """σ: approvals actually granted, per exact (a, r, s), recorded outside both agents."""

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
    if not any(resource_under_prefix(p.resource, prefix) for prefix in policy.resource_prefixes):
        return AuthorizationVerdict(False, "resource_not_granted")
    if p.scope not in policy.allowed_scopes:
        return AuthorizationVerdict(False, "scope_not_granted")
    if p.action in policy.approval_required and p not in state.granted:
        return AuthorizationVerdict(False, "approval_not_granted")
    return AuthorizationVerdict(True, "authorized")


@dataclass(frozen=True)
class Decision:
    """Execution Gate output: execute = D = V_sem ∧ V_auth."""

    execute: bool
    semantic: SemanticVerdict
    authorization: AuthorizationVerdict


def dualflow(proposal: Proposal, planner_restatements: list[Proposal],
             policy: AuthorizationPolicy, state: AuthorizationState) -> Decision:
    semantic = semantic_flow(proposal, planner_restatements)
    authorization = authorization_flow(proposal, policy, state)
    return Decision(semantic.passed and authorization.allowed, semantic, authorization)
