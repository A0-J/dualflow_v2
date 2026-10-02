from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _norm(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


@dataclass(frozen=True)
class Interpretation:
    """`condition` is a SET of independent policy tags (e.g.
    `{"recipient=approved_auditor", "human_approval=true"}`), not a single
    nullable string -- real authorization policy is usually more than one
    fact that must all hold (2026-10, Experiment Plan v2 exfiltration-
    scenario redesign). Agents report which conditions they believe are
    already satisfied; `check_authority()` (authority.py) is the only
    place that decides whether that's enough."""

    action: str
    resource: str
    scope: str
    condition: frozenset[str] = frozenset()

    def canonical(self) -> "Interpretation":
        return Interpretation(
            action=_norm(self.action),
            resource=_norm(self.resource),
            scope=_norm(self.scope),
            condition=frozenset(c for c in (_norm(x) for x in self.condition) if c),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "resource": self.resource,
            "scope": self.scope,
            "condition": sorted(self.condition),
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Interpretation":
        raw_condition = obj.get("condition")
        if raw_condition is None:
            raw_condition = []
        elif isinstance(raw_condition, str):
            raw_condition = [raw_condition]  # tolerate a bare string, don't require it
        return cls(
            action=str(obj.get("action", "")),
            resource=str(obj.get("resource", "")),
            scope=str(obj.get("scope", "")),
            condition=frozenset(str(c) for c in raw_condition),
        ).canonical()


@dataclass(frozen=True)
class AuthorityBudget:
    """`required_conditions` is a SET -- ALL of them must be present in a
    proposal's `condition` set (subset check in authority.py) for the
    condition axis to pass. Deterministic: no LLM call, no fuzzy matching,
    exact string equality after canonicalization only."""

    allowed_actions: tuple[str, ...]
    resource: str
    scope_prefix: str
    required_conditions: frozenset[str] = frozenset()

    def canonical(self) -> "AuthorityBudget":
        return AuthorityBudget(
            allowed_actions=tuple(sorted({_norm(a) for a in self.allowed_actions})),
            resource=_norm(self.resource),
            scope_prefix=_norm(self.scope_prefix),
            required_conditions=frozenset(c for c in (_norm(x) for x in self.required_conditions) if c),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_actions": list(self.allowed_actions),
            "resource": self.resource,
            "scope_prefix": self.scope_prefix,
            "required_conditions": sorted(self.required_conditions),
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "AuthorityBudget":
        raw = obj.get("required_conditions")
        if raw is None:
            raw = obj.get("required_condition")  # tolerate the old single-field name
            raw = [raw] if raw else []
        return cls(
            allowed_actions=tuple(obj["allowed_actions"]),
            resource=obj["resource"],
            scope_prefix=obj["scope_prefix"],
            required_conditions=frozenset(str(c) for c in raw),
        ).canonical()


@dataclass(frozen=True)
class LLMResponse:
    text: str
    requested_model: str
    served_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    latency_ms: float
    temperature: float
    top_p: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
