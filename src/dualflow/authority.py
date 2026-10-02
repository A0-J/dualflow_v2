from __future__ import annotations

from dataclasses import dataclass

from .models import AuthorityBudget, Interpretation


@dataclass(frozen=True)
class AuthorityVerdict:
    allowed: bool
    reason: str


def _scope_within(scope: str, prefix: str) -> bool:
    return scope.strip().lower().startswith(prefix.strip().lower())


def check_authority(
    proposal: Interpretation,
    budget: AuthorityBudget,
) -> AuthorityVerdict:
    """Deterministic, no LLM call. Checked in order: action grant -> resource
    -> scope -> required conditions (subset check -- ALL of
    `budget.required_conditions` must appear in `proposal.condition`).

    This function is the ONLY authority on authorization. Semantic
    verification (semantic.py) never substitutes for this check, and this
    check never looks at whether the proposal matches the Principal's
    actual intent -- that's semantic.py's job, strictly separate (2026-10,
    per instruction: neither axis's responsibility may absorb the other's)."""
    p = proposal.canonical()
    b = budget.canonical()

    if p.action not in b.allowed_actions:
        return AuthorityVerdict(False, "action_not_delegated")
    if p.resource != b.resource:
        return AuthorityVerdict(False, "resource_not_delegated")
    if not _scope_within(p.scope, b.scope_prefix):
        return AuthorityVerdict(False, "scope_exceeded")
    missing = b.required_conditions - p.condition
    if missing:
        return AuthorityVerdict(False, f"required_condition_missing:{','.join(sorted(missing))}")
    return AuthorityVerdict(True, "allowed")
