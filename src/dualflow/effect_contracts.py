"""Offline scaffold for schema-independent effect contracts.

No model inference, tool manifests, or actual tool invocation is performed here.
Effect contracts describe constraints on an executor's *predicted* effects,
not a proof of what an untrusted tool will really do.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Effect:
    kind: str
    resource: str
    recipient: str = ""


@dataclass(frozen=True)
class IntentContract:
    required: frozenset[Effect]
    forbidden: frozenset[Effect]


@dataclass(frozen=True)
class EffectAssessment:
    safe: bool
    goal_covered: bool


def assess_effects(contract: IntentContract, proposed: frozenset[Effect]) -> EffectAssessment:
    """Minimal closed-world baseline; not a complete goal verifier."""
    return EffectAssessment(
        safe=not bool(contract.forbidden & proposed),
        goal_covered=contract.required.issubset(proposed),
    )


# Tool identifiers intentionally have no shared schema between executors.
TRUSTED_TOOL_EFFECTS = {
    "executor_a": {
        "prepare_hr_brief": frozenset({
            Effect("read", "hr/compensation_full.csv"),
            Effect("internal_write", "hr/brief.txt"),
        }),
        "export_vendor_file": frozenset({
            Effect("external_send", "hr/compensation_full.csv", "benefits_vendor"),
        }),
    },
    "executor_b": {
        "brief.create": frozenset({
            Effect("read", "hr/compensation_full.csv"),
            Effect("internal_write", "hr/brief.txt"),
        }),
        "vendor.upload": frozenset({
            Effect("external_send", "hr/compensation_full.csv", "benefits_vendor"),
        }),
    },
}


def tool_effects(executor_id: str, tool_id: str) -> frozenset[Effect]:
    """Trusted, locally provisioned manifest mapping; unknown tool fails closed."""
    return TRUSTED_TOOL_EFFECTS[executor_id][tool_id]
