"""Offline scaffold for schema-independent effect contracts.

No model inference, tool manifests, or actual tool invocation is performed here.
Effect contracts describe constraints on an executor's *predicted* effects,
not a proof of what an untrusted tool will really do.
"""
from __future__ import annotations

from dataclasses import dataclass

from .flows import safe_resource_path


@dataclass(frozen=True)
class Effect:
    kind: str
    resource: str
    recipient: str = ""


@dataclass(frozen=True)
class IntentContract:
    required: frozenset[Effect]
    forbidden: frozenset[Effect]
    permitted: frozenset[Effect] = frozenset()


@dataclass(frozen=True)
class EffectAssessment:
    safe: bool
    goal_covered: bool


IRREVERSIBLE_KINDS = frozenset({"external_send", "delete"})
KNOWN_EFFECT_KINDS = frozenset({"read", "internal_write", "external_send", "delete"})


def _valid_effect(effect: Effect) -> bool:
    return (
        effect.kind in KNOWN_EFFECT_KINDS
        and safe_resource_path(effect.resource)
        and (
            bool(effect.recipient)
            and effect.recipient.replace("_", "").replace("-", "").isalnum()
            if effect.kind == "external_send"
            else effect.recipient == ""
        )
    )


def assess_effects(contract: IntentContract, proposed: frozenset[Effect]) -> EffectAssessment:
    """Fail-closed irreversible effect allowlist, plus separate goal coverage.

    Safety is evaluated for the *complete* effect set of one tool operation.
    A caller must require BOTH fields to execute. This only verifies trusted
    predicted effects; it cannot prove side effects of an untrusted tool.
    """
    all_effects = proposed | contract.required | contract.permitted | contract.forbidden
    if not all(_valid_effect(e) for e in all_effects):
        return EffectAssessment(False, False)
    allowed_irreversible = {
        e for e in contract.required | contract.permitted
        if e.kind in IRREVERSIBLE_KINDS
    }
    actual_irreversible = {
        e for e in proposed if e.kind in IRREVERSIBLE_KINDS
    }
    return EffectAssessment(
        safe=(not bool(contract.forbidden & proposed)
              and actual_irreversible.issubset(allowed_irreversible)),
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
    try:
        effects = TRUSTED_TOOL_EFFECTS[executor_id][tool_id]
    except KeyError as exc:
        raise ValueError("untrusted or unknown tool: execution denied") from exc
    if not all(_valid_effect(e) for e in effects):
        raise ValueError("invalid tool effect manifest: execution denied")
    return effects
