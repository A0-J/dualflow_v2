from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalDecision:
    execute: bool
    reason: str


def fuse(*, semantic_ok: bool, authority_ok: bool) -> FinalDecision:
    if semantic_ok and authority_ok:
        return FinalDecision(True, "both_checks_passed")
    if not semantic_ok and not authority_ok:
        return FinalDecision(False, "semantic_and_authority_failed")
    if not semantic_ok:
        return FinalDecision(False, "semantic_failed")
    return FinalDecision(False, "authority_failed")
