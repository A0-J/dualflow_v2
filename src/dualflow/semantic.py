from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .models import Interpretation


FACETS = ("action", "resource", "scope", "condition")


def entropy(values: Iterable[str | None]) -> float:
    values = list(values)
    if not values:
        raise ValueError("entropy requires at least one value")
    counts = Counter(values)
    n = len(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _mode(values: list[str | None]) -> str | None:
    """Deterministic plurality: count desc, then first observed occurrence."""
    counts = Counter(values)
    first = {value: values.index(value) for value in counts}
    return min(counts, key=lambda value: (-counts[value], first[value]))


def single_anchor(samples: list[Interpretation]) -> Interpretation:
    if not samples:
        raise ValueError("single_anchor requires at least one sample")
    return samples[0].canonical()


def repeated_anchor(samples: list[Interpretation]) -> Interpretation:
    if not samples:
        raise ValueError("repeated_anchor requires at least one sample")
    normalized = [s.canonical() for s in samples]
    counts = Counter(normalized)
    first = {value: normalized.index(value) for value in counts}
    return min(counts, key=lambda value: (-counts[value], first[value]))


@dataclass(frozen=True)
class FacetEvidence:
    value: str | None
    entropy: float
    confirmed: bool


@dataclass(frozen=True)
class GroundedSemanticVerdict:
    passed: bool
    facets: dict[str, FacetEvidence]
    mismatches: tuple[str, ...]


def grounded_verdict(
    proposal: Interpretation,
    samples: list[Interpretation],
    *,
    threshold: float,
) -> GroundedSemanticVerdict:
    if not samples:
        raise ValueError("grounded_verdict requires at least one sample")

    proposal = proposal.canonical()
    normalized = [s.canonical() for s in samples]
    facets: dict[str, FacetEvidence] = {}
    mismatches: list[str] = []

    for facet in FACETS:
        values = [getattr(sample, facet) for sample in normalized]
        h = entropy(values)
        mode = _mode(values)
        confirmed = h <= threshold
        facets[facet] = FacetEvidence(mode, h, confirmed)
        if confirmed and getattr(proposal, facet) != mode:
            mismatches.append(facet)

    return GroundedSemanticVerdict(
        passed=not mismatches,
        facets=facets,
        mismatches=tuple(mismatches),
    )


def exact_semantic_match(a: Interpretation, b: Interpretation) -> bool:
    return a.canonical() == b.canonical()
