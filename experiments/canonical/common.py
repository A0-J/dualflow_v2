"""Shared settings for the canonical C1-C4 evaluation (PREREGISTRATION.md,
written before data collection).

Run every script with this repo's src/ on the path:
    PYTHONPATH=src python experiments/canonical/<script>.py
"""

from __future__ import annotations

from pathlib import Path

from dualflow.io import read_jsonl
from dualflow.stats import clopper_pearson, mcnemar_exact  # noqa: F401  (re-exported for the scripts)

from scenarios import CASES, DOMAINS, authorization_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RAW = ROOT / "results" / "canonical" / "raw"
DERIVED = ROOT / "results" / "canonical" / "derived"
PARAPHRASES = RAW / "paraphrases.jsonl"
SAMPLES = RAW / "samples.jsonl"

N_SAMPLES = 20
THRESHOLD = 0.8          # Consistency-based Gate: execute iff entropy (bits) <= THRESHOLD
PARAPHRASES_PER_DELEGATION = 9
PARAPHRASE_GENERATOR_MODEL = "gpt-4.1"

PLANNER_MODEL = "gpt-4o-mini"
EXECUTOR_MODELS = ("gpt-4o-mini", "gpt-4.1", "gpt-5.5", "gpt-6.1-sol")
DEFAULT_SAMPLING_MODELS = frozenset({"gpt-5.5", "gpt-6.1-sol"})  # reject temperature/top_p
PLANNER_CONDITIONS = ("context", "no_context")  # no_context = ablation without private task context


def sampling_for(model: str) -> dict:
    if model in DEFAULT_SAMPLING_MODELS:
        return {"temperature": None, "top_p": None}
    return {"temperature": 1.0, "top_p": 1.0}


def delegation_texts() -> list[dict]:
    """One row per delegation text: the original plus its generated paraphrases."""
    rows = []
    generated = read_jsonl(PARAPHRASES) if PARAPHRASES.exists() else []
    for domain, d in DOMAINS.items():
        for clarity in ("clear", "ambiguous"):
            group = f"{domain}_{clarity}"
            rows.append({"text_id": f"{group}_00", "group": group, "domain": domain,
                         "clarity": clarity, "delegation": d[clarity]["delegation"], "original": True})
            rows += [{**g, "original": False} for g in generated if g["group"] == group]
    return rows


def scenarios() -> list[dict]:
    """One row per (delegation text, canonical case)."""
    out = []
    for t in delegation_texts():
        d = DOMAINS[t["domain"]]
        for case, clarity in CASES.items():
            if clarity != t["clarity"]:
                continue
            policy, state = authorization_config(t["domain"], case)
            out.append({**t, "case": case, "scenario_id": f"{t['text_id']}__{case}",
                        "context": d["context"], "private_context": d[clarity]["private_context"],
                        "intended": d[clarity]["intended"], "policy": policy, "state": state})
    return out


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} ({k / n:.0%}) [{lo:.2f}, {hi:.2f}]"
