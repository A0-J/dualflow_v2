"""Collects A3's 20 Delegate + 20 Principal samples -- 40 new real API
calls, the only new sampling needed for the authority-risk subset (A1/A2
reuse already-collected real `exfil_send_external_p1` output under a
different hypothetical budget; see `five_arm_authority_risk.py`).

Same model/temperature/top_p as E2 and the Delegate Repeated-Sampling
Analysis (gpt-4o-mini, T=1.0, top_p=1.0) -- no new prompt wording beyond
scenario_a3.py's context, no threshold changes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from dualflow.agents import DelegateAgent, PrincipalAgent
from dualflow.io import append_jsonl, stable_hash
from dualflow.llm import OpenAILLMClient

from scenario_a3 import SCENARIO_A3

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent


def usage_dict(call) -> dict:
    return {"interpretation": call.interpretation.to_dict(), "response": call.response.to_dict()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect A3 (approval_missing) -- 40 real calls.")
    p.add_argument("--output", default=str(_ROOT / "results" / "delegate_repeated_sampling" / "authority_risk" / "raw" / "a3_episode.jsonl"))
    p.add_argument("--delegate-model", default="gpt-4o-mini")
    p.add_argument("--principal-model", default="gpt-4o-mini")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    out = Path(args.output)
    if out.exists() and args.overwrite:
        out.unlink()
    elif out.exists():
        raise SystemExit(f"Output already exists: {out}. Use --overwrite intentionally.")

    scenario = SCENARIO_A3
    sdk = OpenAI()
    delegate = DelegateAgent(OpenAILLMClient(sdk, args.delegate_model, temperature=args.temperature, top_p=args.top_p))
    principal = PrincipalAgent(OpenAILLMClient(sdk, args.principal_model, temperature=args.temperature, top_p=args.top_p))

    delegate_calls = []
    for _ in range(args.samples):
        delegate_calls.append(delegate.propose(delegation=scenario["delegation"], context=scenario["context"]))

    principal_calls = []
    for _ in range(args.samples):
        principal_calls.append(principal.restate_intent(
            delegation=scenario["delegation"],
            private_goal=scenario["principal_private_goal"],
            context=scenario["context"],
        ))

    total_calls = len(delegate_calls) + len(principal_calls)
    total_input = sum((c.response.input_tokens or 0) for c in [*delegate_calls, *principal_calls])
    total_output = sum((c.response.output_tokens or 0) for c in [*delegate_calls, *principal_calls])

    record = {
        "experiment": "authority_risk_A3",
        "scenario_id": scenario["scenario_id"],
        "split": scenario["split"],
        "delegation": scenario["delegation"],
        "context": scenario["context"],
        "principal_private_goal": scenario["principal_private_goal"],
        "principal_intent": scenario["principal_intent"],
        "authority_budget": scenario["authority_budget"],
        "delegate_bank_id": stable_hash({"scenario_id": scenario["scenario_id"], "samples": [c.interpretation.to_dict() for c in delegate_calls]}),
        "delegate_samples": [usage_dict(c) for c in delegate_calls],
        "principal_bank_id": stable_hash({"scenario_id": scenario["scenario_id"], "samples": [c.interpretation.to_dict() for c in principal_calls]}),
        "principal_samples": [usage_dict(c) for c in principal_calls],
        "settings": {
            "delegate_model": args.delegate_model, "principal_model": args.principal_model,
            "temperature": args.temperature, "top_p": args.top_p, "n_samples": args.samples,
        },
    }
    append_jsonl(out, record)

    print(f"[{scenario['scenario_id']}] calls={total_calls}")
    print(f"input_tokens={total_input} output_tokens={total_output}")


if __name__ == "__main__":
    main()
