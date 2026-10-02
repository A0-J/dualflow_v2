from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from dualflow.agents import DelegateAgent, PrincipalAgent
from dualflow.io import append_jsonl, read_jsonl, stable_hash
from dualflow.llm import OpenAILLMClient
from dualflow.models import AuthorityBudget, Interpretation


def usage_dict(call) -> dict:
    return {
        "interpretation": call.interpretation.to_dict(),
        "response": call.response.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect one shared sample bank per episode.")
    p.add_argument("--scenarios", default="experiments/e2_semantic_verification/scenarios.jsonl")
    p.add_argument("--output", default="results/e2/raw/episodes.jsonl")
    p.add_argument("--delegate-model", default="gpt-4o-mini")
    p.add_argument("--principal-model", default="gpt-4o-mini")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--episodes-per-scenario", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
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

    scenarios = read_jsonl(args.scenarios)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    sdk = OpenAI()
    delegate = DelegateAgent(OpenAILLMClient(
        sdk, args.delegate_model, temperature=args.temperature, top_p=args.top_p
    ))
    principal = PrincipalAgent(OpenAILLMClient(
        sdk, args.principal_model, temperature=args.temperature, top_p=args.top_p
    ))

    total_calls = 0
    total_input = 0
    total_output = 0
    total_cached = 0

    for scenario in scenarios:
        private_intent = Interpretation.from_dict(scenario["principal_intent"])
        budget = AuthorityBudget.from_dict(scenario["authority_budget"])
        for rep in range(args.episodes_per_scenario):
            episode_id = f'{scenario["scenario_id"]}__r{rep:02d}'
            delegate_call = delegate.propose(
                delegation=scenario["delegation"],
                context=scenario["context"],
            )
            total_calls += 1

            principal_calls = []
            for i in range(args.samples):
                call = principal.restate_intent(
                    delegation=scenario["delegation"],
                    private_goal=scenario["principal_private_goal"],
                    context=scenario["context"],
                )
                principal_calls.append(call)
                total_calls += 1

            for call in [delegate_call, *principal_calls]:
                r = call.response
                total_input += r.input_tokens or 0
                total_output += r.output_tokens or 0
                total_cached += r.cached_input_tokens or 0

            sample_bank_id = stable_hash({
                "episode_id": episode_id,
                "samples": [c.interpretation.to_dict() for c in principal_calls],
            })
            record = {
                "experiment": "E2",
                "episode_id": episode_id,
                "scenario_id": scenario["scenario_id"],
                "split": scenario["split"],
                "domain": scenario["domain"],
                "delegation": scenario["delegation"],
                "context": scenario["context"],
                "principal_private_goal": scenario["principal_private_goal"],
                "principal_intent": private_intent.to_dict(),
                "authority_budget": budget.to_dict(),
                "delegate": usage_dict(delegate_call),
                "sample_bank_id": sample_bank_id,
                "principal_samples": [usage_dict(c) for c in principal_calls],
                "settings": {
                    "delegate_model": args.delegate_model,
                    "principal_model": args.principal_model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "n_samples": args.samples,
                },
            }
            append_jsonl(out, record)
            print(
                f"[{episode_id}] bank={sample_bank_id} "
                f"calls={1 + len(principal_calls)} cumulative={total_calls}"
            )

    print("\n=== Actual API usage reported by collected responses ===")
    print(f"calls={total_calls}")
    print(f"input_tokens={total_input}")
    print(f"output_tokens={total_output}")
    print(f"cached_input_tokens={total_cached}")


if __name__ == "__main__":
    main()
