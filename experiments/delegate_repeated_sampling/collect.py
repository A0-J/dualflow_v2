"""Delegate Repeated-Sampling Analysis -- collection step.

Separate, supplementary artifact to E2 (per the user's explicit point #9:
own directory, own results, never merged into or overwriting E2's numbers).
E2 only ever sampled the Delegate ONCE per episode and repeated the
Principal 20x -- so E2's own data cannot show whether the Delegate's
semantic uncertainty is itself low or high. This script closes exactly
that gap: it samples the Delegate 20x independently per base scenario,
using the identical model/temperature/top_p as E2, and changes nothing
else (no E2 code, prompt, threshold, or result file is touched).

Principal-side evidence is NOT re-collected -- the existing 20-sample
Principal bank already collected for each scenario's `__r00` episode in
results/e2/raw/episodes.jsonl is reused verbatim (0 new Principal calls),
per the user's instruction #3.

Budget: 6 base scenarios x 20 Delegate samples = 120 new API calls, 0 new
Principal calls.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from dualflow.agents import DelegateAgent
from dualflow.io import append_jsonl, read_jsonl, stable_hash
from dualflow.llm import OpenAILLMClient

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

# Reused verbatim from E2's r00 episodes -- the existing 20-sample
# Principal bank for each base scenario (see pre-execution table shown to
# the user before this script was written).
EXISTING_PRINCIPAL_EPISODE = {
    "exfil_summarize_p1": "exfil_summarize_p1__r00",
    "exfil_summarize_p2": "exfil_summarize_p2__r00",
    "exfil_summarize_p3": "exfil_summarize_p3__r00",
    "exfil_send_external_p1": "exfil_send_external_p1__r00",
    "exfil_send_external_p2": "exfil_send_external_p2__r00",
    "exfil_send_external_p3": "exfil_send_external_p3__r00",
}


def usage_dict(call) -> dict:
    return {
        "interpretation": call.interpretation.to_dict(),
        "response": call.response.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delegate-only repeated sampling, 20x per base scenario.")
    p.add_argument("--scenarios", default=str(_ROOT / "experiments" / "e2_semantic_verification" / "scenarios.jsonl"))
    p.add_argument("--principal-episodes", default=str(_ROOT / "results" / "e2" / "raw" / "episodes.jsonl"))
    p.add_argument("--output", default=str(_ROOT / "results" / "delegate_repeated_sampling" / "raw" / "episodes.jsonl"))
    p.add_argument("--delegate-model", default="gpt-4o-mini")
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

    scenarios = {s["scenario_id"]: s for s in read_jsonl(args.scenarios)}
    principal_episodes = {e["episode_id"]: e for e in read_jsonl(args.principal_episodes)}

    sdk = OpenAI()
    delegate = DelegateAgent(OpenAILLMClient(
        sdk, args.delegate_model, temperature=args.temperature, top_p=args.top_p
    ))

    total_calls = 0
    total_input = 0
    total_output = 0
    total_cached = 0

    for scenario_id, principal_episode_id in EXISTING_PRINCIPAL_EPISODE.items():
        scenario = scenarios[scenario_id]
        principal_episode = principal_episodes[principal_episode_id]
        if len(principal_episode["principal_samples"]) < args.samples:
            raise ValueError(
                f"{principal_episode_id}: need {args.samples} existing Principal samples, "
                f"only {len(principal_episode['principal_samples'])} available"
            )

        delegate_calls = []
        for _ in range(args.samples):
            call = delegate.propose(delegation=scenario["delegation"], context=scenario["context"])
            delegate_calls.append(call)
            total_calls += 1

        for call in delegate_calls:
            r = call.response
            total_input += r.input_tokens or 0
            total_output += r.output_tokens or 0
            total_cached += r.cached_input_tokens or 0

        delegate_bank_id = stable_hash({
            "scenario_id": scenario_id,
            "samples": [c.interpretation.to_dict() for c in delegate_calls],
        })

        record = {
            "experiment": "delegate_repeated_sampling",
            "scenario_id": scenario_id,
            "split": scenario["split"],
            "domain": scenario["domain"],
            "delegation": scenario["delegation"],
            "context": scenario["context"],
            "principal_intent": scenario["principal_intent"],
            "authority_budget": scenario["authority_budget"],
            "delegate_bank_id": delegate_bank_id,
            "delegate_samples": [usage_dict(c) for c in delegate_calls],
            "principal_episode_id": principal_episode_id,
            "principal_sample_bank_id": principal_episode["sample_bank_id"],
            "principal_samples": principal_episode["principal_samples"],
            "settings": {
                "delegate_model": args.delegate_model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "n_samples": args.samples,
            },
        }
        append_jsonl(out, record)
        print(f"[{scenario_id}] delegate_bank={delegate_bank_id} calls={len(delegate_calls)} cumulative={total_calls}")

    print("\n=== Actual API usage reported by collected responses ===")
    print(f"calls={total_calls}")
    print(f"input_tokens={total_input}")
    print(f"output_tokens={total_output}")
    print(f"cached_input_tokens={total_cached}")


if __name__ == "__main__":
    main()
