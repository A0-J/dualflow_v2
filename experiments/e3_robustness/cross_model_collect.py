from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from dualflow.agents import PrincipalAgent
from dualflow.io import append_jsonl, read_jsonl, stable_hash
from dualflow.llm import OpenAILLMClient
from dualflow.models import Interpretation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate only Principal sample banks with another model; Delegate outputs stay frozen."
    )
    p.add_argument("--input", default="results/e2/raw/episodes.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--split", default="test")
    p.add_argument("--paraphrase-id", default="P2")
    p.add_argument("--max-episodes", type=int, default=24)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    source = read_jsonl(args.input)
    candidates = []
    seen_scenarios = set()
    for ep in source:
        if ep["split"] != args.split:
            continue
        # P2/P3 can be inferred from scenario_id suffix in the current manifest.
        if not ep["scenario_id"].endswith(args.paraphrase_id.lower()):
            continue
        candidates.append(ep)
        if len(candidates) >= args.max_episodes:
            break

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Output already exists: {out}")

    sdk = OpenAI()
    principal = PrincipalAgent(OpenAILLMClient(
        sdk, args.model, temperature=args.temperature, top_p=args.top_p
    ))

    calls = 0
    input_tokens = output_tokens = cached_tokens = 0
    for ep in candidates:
        samples = []
        for _ in range(args.samples):
            call = principal.restate_intent(
                delegation=ep["delegation"],
                private_goal=ep["principal_private_goal"],
                context=ep["context"],
            )
            samples.append({
                "interpretation": call.interpretation.to_dict(),
                "response": call.response.to_dict(),
            })
            calls += 1
            input_tokens += call.response.input_tokens or 0
            output_tokens += call.response.output_tokens or 0
            cached_tokens += call.response.cached_input_tokens or 0

        bank_id = stable_hash({
            "episode_id": ep["episode_id"],
            "principal_model": args.model,
            "samples": [s["interpretation"] for s in samples],
        })
        new_ep = dict(ep)
        new_ep["principal_samples"] = samples
        new_ep["sample_bank_id"] = bank_id
        new_ep["settings"] = dict(ep["settings"])
        new_ep["settings"]["principal_model"] = args.model
        new_ep["settings"]["n_samples"] = args.samples
        append_jsonl(out, new_ep)
        print(f'{ep["episode_id"]}: bank={bank_id}')

    print(f"calls={calls} input_tokens={input_tokens} output_tokens={output_tokens} cached_input_tokens={cached_tokens}")


if __name__ == "__main__":
    main()
