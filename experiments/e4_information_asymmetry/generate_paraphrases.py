"""E4 step 1: generate delegation paraphrases automatically -- no hand
selection. Seeds are the two dev-split (P1) delegations. Every paraphrase
the generator returns is kept, in returned order, except exact duplicates
(case/whitespace-insensitive) and copies of any original delegation; the
first PARAPHRASES_PER_INTENT survivors are used (PREREGISTRATION.md §3).
"""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

from dualflow.io import read_jsonl, write_jsonl
from dualflow.parsing import _FENCE_RE

from common import (
    DEFAULT_SAMPLING_MODELS, GENERATED_PREFIX, ORIGINAL_SCENARIOS,
    PARAPHRASE_GENERATOR_MODEL, PARAPHRASES, PARAPHRASES_PER_INTENT,
    SEED_PARAPHRASE_ID, TEMPERATURE, TOP_P,
)

PROMPT = """Here is a short instruction that someone gave to an assistant:

"{seed}"

Write {k} different paraphrases of this instruction. Each paraphrase must
keep exactly the same meaning and the same level of detail: do not add or
remove information, and do not make the requested action more or less
explicit than it is in the original. Vary the vocabulary and sentence
structure. Return only a JSON array of {k} strings, with no other text."""

REQUEST_K = 30


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if PARAPHRASES.exists() and not args.overwrite:
        raise SystemExit(f"{PARAPHRASES} exists; use --overwrite intentionally.")

    originals = read_jsonl(ORIGINAL_SCENARIOS)
    seen = {_norm(s["delegation"]) for s in originals}
    seeds = {s["principal_intent"]["action"]: s for s in originals if s["paraphrase_id"] == SEED_PARAPHRASE_ID}
    sampling = {} if PARAPHRASE_GENERATOR_MODEL in DEFAULT_SAMPLING_MODELS else {"temperature": TEMPERATURE, "top_p": TOP_P}
    client = OpenAI(max_retries=8)

    rows = []
    for intent in ("summarize", "send_external"):
        seed = seeds[intent]
        kept: list[str] = []
        calls = []
        while len(kept) < PARAPHRASES_PER_INTENT:
            if len(calls) >= 5:
                raise SystemExit(f"{intent}: only {len(kept)} unique paraphrases after 5 calls")
            r = client.responses.create(
                model=PARAPHRASE_GENERATOR_MODEL,
                input=PROMPT.format(seed=seed["delegation"], k=REQUEST_K),
                **sampling,
            )
            calls.append(r.output_text)
            for text in json.loads(_FENCE_RE.sub("", r.output_text.strip()).strip()):
                if isinstance(text, str) and text.strip() and _norm(text) not in seen:
                    seen.add(_norm(text))
                    kept.append(text.strip())
        for i, text in enumerate(kept[:PARAPHRASES_PER_INTENT], 1):
            rows.append({
                "scenario_id": f"{GENERATED_PREFIX}_{intent}_{i:02d}",
                "intent": intent,
                "paraphrase_id": f"G{i:02d}",
                "delegation": text,
                "seed_scenario_id": seed["scenario_id"],
                "generator_model": PARAPHRASE_GENERATOR_MODEL,
                "generator_calls": len(calls),
            })
        print(f"{intent}: kept {min(len(kept), PARAPHRASES_PER_INTENT)} paraphrases from {len(calls)} call(s)")

    write_jsonl(PARAPHRASES, rows)
    print(f"wrote {len(rows)} -> {PARAPHRASES}")


if __name__ == "__main__":
    main()
