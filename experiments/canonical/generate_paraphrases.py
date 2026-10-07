"""Generate paraphrases of each of the 8 delegations automatically. Every
returned paraphrase is kept in returned order except exact duplicates
(case/whitespace-insensitive) and copies of an original delegation; the
first PARAPHRASES_PER_DELEGATION are used. No hand selection.
"""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

from dualflow.io import write_jsonl
from dualflow.parsing import _FENCE_RE

from common import PARAPHRASE_GENERATOR_MODEL, PARAPHRASES, PARAPHRASES_PER_DELEGATION
from scenarios import DOMAINS

PROMPT = """Here is a short instruction that someone gave to an assistant:

"{seed}"

Write {k} different paraphrases of this instruction. Each paraphrase must
keep exactly the same meaning and the same level of detail: do not add or
remove information, and do not make the requested action more or less
explicit than it is in the original. Vary the vocabulary and sentence
structure. Return only a JSON array of {k} strings, with no other text."""

REQUEST_K = 15


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

    client = OpenAI(max_retries=8)
    seen = {_norm(d[c]["delegation"]) for d in DOMAINS.values() for c in ("clear", "ambiguous")}
    rows = []
    for domain, d in DOMAINS.items():
        for clarity in ("clear", "ambiguous"):
            seed = d[clarity]["delegation"]
            kept, calls = [], 0
            while len(kept) < PARAPHRASES_PER_DELEGATION:
                if calls >= 5:
                    raise SystemExit(f"{domain}/{clarity}: only {len(kept)} unique paraphrases")
                r = client.responses.create(model=PARAPHRASE_GENERATOR_MODEL,
                                            input=PROMPT.format(seed=seed, k=REQUEST_K),
                                            temperature=1.0, top_p=1.0)
                calls += 1
                for text in json.loads(_FENCE_RE.sub("", r.output_text.strip()).strip()):
                    if isinstance(text, str) and text.strip() and _norm(text) not in seen:
                        seen.add(_norm(text))
                        kept.append(text.strip())
            group = f"{domain}_{clarity}"
            for i, text in enumerate(kept[:PARAPHRASES_PER_DELEGATION], 1):
                rows.append({"text_id": f"{group}_{i:02d}", "group": group, "domain": domain,
                             "clarity": clarity, "delegation": text,
                             "generator_model": PARAPHRASE_GENERATOR_MODEL})
            print(f"{group}: kept {PARAPHRASES_PER_DELEGATION} from {calls} call(s)")
    write_jsonl(PARAPHRASES, rows)
    print(f"wrote {len(rows)} -> {PARAPHRASES}")


if __name__ == "__main__":
    main()
