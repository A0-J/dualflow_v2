from __future__ import annotations

import json
import re

from .models import Interpretation


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
_DECODER = json.JSONDecoder()


def _first_json_object(text: str) -> dict:
    """Finds and decodes the first complete JSON object in `text`, ignoring
    any trailing content after it.

    2026-10 fix, two real failures in a row: a regex-based extractor
    (`\\{.*\\}` greedy, then `\\{.*?\\}` non-greedy) cannot correctly tell
    a `}` that closes the object from a `}` that happens to appear earlier
    (inside a string value, or in trailing commentary after the object) --
    greedy over-matched into trailing content (`Extra data`), non-greedy
    under-matched and cut the object short (`Expecting ',' delimiter`).
    `json.JSONDecoder.raw_decode()` does real brace/string-depth tracking
    via the stdlib's own scanner instead of guessing with a regex, so it
    is correct on both failure shapes and doesn't need fixing again for
    whatever the next irregular Delegate response looks like.
    """
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = _DECODER.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise ValueError(f"No JSON object found in model output: {text[:200]!r}")


def parse_interpretation(text: str) -> Interpretation:
    """Parse the first JSON object from model output and canonicalize it.
    Strips markdown code fences first (2026-10 fix -- real model output
    sometimes wraps JSON in ```json ... ``` despite instructions asking
    for no prose; `_first_json_object` would still find the object either
    way, but stripping fences first avoids depending on that)."""
    text = _FENCE_RE.sub("", text.strip()).strip()
    obj = _first_json_object(text)
    required = {"action", "resource", "scope", "condition"}
    missing = required.difference(obj)
    if missing:
        raise ValueError(f"Missing interpretation fields: {sorted(missing)}")
    return Interpretation.from_dict(obj)
