from __future__ import annotations

import json
import re

from .models import Interpretation


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)


def parse_interpretation(text: str) -> Interpretation:
    """Parse the first JSON object from model output and canonicalize it.
    Strips markdown code fences first (2026-10 fix -- real model output
    sometimes wraps JSON in ```json ... ``` despite instructions asking
    for no prose; the regex below would still find the object either way,
    but stripping fences first avoids depending on that)."""
    text = _FENCE_RE.sub("", text.strip()).strip()
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    obj = json.loads(match.group(0))
    required = {"action", "resource", "scope", "condition"}
    missing = required.difference(obj)
    if missing:
        raise ValueError(f"Missing interpretation fields: {sorted(missing)}")
    return Interpretation.from_dict(obj)
