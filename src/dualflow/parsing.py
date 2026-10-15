"""JSON extraction from model output."""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
_DECODER = json.JSONDecoder()


def strip_fences(text: str) -> str:
    """Remove markdown code fences that models add despite instructions."""
    return _FENCE_RE.sub("", text.strip()).strip()


def first_json_object(text: str) -> dict:
    """Decode the first complete JSON object in `text`, ignoring anything after it.

    `raw_decode` tracks string and brace depth, so a `}` inside a string value
    or in trailing commentary cannot end the object early or extend it.
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


def parse_json_object(text: str) -> dict:
    return first_json_object(strip_fences(text))
