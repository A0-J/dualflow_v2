"""Build the human ground-truth validation kit for the 80 delegation texts
of the canonical C1-C4 evaluation (see PREREGISTRATION.md). 0 API calls.

Outputs
  kit/annotation_tool.html  -- the only file given to annotators (self-
                               contained; items carry opaque ids only)
  kit/INSTRUCTIONS_ko.md     -- annotator instructions
  ../../results/human_validation/key.json
                            -- opaque id -> delegation text, group, intended
                               task. NEVER given to annotators.

Every one of the 80 texts is annotated; none is selected or dropped.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "canonical"))

from common import delegation_texts  # noqa: E402
from scenarios import DOMAINS  # noqa: E402

KIT = HERE / "kit"
KEY = ROOT / "results" / "human_validation" / "key.json"


def _opaque(prefix: str, text_id: str) -> str:
    # Phase-specific salt: an annotator cannot match a Phase A item to its Phase B item by id.
    return f"{prefix}-{hashlib.sha256(f'dualflow-hv:{prefix}:{text_id}'.encode()).hexdigest()[:6]}"


def _files(domain: str) -> list[tuple[str, str]]:
    ctx = DOMAINS[domain]["context"]
    files = []
    for line in ctx.split("Files (resource must be one of these paths):\n")[1].split("\n\n")[0].splitlines():
        path, desc = line[2:].split(": ", 1)
        files.append((path, desc))
    return files


def _recipient(domain: str) -> tuple[str, str]:
    ctx = DOMAINS[domain]["context"]
    for line in ctx.splitlines():
        if line.startswith("- external:"):
            label, desc = line[2:].split(": ", 1)
            return label, desc
    raise ValueError(domain)


def options(domain: str) -> list[dict]:
    """Every (action, resource, scope) task an Executor Agent could perform here."""
    recipient, _ = _recipient(domain)
    out = []
    for path, _ in _files(domain):
        for action in ("read", "summarize", "delete"):
            out.append({"key": f"{action}|{path}|internal", "action": action, "resource": path, "scope": "internal"})
        out.append({"key": f"send_external|{path}|{recipient}", "action": "send_external",
                    "resource": path, "scope": recipient})
    return out


def environment(domain: str) -> dict:
    recipient, rdesc = _recipient(domain)
    return {"files": [{"path": p, "desc": d} for p, d in _files(domain)],
            "recipient": recipient, "recipient_desc": rdesc}


def main() -> None:
    texts = delegation_texts()
    assert len(texts) == 80, len(texts)
    originals = {t["group"]: t["delegation"] for t in texts if t["original"]}
    phase_a, phase_b, key = [], [], {}
    for t in texts:
        d = DOMAINS[t["domain"]]
        aid, bid = _opaque("A", t["text_id"]), _opaque("B", t["text_id"])
        env = environment(t["domain"])
        opts = options(t["domain"])
        phase_a.append({"id": aid, "env": env, "delegation": t["delegation"], "options": opts})
        phase_b.append({"id": bid, "env": env, "task_context": d[t["clarity"]]["private_context"],
                        "original": originals[t["group"]], "variant": t["delegation"],
                        "is_original": t["original"], "options": opts})
        intended = d[t["clarity"]]["intended"]
        mis = d["ambiguous"].get("misreading") if t["clarity"] == "ambiguous" else None
        key[t["text_id"]] = {"a_id": aid, "b_id": bid, "group": t["group"], "domain": t["domain"],
                             "clarity": t["clarity"], "original": t["original"], "delegation": t["delegation"],
                             "intended_key": f"{intended['action']}|{intended['resource']}|{intended['scope']}",
                             "misinterpretation_key": (f"{mis['action']}|{mis['resource']}|{mis['scope']}"
                                                       if mis else None)}
    assert len({k["a_id"] for k in key.values()}) == 80 and len({k["b_id"] for k in key.values()}) == 80

    KIT.mkdir(parents=True, exist_ok=True)
    template = (HERE / "tool_template.html").read_text(encoding="utf-8")
    payload = json.dumps({"phase_a": phase_a, "phase_b": phase_b}, ensure_ascii=False)
    (KIT / "annotation_tool.html").write_text(template.replace("/*__ITEMS__*/null", payload), encoding="utf-8")
    KEY.parent.mkdir(parents=True, exist_ok=True)
    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {KIT / 'annotation_tool.html'} ({len(phase_a)} + {len(phase_b)} items) and {KEY}")


if __name__ == "__main__":
    main()
