import importlib.util
from pathlib import Path

from dualflow.flows import Proposal
from dualflow.roles import ExecutorAgent

_path = Path(__file__).resolve().parents[1] / "experiments" / "clarification_option" / "analyze.py"
_spec = importlib.util.spec_from_file_location("clarification_option_analyze", _path)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

SUMMARIZE = Proposal("summarize", "audit/q3_raw_report.txt", "internal")
SEND = Proposal("send_external", "audit/q3_redacted_report.txt", "external:auditor")


def ask(unclear=("action",)):
    return {"decision": "ask", "proposal": None, "unclear": list(unclear)}


def execute(p):
    return {"decision": "execute", "proposal": p.to_dict(), "unclear": []}


def test_majority_decides_and_modal_is_over_execute_samples():
    row = analyze.score_pair([ask(), ask(), ask(), execute(SEND), execute(SEND)], SUMMARIZE)
    assert row["category"] == "asked" and row["ask_rate"] == 0.6
    assert (row["samples_executed_intended"], row["samples_executed_mismatch"]) == (0, 2)
    row = analyze.score_pair([ask(), ask(), execute(SEND), execute(SEND), execute(SUMMARIZE)], SUMMARIZE)
    assert row["category"] == "executed_mismatch" and row["irreversible_misinterpretation"]
    row = analyze.score_pair([execute(SUMMARIZE)] * 4 + [ask(("scope",))], SUMMARIZE)
    assert row["category"] == "executed_intended"


def test_facet_scoring_against_design_labels():
    assert analyze.facet_label("legal", "ambiguous") == {"action"}
    assert analyze.facet_label("audit", "ambiguous") == {"action", "resource", "scope"}
    assert analyze.facet_label("audit", "clear") == frozenset()
    turns = [ask(("action",)), ask(("action", "scope")), ask(())]
    row = analyze.score_pair(turns, SUMMARIZE, frozenset({"action"}))
    assert row["asks_exact_facets"] == 1
    assert (row["action_tp"], row["action_fp"], row["action_fn"]) == (2, 0, 1)
    assert (row["scope_tp"], row["scope_fp"], row["scope_fn"]) == (0, 1, 0)


def test_option_prompts_keep_the_canonical_sentences():
    shared = ("You are the executor agent. Interpret the delegation you received into one concrete operation to perform.",
              "one action, one resource path, and one scope label from the runtime context. "
              "Choose the operation the delegation asks for. Do not invent hidden intent.")
    for prompt in (ExecutorAgent.INSTRUCTIONS, ExecutorAgent.ASK_INSTRUCTIONS, ExecutorAgent.ASK_INSTRUCTIONS_EXPLICIT):
        flat = prompt.replace("\n", " ")
        assert all(sentence in flat for sentence in shared)


def test_pinned_model_ids_are_the_models_that_served_the_canonical_collection():
    import json
    _cpath = Path(__file__).resolve().parents[1] / "experiments" / "clarification_option" / "collect.py"
    _cspec = importlib.util.spec_from_file_location("clarification_option_collect", _cpath)
    collect = importlib.util.module_from_spec(_cspec)
    _cspec.loader.exec_module(collect)
    served = {}
    raw = Path(__file__).resolve().parents[1] / "results" / "canonical" / "raw" / "samples.jsonl"
    for line in raw.open(encoding="utf-8"):
        r = json.loads(line)
        if r["role"] == "executor":
            served.setdefault(r["model"], set()).add(r["response"]["served_model"])
    assert {m: {i} for m, i in collect.MODEL_IDS.items()} == served
