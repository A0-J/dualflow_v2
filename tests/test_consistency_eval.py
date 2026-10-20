"""Frozen parts of the d-contract consistency evaluation: the explicit stop without a reference, the
evaluation tasks, checker v2's parser and decision rule, and the mutation rules."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dualflow.effect_contracts import Effect, IntentContract
from dualflow.llm import LLMResponse
from dualflow.schema_free import (
    EXECUTOR_TOOLS, NO_REFERENCE, AgentText, PlannerOutputs, SchemaFreePlanner, ToolExecutor, authorize,
    contract_schema_problems, gate, run_episode, score, summary_of,
)

HERE = Path(__file__).resolve().parents[1] / "experiments" / "schema_free"
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"eval_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tasks_mod = _load("tasks")
cc = _load("check_consistency")
mc = _load("mutate_contracts")
DEV = {t.task_id: t for t in tasks_mod.DEV_TASKS}
MAIN = {t.task_id: t for t in tasks_mod.MAIN_TASKS}
NONE = LLMResponse("", "test", None, None, None, None, 0.0, None, None)


# --- explicit stop ---------------------------------------------------------------------------

def test_no_reference_never_executes_and_is_not_the_no_check_value():
    task = DEV["dev_audit_clear"]
    plan = frozenset(task.truth.required)
    assert gate(None, plan, task.policy, task.state).execute  # None: no semantic check (A, B)
    v = gate(NO_REFERENCE, plan, task.policy, task.state)
    assert not v.execute and v.reason == "no_reference"


def test_a_gated_episode_without_a_reference_stops_and_counts_a_valid_plan_as_a_false_rejection(tmp_path):
    task = DEV["dev_audit_clear"]
    plan = AgentText([], NONE)
    from dualflow.schema_free import ToolCall
    calls = [ToolCall("send_file", {"path": "audit/q3_redacted_report.txt", "recipient": "auditor"})]
    plan = AgentText(calls, NONE)
    outputs = PlannerOutputs(AgentText("d", NONE), AgentText(task.truth, NONE), (AgentText(task.truth, NONE),))
    plans = {"nl:planner": plan, "nl:terse": plan, "contract": plan}
    ep = run_episode("E", task, SchemaFreePlanner(None), outputs, "granular", ToolExecutor(None, EXECUTOR_TOOLS["granular"]),
                     plans, tmp_path / "e", gate_fn=lambda ref, d, p, s: gate(NO_REFERENCE, d, p, s))
    assert ep.stop_reason == "no_reference" and len(ep.rounds) == 1 and not ep.actual_effects
    assert score(ep, task).unnecessary_rejections == 1


def test_a_gated_condition_refuses_none_as_its_reference(tmp_path):
    task = DEV["dev_audit_clear"]
    outputs = PlannerOutputs(AgentText("d", NONE), AgentText(None, NONE), (AgentText(task.truth, NONE),))
    plan = AgentText([], NONE)
    with pytest.raises(ValueError, match="needs a reference"):
        run_episode("C", task, SchemaFreePlanner(None), outputs, "granular", ToolExecutor(None, EXECUTOR_TOOLS["granular"]),
                    {"nl:planner": plan, "nl:terse": plan, "contract": plan}, tmp_path / "c")


# --- tasks -----------------------------------------------------------------------------------

def test_dev_catalogs_and_policies_are_unchanged_by_output_paths():
    for t in DEV.values():
        assert "Other output paths" not in t.catalog()
        assert t.resources() == frozenset(t.files) | {summary_of(f) for f in t.files}
        assert t.policy.resource_prefixes[1:] == ("summaries/",)


def test_evaluation_tasks_are_complete_valid_and_separate_from_dev():
    assert len(MAIN) == 24 and all(t.split == "main" for t in MAIN.values())
    cells = {(tasks_mod.domain_of(t), t.kind) for t in MAIN.values()}
    assert cells == {(d, k) for d in tasks_mod.DOMAINS["main"] for k in tasks_mod.KINDS}
    dev_text = {s for t in DEV.values() for s in (t.user_request, t.private_context)}
    dev_names = {x for t in DEV.values() for x in (*t.files, *t.recipients)}
    for t in MAIN.values():
        assert contract_schema_problems(t.truth, t) == []
        assert authorize(t.truth.required | t.truth.permitted, t.policy, t.state).allowed
        assert not ({t.user_request, t.private_context} & dev_text)
        assert not ({*t.files, *t.recipients} & dev_names)
        for out in t.outputs:
            assert out in t.catalog()


# --- checker v2 ------------------------------------------------------------------------------

LINES = ["required: internal_write summaries/a.txt.summary.txt", "permitted: external_send a.txt bob"]


def _v2(effects, missing=()):
    return json.dumps({"effects": effects, "missing": list(missing)})


def _e(effect, supported="yes", forbidden="no"):
    return {"effect": effect, "supported": supported, "support_quote": "", "forbidden": forbidden, "forbid_quote": ""}


def test_checker_v2_parser_accepts_effects_with_or_without_labels_and_needs_every_judgement():
    ok = cc.parse_v2(_v2([_e("internal_write summaries/a.txt.summary.txt"), _e(LINES[1])]), LINES)
    assert cc.decide_v2(ok, LINES)
    with pytest.raises(ValueError):
        cc.parse_v2(_v2([_e(LINES[0])]), LINES)  # the permitted effect is not judged
    with pytest.raises(ValueError):
        cc.parse_v2(_v2([_e(LINES[0]), _e(LINES[1], forbidden="maybe")]), LINES)


def test_checker_v2_rule_holds_on_unsupported_unclear_forbidden_or_missing_effects():
    for bad in (_e(LINES[1], supported="no"), _e(LINES[1], supported="unclear"), _e(LINES[1], forbidden="yes")):
        assert not cc.decide_v2(cc.parse_v2(_v2([_e(LINES[0]), bad]), LINES), LINES)
    missing = [{"effect": "delete a.txt", "quote": "then delete it"}]
    assert not cc.decide_v2(cc.parse_v2(_v2([_e(LINES[0]), _e(LINES[1])], missing), LINES), LINES)
    ignored = [{"effect": "read a.txt", "quote": ""}, {"effect": LINES[1], "quote": ""}]  # a read; a listed effect
    assert cc.decide_v2(cc.parse_v2(_v2([_e(LINES[0]), _e(LINES[1])], ignored), LINES), LINES)


# --- mutations -------------------------------------------------------------------------------

def test_mutations_are_deterministic_schema_valid_and_differ_from_the_truth_unless_controls():
    for t in MAIN.values():
        ms = mc.mutations(t)
        assert ms == mc.mutations(t)
        kinds = [m["mutation"] for m in ms]
        assert "control_exact" in kinds and "control_reordered" in kinds and "omission" in kinds
        for m in ms:
            assert contract_schema_problems(m["contract"], t) == []
            if m["mutation"] in mc.CONTROLS:
                assert cc_goal(m["contract"]) == cc_goal(t.truth)
            else:
                assert cc_goal(m["contract"]) != cc_goal(t.truth)


def cc_goal(c: IntentContract):
    return (frozenset(e for e in c.required if e.kind != "read"),
            frozenset(e for e in c.required | c.permitted if e.kind in ("external_send", "delete")))


def test_attribute_mutation_includes_the_dev_miss_internal_write_on_the_source():
    ms = mc.mutations(MAIN["main_sales_ambiguous"])
    assert any(m["mutation"] == "attribute" and Effect("internal_write", "sales/pricing_model.xlsx") in m["contract"].required
               for m in ms)
