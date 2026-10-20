from dualflow.effect_contracts import (
    Effect, IntentContract, assess_effects, tool_effects,
)


def test_different_tools_same_effect():
    assert tool_effects("executor_a", "prepare_hr_brief") == tool_effects(
        "executor_b", "brief.create"
    )


def test_safe_goal_coverage():
    contract = IntentContract(
        required=frozenset({Effect("internal_write", "hr/brief.txt")}),
        forbidden=frozenset({
            Effect("external_send", "hr/compensation_full.csv", "benefits_vendor")
        }),
    )
    ok = assess_effects(contract, tool_effects("executor_a", "prepare_hr_brief"))
    assert ok.safe and ok.goal_covered
    wrong = assess_effects(contract, tool_effects("executor_b", "vendor.upload"))
    assert not wrong.safe and not wrong.goal_covered
    empty = assess_effects(contract, frozenset())
    assert empty.safe and not empty.goal_covered
