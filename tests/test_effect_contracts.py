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


def test_irreversible_allowlist_blocks_unlisted_effects_even_when_goal_met():
    goal = Effect("internal_write", "hr/brief.txt")
    forbidden_full = Effect("external_send", "hr/compensation_full.csv", "benefits_vendor")
    contract = IntentContract(required=frozenset({goal}), forbidden=frozenset({forbidden_full}))
    cases = [
        frozenset({Effect("external_send", "hr/compensation_bands.txt", "benefits_vendor")}),
        frozenset({Effect("external_send", "hr/compensation_full.csv", "other_vendor")}),
        frozenset({goal, Effect("external_send", "hr/compensation_bands.txt", "benefits_vendor")}),
        frozenset({goal, Effect("delete", "hr/compensation_full.csv")}),
    ]
    for proposed in cases:
        decision = assess_effects(contract, proposed)
        assert not decision.safe
    assert not assess_effects(contract, cases[0]).goal_covered
    assert assess_effects(contract, cases[2]).goal_covered


def test_explicitly_permitted_irreversible_effect_and_goal():
    goal = Effect("internal_write", "hr/brief.txt")
    sent = Effect("external_send", "hr/compensation_bands.txt", "benefits_vendor")
    contract = IntentContract(required=frozenset({goal}), forbidden=frozenset(), permitted=frozenset({sent}))
    decision = assess_effects(contract, frozenset({goal, sent}))
    assert decision.safe and decision.goal_covered


def test_unknown_effect_or_noncanonical_resource_fails_closed():
    good = Effect("internal_write", "hr/brief.txt")
    contract = IntentContract(required=frozenset({good}), forbidden=frozenset())
    for bad in (Effect("unknown_side_effect", "hr/brief.txt"),
                Effect("delete", "hr/../secret.txt"),
                Effect("external_send", "hr/bands.txt", "../other")):
        assert not assess_effects(contract, frozenset({good, bad})).safe


def test_unknown_tool_is_explicit_denial():
    import pytest
    with pytest.raises(ValueError, match="execution denied"):
        tool_effects("executor_a", "unknown")
