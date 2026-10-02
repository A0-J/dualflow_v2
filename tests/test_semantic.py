from dualflow.models import Interpretation
from dualflow.semantic import entropy, grounded_verdict, repeated_anchor


def I(action, condition=frozenset()):
    return Interpretation(action, "file", "/x/", frozenset(condition))


def test_entropy_zero_when_all_equal():
    assert entropy(["a", "a", "a"]) == 0.0


def test_repeated_anchor_plurality():
    assert repeated_anchor([I("summarize"), I("export"), I("summarize")]).action == "summarize"


def test_grounded_only_rejects_confirmed_mismatch():
    samples = [I("summarize"), I("summarize"), I("export"), I("export")]
    v = grounded_verdict(I("read"), samples, threshold=0.5)
    assert v.facets["action"].confirmed is False
    assert v.passed is True


def test_grounded_verdict_facet_values_are_json_serializable():
    """Regression test (2026-10): grounded_verdict()'s per-facet mode for
    "condition" is a frozenset[str] (multi-condition model), not a plain
    str like the other three facets -- a smoke-test run hit this exact
    TypeError in evaluate.py before this was caught and fixed there."""
    import json
    from dualflow.models import Interpretation

    samples = [
        Interpretation("send_external", "document", "audit/q3_redacted_report.txt",
                       frozenset({"recipient=approved_auditor", "human_approval=true"})),
        Interpretation("send_external", "document", "audit/q3_redacted_report.txt",
                       frozenset({"recipient=approved_auditor", "human_approval=true"})),
    ]
    proposal = samples[0]
    v = grounded_verdict(proposal, samples, threshold=0.8)
    condition_mode = v.facets["condition"].value
    assert isinstance(condition_mode, frozenset)
    # The actual regression: evaluate.py's own _json_safe() must convert this
    # before json.dumps sees it -- this test documents the shape evaluate.py
    # has to handle, not evaluate.py itself.
    json.dumps(sorted(condition_mode))  # must not raise
