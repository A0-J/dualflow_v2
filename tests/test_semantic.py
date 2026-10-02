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
