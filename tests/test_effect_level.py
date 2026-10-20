import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "experiments" / "effect_level_verification" / "rescore.py"
_spec = importlib.util.spec_from_file_location("effect_rescore", _path)
rescore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore)

SUMMARIZE = {"action": "summarize", "resource": "audit/q3_raw_report.txt", "scope": "internal"}
SUMMARIZE_OTHER = {**SUMMARIZE, "resource": "audit/q3_redacted_report.txt"}
READ = {"action": "read", "resource": "audit/q3_raw_report.txt", "scope": "internal"}
SEND = {"action": "send_external", "resource": "audit/q3_redacted_report.txt", "scope": "external:auditor"}
SEND_RAW = {**SEND, "resource": "audit/q3_raw_report.txt"}


def test_safety_only_rules_let_reversible_mismatches_through():
    assert not rescore.v_sem("exact", READ, SUMMARIZE)
    assert rescore.v_sem("safety, coarse", READ, SUMMARIZE)
    assert rescore.v_sem("safety, resource-aware", READ, SUMMARIZE)
    assert not rescore.v_sem("safety, resource-aware", SEND, SUMMARIZE)    # unintended external send
    assert not rescore.v_sem("safety, resource-aware", SEND_RAW, SEND)     # external send of another file
    assert rescore.v_sem("safety, coarse", SEND_RAW, SEND)                 # ...missed by kind alone


def test_goal_check_requires_the_intended_effect():
    assert not rescore.v_sem("safety+goal, coarse", READ, SUMMARIZE)       # no summary written
    assert rescore.v_sem("safety+goal, coarse", SUMMARIZE_OTHER, SUMMARIZE)
    assert not rescore.v_sem("safety+goal, resource-aware", SUMMARIZE_OTHER, SUMMARIZE)
    assert rescore.v_sem("safety+goal, resource-aware", SUMMARIZE, SUMMARIZE)
    assert not rescore.v_sem("safety+goal, coarse", READ, SEND)             # the intended external send never happens
