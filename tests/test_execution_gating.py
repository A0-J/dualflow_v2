"""Proves the execution-gating invariant directly (2026-10, exfiltration
scenario): `execute_action()` must run (and produce a real file) when the
fused decision is EXECUTE, and must never be called at all -- not called-
and-no-op, never invoked -- when the fused decision is REJECT. Both the
positive and negative case are exercised here; experiments/
e1_independent_verification/run.py's 4 narrative cases only exercise the
negative case (none of them EXECUTE a send_external), so this file is
where the positive case actually gets proven.
"""

from __future__ import annotations

from pathlib import Path

from dualflow.authority import check_authority
from dualflow.execution import build_sandbox, execute_action
from dualflow.fusion import fuse
from dualflow.models import AuthorityBudget, Interpretation
from dualflow.semantic import exact_semantic_match


def _run_gated(proposal: Interpretation, truth: Interpretation, budget: AuthorityBudget,
              workspace_root: Path):
    """Mirrors exactly how every real caller in this pack is required to
    gate the executor -- the thing under test is this control flow
    itself, not a mocked stand-in for it."""
    semantic_ok = exact_semantic_match(proposal, truth)
    authority = check_authority(proposal, budget)
    decision = fuse(semantic_ok=semantic_ok, authority_ok=authority.allowed)
    executed = False
    if decision.execute:
        execute_action(action=proposal.action, resource_path=proposal.scope,
                       workspace_root=workspace_root)
        executed = True
    return decision, executed


class TestExecutionGating:
    def test_execute_decision_actually_copies_the_file(self, tmp_path):
        build_sandbox(tmp_path)
        proposal = Interpretation("send_external", "document", "audit/q3_redacted_report.txt",
                                  frozenset({"recipient=approved_auditor", "human_approval=true"}))
        truth = proposal  # Principal truly intended exactly this
        budget = AuthorityBudget(("send_external",), "document", "audit/",
                                 frozenset({"recipient=approved_auditor", "human_approval=true"}))

        decision, executed = _run_gated(proposal, truth, budget, tmp_path)

        assert decision.execute is True
        assert executed is True
        outbox = list((tmp_path / "external_outbox").iterdir())
        assert [p.name for p in outbox] == ["q3_redacted_report.txt"]

    def test_reject_decision_never_calls_the_executor_semantic_fail(self, tmp_path):
        build_sandbox(tmp_path)
        truth = Interpretation("summarize", "document", "audit/q3_raw_report.txt")
        proposal = Interpretation("send_external", "document", "audit/q3_redacted_report.txt")
        budget = AuthorityBudget(("read", "summarize", "send_external"), "document", "audit/")

        decision, executed = _run_gated(proposal, truth, budget, tmp_path)

        assert decision.execute is False
        assert executed is False
        assert list((tmp_path / "external_outbox").iterdir()) == []

    def test_reject_decision_never_calls_the_executor_authority_fail(self, tmp_path):
        build_sandbox(tmp_path)
        proposal = Interpretation("send_external", "document", "audit/q3_redacted_report.txt",
                                  frozenset({"recipient=approved_auditor", "human_approval=true"}))
        truth = proposal
        budget = AuthorityBudget(("read", "summarize"), "document", "audit/")  # no send_external grant

        decision, executed = _run_gated(proposal, truth, budget, tmp_path)

        assert decision.execute is False
        assert executed is False
        assert list((tmp_path / "external_outbox").iterdir()) == []

    def test_read_only_actions_never_touch_the_filesystem_even_directly(self, tmp_path):
        build_sandbox(tmp_path)
        result = execute_action(action="summarize", resource_path="audit/q3_raw_report.txt",
                                workspace_root=tmp_path)
        assert result.side_effect is False
        assert list((tmp_path / "external_outbox").iterdir()) == []

    def test_send_external_on_missing_resource_raises_rather_than_silently_no_op(self, tmp_path):
        build_sandbox(tmp_path)
        try:
            execute_action(action="send_external", resource_path="audit/does_not_exist.txt",
                           workspace_root=tmp_path)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError for a missing sandbox resource")
