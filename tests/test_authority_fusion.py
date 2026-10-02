from dualflow.authority import check_authority
from dualflow.fusion import fuse
from dualflow.models import AuthorityBudget, Interpretation


def test_authority_and_fusion():
    budget = AuthorityBudget(("summarize", "export"), "file", "/reports/")
    good = Interpretation("summarize", "file", "/reports/2026-09/")
    bad = Interpretation("delete", "file", "/reports/2026-09/")
    assert check_authority(good, budget).allowed
    assert not check_authority(bad, budget).allowed
    assert fuse(semantic_ok=True, authority_ok=True).execute
    assert not fuse(semantic_ok=True, authority_ok=False).execute


def test_scope_exceeded_is_rejected():
    budget = AuthorityBudget(("summarize",), "file", "/reports/2026-09/")
    out_of_scope = Interpretation("summarize", "file", "/reports/2026-08/")
    result = check_authority(out_of_scope, budget)
    assert not result.allowed
    assert result.reason == "scope_exceeded"


class TestMultiConditionAuthority:
    """2026-10 -- condition is a SET, not a single nullable string. ALL of
    a budget's required_conditions must appear in the proposal's own
    condition set (subset check); extra conditions the proposal claims
    beyond what's required don't matter."""

    def test_all_required_conditions_present_passes(self):
        budget = AuthorityBudget(("send_external",), "document", "audit/",
                                 frozenset({"recipient=approved_auditor", "human_approval=true"}))
        proposal = Interpretation("send_external", "document", "audit/q3_redacted_report.txt",
                                  frozenset({"recipient=approved_auditor", "human_approval=true"}))
        assert check_authority(proposal, budget).allowed

    def test_missing_one_of_two_required_conditions_fails(self):
        budget = AuthorityBudget(("send_external",), "document", "audit/",
                                 frozenset({"recipient=approved_auditor", "human_approval=true"}))
        proposal = Interpretation("send_external", "document", "audit/q3_redacted_report.txt",
                                  frozenset({"recipient=approved_auditor"}))  # missing human_approval
        result = check_authority(proposal, budget)
        assert not result.allowed
        assert result.reason.startswith("required_condition_missing")
        assert "human_approval=true" in result.reason

    def test_proposal_may_claim_extra_conditions_beyond_whats_required(self):
        budget = AuthorityBudget(("send_external",), "document", "audit/",
                                 frozenset({"recipient=approved_auditor"}))
        proposal = Interpretation("send_external", "document", "audit/q3_redacted_report.txt",
                                  frozenset({"recipient=approved_auditor", "human_approval=true"}))
        assert check_authority(proposal, budget).allowed

    def test_no_required_conditions_means_condition_set_is_irrelevant(self):
        budget = AuthorityBudget(("summarize",), "document", "audit/")  # no required_conditions
        proposal = Interpretation("summarize", "document", "audit/q3_raw_report.txt")
        assert check_authority(proposal, budget).allowed

    def test_semantic_match_is_sensitive_to_condition_set_not_just_action_resource_scope(self):
        """A mismatched condition set must count as a semantic mismatch --
        this is what lets Q3 in E1 distinguish 'understood correctly
        including the conditions' from 'got the action/resource/scope
        right but missed what conditions apply' (2026-10, per instruction:
        semantic and authority responsibilities must stay separate -- this
        test guards that condition mismatches are visible to the semantic
        axis too, not swallowed silently)."""
        from dualflow.semantic import exact_semantic_match

        truth = Interpretation("send_external", "document", "audit/q3_redacted_report.txt",
                               frozenset({"recipient=approved_auditor", "human_approval=true"}))
        proposal_missing_condition = Interpretation(
            "send_external", "document", "audit/q3_redacted_report.txt",
            frozenset({"recipient=approved_auditor"}))
        assert not exact_semantic_match(proposal_missing_condition, truth)
