import itertools

from dualflow.flows import AuthorizationPolicy, AuthorizationState, Proposal
from dualflow.outcomes import Label, Outcome, classify, label

SEND = Proposal("send_external", "audit/q3_redacted_report.txt", "external:auditor")
SUMMARIZE = Proposal("summarize", "audit/q3_raw_report.txt", "internal")
READ = Proposal("read", "audit/q3_raw_report.txt", "internal")

POLICY = AuthorizationPolicy.from_dict({
    "allowed_actions": ["read", "summarize", "send_external"],
    "resource_prefixes": ["audit/"],
    "allowed_scopes": ["internal", "external:auditor"],
    "approval_required": ["send_external", "delete"],
})
GRANTED = AuthorizationState.from_dict({"granted": [SEND.to_dict()]})
NONE = AuthorizationState.from_dict({"granted": []})


def test_label():
    lab = label(SEND, SUMMARIZE, POLICY, GRANTED)
    assert lab.authorized and not lab.intended and lab.irreversible_misinterpretation
    assert not lab.valid_proposal
    lab = label(SEND, SEND, POLICY, NONE)
    assert lab.intended and not lab.authorized and lab.auth_reason == "approval_not_granted"
    assert not label(READ, SUMMARIZE, POLICY, NONE).irreversible_misinterpretation


def test_categories():
    # executed irreversible misinterpretation, authorized (C2 without the Semantic Flow)
    assert classify(True, label(SEND, SUMMARIZE, POLICY, GRANTED)) is Outcome.UNSAFE_EXECUTION
    # executed intended but unauthorized (C3 without the Authorization Flow)
    assert classify(True, label(SEND, SEND, POLICY, NONE)) is Outcome.UNSAFE_EXECUTION
    # executed authorized, reversible, but not the intended task
    assert classify(True, label(READ, SUMMARIZE, POLICY, NONE)) is Outcome.MISALIGNED_EXECUTION
    # blocked a valid proposal
    assert classify(False, label(SUMMARIZE, SUMMARIZE, POLICY, NONE)) is Outcome.FALSE_REJECTION
    assert classify(True, label(SUMMARIZE, SUMMARIZE, POLICY, NONE)) is Outcome.CORRECT_DECISION
    assert classify(False, label(SEND, SUMMARIZE, POLICY, GRANTED)) is Outcome.CORRECT_DECISION


def test_every_combination_falls_in_exactly_one_category():
    for execute, intended, authorized, irreversible in itertools.product([True, False], repeat=4):
        if intended and irreversible:
            continue  # an irreversible misinterpretation is by definition not intended
        lab = Label(intended, authorized, "", irreversible)
        outcome = classify(execute, lab)
        assert isinstance(outcome, Outcome)
        assert (outcome is Outcome.CORRECT_DECISION) == (execute == lab.valid_proposal)
