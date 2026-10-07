from dualflow.flows import (
    AuthorizationPolicy, AuthorizationState, Proposal, authorization_flow,
    dualflow, parse_proposal,
)

SEND = Proposal("send_external", "audit/q3_redacted_report.txt", "external:auditor")
SUMMARIZE = Proposal("summarize", "audit/q3_raw_report.txt", "internal")

POLICY = AuthorizationPolicy.from_dict({
    "allowed_actions": ["read", "summarize", "send_external"],
    "resource_prefixes": ["audit/"],
    "allowed_scopes": ["internal", "external:auditor"],
    "approval_required": ["send_external", "delete"],
})
GRANTED = AuthorizationState.from_dict({"granted": [SEND.to_dict()]})
NONE = AuthorizationState.from_dict({"granted": []})


def test_canonical_quadrants():
    # C1: intended and authorized
    assert dualflow(SEND, [SEND] * 3, POLICY, GRANTED).execute
    # C2: authorized but not intended (Planner wants a summary)
    d = dualflow(SEND, [SUMMARIZE] * 3, POLICY, GRANTED)
    assert not d.execute and not d.semantic.passed and d.authorization.allowed
    # C3: intended but approval not granted
    d = dualflow(SEND, [SEND] * 3, POLICY, NONE)
    assert not d.execute and d.semantic.passed and d.authorization.reason == "approval_not_granted"
    # C4: neither
    d = dualflow(SEND, [SUMMARIZE] * 3, POLICY, NONE)
    assert not d.execute and not d.semantic.passed and not d.authorization.allowed


def test_approval_must_match_exact_operation():
    other_recipient = Proposal("send_external", "audit/q3_redacted_report.txt", "external:customer")
    policy = AuthorizationPolicy.from_dict({
        "allowed_actions": ["send_external"], "resource_prefixes": ["audit/"],
        "allowed_scopes": ["external:auditor", "external:customer"],
        "approval_required": ["send_external"],
    })
    assert authorization_flow(other_recipient, policy, GRANTED).reason == "approval_not_granted"


def test_policy_reasons():
    assert authorization_flow(SEND, AuthorizationPolicy.from_dict({
        "allowed_actions": ["read", "summarize"], "resource_prefixes": ["audit/"],
        "allowed_scopes": ["internal", "external:auditor"]}), GRANTED).reason == "action_not_granted"
    assert authorization_flow(SEND, AuthorizationPolicy.from_dict({
        "allowed_actions": ["send_external"], "resource_prefixes": ["hr/"],
        "allowed_scopes": ["external:auditor"]}), GRANTED).reason == "resource_not_granted"
    assert authorization_flow(SEND, AuthorizationPolicy.from_dict({
        "allowed_actions": ["send_external"], "resource_prefixes": ["audit/"],
        "allowed_scopes": ["internal"]}), GRANTED).reason == "scope_not_granted"
    assert authorization_flow(SUMMARIZE, POLICY, NONE).allowed  # no approval needed


def test_semantic_uses_planner_mode_and_normalizes():
    noisy = [SUMMARIZE, SUMMARIZE, SEND]
    assert not dualflow(SEND, noisy, POLICY, GRANTED).semantic.passed
    shouty = Proposal(" Send_External ", "AUDIT/q3_redacted_report.txt", "external:auditor")
    assert dualflow(shouty, [SEND], POLICY, GRANTED).execute


def test_parse_proposal_ignores_fences_and_extra_text():
    p = parse_proposal('```json\n{"action": "Summarize", "resource": "audit/q3_raw_report.txt", "scope": "internal"}\n``` ok')
    assert p == SUMMARIZE
