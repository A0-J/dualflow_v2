import pytest

from dualflow.flows import parse_proposal
from dualflow.roles import parse_delegation


def test_parses_plain_json():
    p = parse_proposal('{"action": "summarize", "resource": "audit/q3_raw_report.txt", "scope": "internal"}')
    assert (p.action, p.resource, p.scope) == ("summarize", "audit/q3_raw_report.txt", "internal")


def test_strips_markdown_fence():
    text = '```json\n{"action": "read", "resource": "audit/q3_raw_report.txt", "scope": "internal"}\n```'
    assert parse_proposal(text).action == "read"


def test_ignores_trailing_content_after_json():
    # A greedy regex would span to the last `}` and fail to decode.
    text = ('{"action": "send_external", "resource": "audit/q3_redacted_report.txt", "scope": "external:auditor"}\n'
            'Note: approval confirmed separately. {"status": "ok"}')
    p = parse_proposal(text)
    assert p.action == "send_external" and p.scope == "external:auditor"


def test_brace_inside_string_value_does_not_end_the_object():
    p = parse_proposal('{"action": "read", "resource": "notes/{draft}.txt", "scope": "internal"}')
    assert p.resource == "notes/{draft}.txt"


def test_proposal_is_canonicalized():
    p = parse_proposal('{"action": " Summarize ", "resource": "Audit/Q3_raw_report.txt", "scope": "INTERNAL"}')
    assert (p.action, p.resource, p.scope) == ("summarize", "audit/q3_raw_report.txt", "internal")


def test_missing_field_is_an_error():
    with pytest.raises(ValueError, match="Missing proposal fields"):
        parse_proposal('{"action": "read", "resource": "audit/q3_raw_report.txt"}')


def test_parse_delegation():
    assert parse_delegation('```json\n{"delegation": " Summarize the Q3 report. "}\n```') == "Summarize the Q3 report."
    with pytest.raises(ValueError, match="Missing delegation"):
        parse_delegation('{"delegation": ""}')


def test_parse_executor_turn():
    from dualflow.roles import parse_executor_turn
    p, q, unclear = parse_executor_turn(
        '{"decision": "execute", "action": "summarize", "resource": "audit/q3_raw_report.txt", "scope": "internal"}')
    assert p.action == "summarize" and q is None and unclear == ()
    p, q, unclear = parse_executor_turn(
        '```json\n{"decision": "ASK", "question": "Send it, or summarize it?", "unclear": ["Scope", "action", "x"]}\n```')
    assert p is None and q == "Send it, or summarize it?" and unclear == ("action", "scope")
    for bad in ('{"decision": "execute", "action": "read"}', '{"decision": "ask", "question": ""}',
                '{"decision": "maybe"}', '{"decision": "ask", "question": "q", "unclear": "action"}'):
        with pytest.raises(ValueError):
            parse_executor_turn(bad)
