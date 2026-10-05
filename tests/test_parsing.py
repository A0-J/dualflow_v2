from dualflow.parsing import parse_interpretation


def test_parses_plain_json():
    interp = parse_interpretation(
        '{"action": "summarize", "resource": "document", "scope": "audit/q3_raw_report.txt", "condition": []}'
    )
    assert interp.action == "summarize"


def test_strips_markdown_fence():
    text = '```json\n{"action": "read", "resource": "document", "scope": "audit/q3_raw_report.txt", "condition": []}\n```'
    interp = parse_interpretation(text)
    assert interp.action == "read"


def test_ignores_trailing_content_after_json():
    # Regression: a greedy `\{.*\}` regex spans to the LAST `}` in the
    # text, which breaks (json.JSONDecodeError: Extra data) if the model
    # appends anything containing a `}` after its answer -- seen live
    # during the V2-V5 valid-scenario collection (2026-10).
    text = (
        '{"action": "send_external", "resource": "document", '
        '"scope": "audit/q3_redacted_report.txt", "condition": ["recipient=approved_auditor"]}\n'
        'Note: approval confirmed separately. {"status": "ok"}'
    )
    interp = parse_interpretation(text)
    assert interp.action == "send_external"
    assert interp.scope == "audit/q3_redacted_report.txt"


def test_recovers_from_stray_brace_before_the_real_object():
    # Regression: a non-greedy `\{.*?\}` regex stops at the FIRST `}` it
    # sees, which breaks (json.JSONDecodeError: Expecting ',' delimiter)
    # if a `}` appears before the real object's end -- seen live on the
    # retry after the greedy-regex fix above (2026-10).
    text = (
        'Treating {the redacted file} as the target.\n'
        '{"action": "send_external", "resource": "document", '
        '"scope": "audit/q3_redacted_report.txt", "condition": []}'
    )
    interp = parse_interpretation(text)
    assert interp.action == "send_external"


def test_missing_field_raises():
    try:
        parse_interpretation('{"action": "read", "resource": "document", "scope": "x"}')
    except ValueError as e:
        assert "condition" in str(e)
    else:
        raise AssertionError("expected ValueError for missing field")
