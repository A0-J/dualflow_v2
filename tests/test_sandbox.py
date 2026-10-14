import pytest

from dualflow.flows import Proposal
from dualflow.sandbox import Sandbox

FILES = {"audit/q3_raw_report.txt": "full report", "audit/q3_redacted_report.txt": "redacted report"}


def test_tools_have_the_documented_side_effects(tmp_path):
    box = Sandbox(tmp_path / "s", FILES)
    before = box.snapshot()
    assert box.execute(Proposal("read", "audit/q3_raw_report.txt", "internal")).effect == "none"
    assert box.snapshot() == before

    r = box.execute(Proposal("summarize", "audit/q3_raw_report.txt", "internal"))
    assert r.effect == "summary_written" and r.path in box.snapshot()

    r = box.execute(Proposal("send_external", "audit/q3_redacted_report.txt", "external:auditor"))
    assert r.effect == "sent_external"
    assert box.snapshot()["outbox/auditor/q3_redacted_report.txt"] == "redacted report"

    r = box.execute(Proposal("delete", "audit/q3_raw_report.txt", "internal"))
    assert r.effect == "deleted" and "files/audit/q3_raw_report.txt" not in box.snapshot()


def test_paths_cannot_escape_the_sandbox(tmp_path):
    box = Sandbox(tmp_path / "s", FILES)
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the sandbox"):
        box.execute(Proposal("delete", "../../secret.txt", "internal"))
    with pytest.raises(ValueError, match="escapes the sandbox"):
        box.execute(Proposal("send_external", "audit/q3_redacted_report.txt", "external:../../x"))
    assert (tmp_path / "secret.txt").exists()


def test_errors(tmp_path):
    box = Sandbox(tmp_path / "s", FILES)
    with pytest.raises(FileNotFoundError):
        box.execute(Proposal("read", "audit/missing.txt", "internal"))
    with pytest.raises(ValueError, match="recipient"):
        box.execute(Proposal("send_external", "audit/q3_redacted_report.txt", "internal"))
    with pytest.raises(ValueError, match="unknown action"):
        box.execute(Proposal("export", "audit/q3_raw_report.txt", "internal"))


def test_refuses_a_non_empty_root(tmp_path):
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        Sandbox(tmp_path, FILES)
