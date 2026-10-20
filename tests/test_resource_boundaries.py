import pytest

from dualflow.flows import (
    AuthorizationPolicy, AuthorizationState, Proposal, authorization_flow,
    resource_under_prefix, safe_resource_path,
)
from dualflow.sandbox import Sandbox


POLICY = AuthorizationPolicy.from_dict({
    "allowed_actions": ["read", "summarize"],
    "resource_prefixes": ["legal/executed/"],
    "allowed_scopes": ["internal"],
})
STATE = AuthorizationState.from_dict({"granted": []})


@pytest.mark.parametrize("resource", [
    "legal/executed/../drafts/contract_v3.docx",
    "legal/executed/./contract.pdf",
    "legal/executed//contract.pdf",
    "legal/executed_backup/contract.pdf",
    "/legal/executed/contract.pdf",
    "legal\\executed\\contract.pdf",
])
def test_authorization_rejects_noncanonical_or_outside_paths(resource):
    outcome = authorization_flow(Proposal("read", resource, "internal"), POLICY, STATE)
    assert not outcome.allowed and outcome.reason == "resource_not_granted"


def test_authorization_allows_real_child():
    assert resource_under_prefix("legal/executed/contract.pdf", "legal/executed/")
    assert authorization_flow(
        Proposal("read", "legal/executed/contract.pdf", "internal"), POLICY, STATE
    ).allowed


@pytest.mark.parametrize("path", [
    "legal/executed/../drafts/secret.txt",
    "legal/executed/./contract.pdf",
    "legal/executed//contract.pdf",
    "/legal/executed/contract.pdf",
])
def test_sandbox_rejects_alias_paths(tmp_path, path):
    box = Sandbox(tmp_path / "box", {"legal/executed/contract.pdf": "contract"})
    with pytest.raises(ValueError, match="escapes the sandbox"):
        box.execute(Proposal("read", path, "internal"))


def test_sandbox_rejects_symlink_to_other_resource(tmp_path):
    box = Sandbox(tmp_path / "box", {
        "legal/executed/contract.pdf": "contract",
        "legal/drafts/secret.txt": "secret",
    })
    link = box.root / "files/legal/executed/alias.txt"
    try:
        link.symlink_to(box.root / "files/legal/drafts/secret.txt")
    except OSError:
        pytest.skip("symlink unavailable")
    assert authorization_flow(
        Proposal("read", "legal/executed/alias.txt", "internal"), POLICY, STATE
    ).allowed
    # The policy cannot detect a symlink; the sandbox must fail closed.
    with pytest.raises(ValueError, match="symlink"):
        box.execute(Proposal("read", "legal/executed/alias.txt", "internal"))
