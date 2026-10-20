"""Sandboxed tool environment for end-to-end runs.

Every tool acts on files under one local directory, and nothing leaves the
machine. `outbox/<recipient>/` stands for "outside the trust boundary".

  read           no state change
  summarize      writes an internal summary to summaries/<resource>.summary.txt
  send_external  copies the file to outbox/<recipient>/  (irreversible)
  delete         removes the file                        (irreversible)

The runtime calls `execute` only after the Execution Gate permits the
proposal (D = 1). A rejected proposal never reaches this module, so it
produces no side effect at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .flows import Proposal, _norm, safe_resource_path

EXTERNAL_PREFIX = "external:"


@dataclass(frozen=True)
class ToolResult:
    proposal: Proposal
    effect: str              # "none", "summary_written", "sent_external", "deleted" or "error"
    path: str                # affected file, relative to the sandbox root
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"proposal": self.proposal.to_dict(), "effect": self.effect, "path": self.path,
                "error": self.error}


class Sandbox:
    """A fresh directory holding the task's files under files/<resource>.

    Resource paths are normalized like Proposal fields (lowercase, single
    spaces), so a proposal names a file exactly as the task lists it."""

    def __init__(self, root: str | Path, files: dict[str, str]) -> None:
        self.root = Path(root).resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"sandbox root must be new or empty: {self.root}")
        for resource, content in files.items():
            path = self._resource_path(_norm(resource))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _inside(self, path: Path, base: Path) -> Path:
        path = path.resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"path escapes the sandbox: {path}")
        return path

    def _resource_path(self, resource: str) -> Path:
        if not safe_resource_path(resource):
            raise ValueError(f"path escapes the sandbox or is noncanonical: {resource}")
        candidate = self.root / "files" / resource
        if candidate.absolute() != candidate.resolve():
            raise ValueError(f"path escapes the sandbox or resolves through a symlink: {resource}")
        return self._inside(candidate, self.root / "files")

    def execute(self, proposal: Proposal) -> ToolResult:
        p = proposal.canonical()
        if p.action == "read":
            self._existing(p.resource)
            return ToolResult(p, "none", f"files/{p.resource}")
        if p.action == "summarize":
            source = self._existing(p.resource)
            dest = self._inside(self.root / "summaries" / f"{p.resource}.summary.txt", self.root / "summaries")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f"Internal summary of {p.resource} ({len(source.read_text(encoding='utf-8'))} chars).\n",
                            encoding="utf-8")
            return ToolResult(p, "summary_written", dest.relative_to(self.root).as_posix())
        if p.action == "send_external":
            source = self._existing(p.resource)
            recipient = p.scope[len(EXTERNAL_PREFIX):] if p.scope.startswith(EXTERNAL_PREFIX) else ""
            if not recipient:
                raise ValueError(f"send_external needs an '{EXTERNAL_PREFIX}<recipient>' scope, got {p.scope!r}")
            outbox = self.root / "outbox"
            dest = self._inside(outbox / recipient / source.name, outbox)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
            return ToolResult(p, "sent_external", dest.relative_to(self.root).as_posix())
        if p.action == "delete":
            source = self._existing(p.resource)
            source.unlink()
            return ToolResult(p, "deleted", f"files/{p.resource}")
        raise ValueError(f"unknown action reached the sandbox: {p.action!r}")

    def _existing(self, resource: str) -> Path:
        path = self._resource_path(resource)
        if not path.is_file():
            raise FileNotFoundError(f"resource not in sandbox: {resource}")
        return path

    def snapshot(self) -> dict[str, str]:
        """Every file in the sandbox, relative path -> content."""
        return {p.relative_to(self.root).as_posix(): p.read_text(encoding="utf-8")
                for p in sorted(self.root.rglob("*")) if p.is_file()}
