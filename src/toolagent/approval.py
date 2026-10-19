"""Human-in-the-loop: who decides an irreversible call the gate will not decide alone.

A reviewer sees what the call does, every argument, where each identifier came from, and the
model's own account of why it wants to do this. A decline goes back to the model as the tool's
result (with the reviewer's note), and the gate refuses the identical call from then on, so the
same person is never asked twice.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TextIO

from toolagent.tools import Effect


@dataclass(frozen=True)
class ApprovalRequest:
    task: str
    tool: str
    effect: Effect
    args: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    reasoning: str = ""  # the model's latest reasoning summary or text, if any

    def render(self) -> str:
        lines = [
            f"The agent wants to run {self.tool} ({self.effect.value}).",
            f"  task: {self.task}",
        ]
        for name, value in self.args.items():
            src = f"   <- from {self.sources[name]}" if name in self.sources else ""
            lines.append(f"  {name} = {json.dumps(value)}{src}")
        if self.reasoning:
            lines.append(f"  why (model): {self.reasoning.strip()[:600]}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Approval:
    approved: bool
    reviewer: str
    note: str = ""


class Approver(Protocol):
    def review(self, request: ApprovalRequest) -> Approval: ...


class ConsoleApprover:
    """Ask the person at the terminal. Anything but an explicit yes is a no."""

    def __init__(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        self.stdin, self.stdout = stdin, stdout

    def _ask(self, prompt: str) -> str:
        self.stdout.write(prompt)
        self.stdout.flush()
        return self.stdin.readline().strip()

    def review(self, request: ApprovalRequest) -> Approval:
        self.stdout.write("\n" + request.render() + "\n")
        answer = self._ask("Approve? [y/N] ").lower()
        if answer in {"y", "yes"}:
            return Approval(True, "console")
        return Approval(False, "console", self._ask("Reason (sent to the agent): "))


class DenyAll:
    """Nobody on call: every irreversible call that no auto rule covers is declined."""

    def review(self, request: ApprovalRequest) -> Approval:
        return Approval(False, "unattended", "no reviewer is available; do not retry")


class ApproveAll:
    """A reviewer who approves whatever is put in front of them. For ablations only: it shows
    what an approval step is worth when nobody actually reviews."""

    def review(self, request: ApprovalRequest) -> Approval:
        return Approval(True, "rubber-stamp")


class RuleApprover:
    """A simulated reviewer for evaluation: a function from the request to a decision."""

    def __init__(self, decide: Callable[[ApprovalRequest], bool], name: str = "simulated") -> None:
        self.decide, self.name = decide, name

    def review(self, request: ApprovalRequest) -> Approval:
        if self.decide(request):
            return Approval(True, self.name)
        return Approval(False, self.name, "not what the customer asked for")
