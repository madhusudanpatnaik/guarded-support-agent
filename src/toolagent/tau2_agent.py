"""τ²-bench: this agent, gate included, on Sierra's airline / retail / telecom domains.

τ²-bench runs an agent against a simulated user and a domain database, executes the agent's tool
calls itself, and grades the final database against a goal state — the method this project's own
evaluation copies, at a scale (hundreds of tasks, several domains) where pass^k is comparable with
published numbers.

The adapter keeps what this project puts in front of every call — schema validation, repeat and
duplicate checks, the tool-call budget, stuck detection — and maps human review onto τ-bench's own
rule for consequential actions: an explicit yes from the user. A call that writes to the database
goes through only when the user's latest message confirms; otherwise the model is told to describe
the action and ask. A confirmation covers the writes of the turn that follows it, not later ones.

Tools are labelled by name (`get_`, `list_`, `search_`, `find_`, `calculate`, `think` and the
hand-off to a human read; everything else writes) unless `effects` says otherwise. τ²-bench marks
its own tools READ / WRITE in code, but hands an agent only `Tool` objects without that label.

    pip install -r requirements-tau2.txt
    toolagent tau2 run --domain airline --agent toolagent_gated --user-llm <model> --num-trials 4

`tau2` is installed from GitHub (it is not on PyPI); its user simulator needs a key for whichever
model `--user-llm` names.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from toolagent.grounding import Ledger
from toolagent.models import Model, ToolCall, Usage
from toolagent.policy import Decision, Gate, Policy, Verdict
from toolagent.tools import Effect, Tool, Toolbox

AGENT_NAME = "toolagent_gated"
_READ_PREFIXES = ("get_", "list_", "search_", "find_", "calculate", "think", "transfer_to_human")
_CONFIRM = re.compile(
    r"^\W*(yes|yeah|yep|confirm(ed)?|correct|go ahead|proceed|please (do|proceed)|sure|ok(ay)?)\b",
    re.IGNORECASE,
)
NEEDS_CONFIRMATION = (
    "Not run: this changes the user's records. Tell the user exactly what you are about to do and "
    "ask them to confirm with a yes; call it again only after they do."
)


def effect_for(name: str) -> Effect:
    return Effect.READ if name.startswith(_READ_PREFIXES) else Effect.IRREVERSIBLE


@dataclass(frozen=True)
class UserText:
    text: str


@dataclass(frozen=True)
class ToolResult:
    id: str
    content: str
    error: bool = False


@dataclass(frozen=True)
class Reply:
    """What the agent says next: text for the user, or tool calls for the environment."""

    text: str | None
    calls: list[ToolCall] = field(default_factory=list)


def _not_run(_: Any) -> Any:
    raise AssertionError("tools are executed by the τ²-bench environment, not here")


class Tau2Policy:
    """The agent's side of a τ²-bench conversation, independent of τ²-bench's classes."""

    def __init__(
        self,
        model: Model,
        tools: Sequence[tuple[str, str, type[BaseModel]]],  # (name, description, params)
        domain_policy: str,
        *,
        policy: Policy | None = None,
        effects: dict[str, Effect] | None = None,
        max_inner_turns: int = 8,
    ) -> None:
        effects = effects or {}
        self.toolbox = Toolbox(
            Tool(name, desc, params, effects.get(name, effect_for(name)), _not_run)
            for name, desc, params in tools
        )
        self.model, self.system = model, domain_policy
        self.gate = Gate(
            self.toolbox, policy or Policy(max_turns=50, max_tool_calls=60), Ledger("")
        )
        self.schemas = self.toolbox.schemas()
        self.max_inner_turns = max_inner_turns
        self.messages: list[dict[str, Any]] = []
        self.usage = Usage()
        self._kept: list[dict[str, Any]] = []  # results for calls held back, sent with the rest
        self._waiting: dict[str, Verdict] = {}  # calls handed to the environment, by id
        self._confirmed = False  # the user's latest message is an explicit yes

    def step(self, incoming: UserText | Sequence[ToolResult]) -> Reply:
        if isinstance(incoming, UserText):
            self._confirmed = bool(_CONFIRM.match(incoming.text))
            if self._kept:  # answers owed for a cut-off turn go first, or the API rejects it
                text = {"type": "text", "text": incoming.text}
                self.messages.append({"role": "user", "content": [*self._kept, text]})
                self._kept = []
            else:
                self.messages.append({"role": "user", "content": incoming.text})
        else:
            blocks = list(self._kept)
            for r in incoming:
                verdict = self._waiting.pop(r.id, None)
                if verdict is not None and not r.error:
                    self.gate.executed(verdict)
                blocks.append(_result(r.id, r.content, r.error))
            self._kept = []
            self.messages.append({"role": "user", "content": blocks})
        return self._think()

    def _think(self) -> Reply:
        for _ in range(self.max_inner_turns):
            turn = self.model.turn(self.system, self.schemas, self.messages)
            self.usage += turn.usage
            self.messages.append({"role": "assistant", "content": turn.content})
            if turn.stop_reason not in ("end_turn", "tool_use", "stop_sequence"):
                # never run a cut-off or refused turn's calls; answer them so history stays valid
                self._kept = [_result(c.id, "Not run: the turn ended early.", True)
                              for c in turn.tool_calls]  # fmt: skip
                return Reply("I'm sorry, I can't continue with this request.")
            if not turn.tool_calls:
                return Reply(turn.text)
            allowed: list[tuple[ToolCall, Verdict]] = []
            kept: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                v = self.gate.check(call.name, call.args)
                if v.decision is Decision.ASK and self._confirmed:
                    allowed.append((call, v))
                elif v.decision is Decision.ASK:
                    kept.append(_result(call.id, NEEDS_CONFIRMATION, True))
                elif v.decision is Decision.ALLOW:
                    allowed.append((call, v))
                else:
                    kept.append(_result(call.id, f"Not run: {v.reason}.", True))
            if any(v.decision is Decision.ASK for _, v in allowed):
                self._confirmed = False  # a yes covers the writes it was given for, no more
            if allowed:
                self._kept = kept
                self._waiting = {c.id: v for c, v in allowed}
                return Reply(None, [c for c, _ in allowed])
            self.messages.append({"role": "user", "content": kept})
        return Reply("I'm sorry, I couldn't complete that.")


def _result(call_id: str, content: str, error: bool) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": error}


# ------------------------------------------------------------------ τ²-bench wiring


def gated_agent_class(model_factory: Callable[[], Model] | None = None) -> type:
    """A `HalfDuplexAgent` subclass wrapping `Tau2Policy`. Imports τ²-bench, so call it only where
    `tau2` is installed."""
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolMessage,
        UserMessage,
    )
    from tau2.data_model.message import ToolCall as Tau2ToolCall

    from toolagent.models import ClaudeModel

    class GatedAgent(HalfDuplexAgent):  # type: ignore[misc]
        def __init__(self, tools: list[Any], domain_policy: str, **kwargs: Any) -> None:
            try:
                super().__init__(tools=tools, domain_policy=domain_policy)
            except TypeError:
                super().__init__()
            specs = [(t.name, t.openai_schema["function"]["description"], t.params) for t in tools]
            model = model_factory() if model_factory else ClaudeModel()
            self.core = Tau2Policy(model, specs, domain_policy)

        def get_init_state(self, message_history: list[Any] | None = None) -> dict[str, Any]:
            return {}

        def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
            incoming: UserText | list[ToolResult]
            if isinstance(message, UserMessage):
                incoming = UserText(message.content or "")
            elif isinstance(message, MultiToolMessage):
                incoming = [
                    ToolResult(m.id, m.content or "", m.error) for m in message.tool_messages
                ]
            elif isinstance(message, ToolMessage):
                incoming = [ToolResult(message.id, message.content or "", message.error)]
            else:
                raise TypeError(f"unexpected message {type(message).__name__}")
            reply = self.core.step(incoming)
            calls = [Tau2ToolCall(id=c.id, name=c.name, arguments=c.args) for c in reply.calls]
            out = AssistantMessage(
                role="assistant", content=None if calls else reply.text, tool_calls=calls or None
            )
            return out, state

    return GatedAgent


def register(name: str = AGENT_NAME, model_factory: Callable[[], Model] | None = None) -> None:
    """Make the agent selectable as `--agent <name>` in this process's τ²-bench registry."""
    from tau2.registry import registry

    cls = gated_agent_class(model_factory)

    def factory(tools: list[Any], domain_policy: str, **kwargs: Any) -> Any:
        return cls(tools, domain_policy, **kwargs)

    registry.register_agent_factory(factory, name)


def main(argv: Sequence[str]) -> int:
    """Register the agent, then hand the arguments to τ²-bench's own command line."""
    register()
    from tau2.cli import main as tau2_main

    sys.argv = ["tau2", *argv]
    result = tau2_main()
    return int(result or 0)
