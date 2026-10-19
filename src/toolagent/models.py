"""The model side of the loop: one call per turn, returning what to do next.

`ClaudeModel` is the production model; `ScriptedModel` replays fixed turns so the loop, the gate
and the evaluation can be tested without a key. Both return a `Turn` whose `content` is appended
to the conversation unchanged: Claude Opus 5.5 ties its thinking blocks to the exact conversation
that produced them, so history is append-only and the system prompt and tool list never change
mid-run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0  # includes thinking
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class Price:
    """USD per million tokens; cache writes at the 5-minute TTL."""

    input: float
    output: float
    cache_read: float
    cache_write: float


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.0, 20.0, 0.20, 5.0),
    # the models a refusal fallback may serve a turn from
    "claude-opus-5": Price(5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-8": Price(5.0, 25.0, 0.50, 6.25),
}


def cost_usd(model: str, usage: Usage) -> float | None:
    """None when the model's price is unknown, rather than a guess."""
    if usage == Usage():
        return 0.0
    p = PRICES.get(model)
    if p is None:
        return None
    return (
        usage.input_tokens * p.input
        + usage.output_tokens * p.output
        + usage.cache_read_tokens * p.cache_read
        + usage.cache_write_tokens * p.cache_write
    ) / 1e6


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Turn:
    content: list[Any]  # the assistant message content, sent back unchanged next turn
    tool_calls: list[ToolCall]
    text: str
    stop_reason: str
    usage: Usage
    model: str
    reasoning: str = ""  # summarized thinking, when the model returns it


class Model(Protocol):
    name: str

    def turn(
        self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Turn: ...


class ClaudeModel:
    """Claude through the Messages API.

    - adaptive thinking (always on for Opus 5.5) with summaries returned, so the trace records why
      the model chose each call; `effort` is the cost/latency control
    - prompt caching on the growing history: each turn re-reads the previous turn's prefix
    - server-side refusal fallback, so a classifier false positive does not end the run
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 16000,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client: Any = client
        self.name = model
        self.effort = effort
        self.max_tokens = max_tokens

    def turn(
        self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Turn:
        r = self.client.beta.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        text = "".join(b.text for b in r.content if b.type == "text")
        reasoning = "\n".join(b.thinking for b in r.content if b.type == "thinking" and b.thinking)
        u = r.usage
        usage = Usage(
            u.input_tokens,
            u.output_tokens,
            u.cache_read_input_tokens or 0,
            u.cache_creation_input_tokens or 0,
        )
        return Turn(list(r.content), calls, text, r.stop_reason or "", usage, r.model, reasoning)


# A scripted step is a final answer, or the tool calls of one turn as (name, args) pairs.
Step = str | Sequence[tuple[str, dict[str, Any]]]


@dataclass
class ScriptedModel:
    """Replays fixed turns; past the end of the script it gives a final answer. Records the
    messages it was shown, so tests can check what the model saw."""

    steps: Sequence[Step]
    name: str = "scripted"
    seen: list[list[dict[str, Any]]] = field(default_factory=list)

    def turn(
        self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Turn:
        n = len(self.seen)
        self.seen.append(list(messages))
        step = self.steps[n] if n < len(self.steps) else "(end of script)"
        if isinstance(step, str):
            content: list[Any] = [{"type": "text", "text": step}]
            return Turn(content, [], step, "end_turn", Usage(), self.name)
        calls = [ToolCall(f"toolu_{n:02d}_{i}", name, args) for i, (name, args) in enumerate(step)]
        content = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in calls]
        return Turn(content, calls, "", "tool_use", Usage(), self.name)
