"""Observability: one span per model turn, tool call, human review and rollback, under one span for
the run. Spans go to JSONL (`--trace FILE`) and can be replayed into OpenTelemetry (`otel.py`).

Attribute names follow the OpenTelemetry GenAI conventions where one exists (operation, model,
token usage, tool name and call id); the agent's own facts — gate decision, review outcome, cost,
cache tokens — are under `agent.*`.
"""

from __future__ import annotations

import json
import math
import secrets
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

OP = "gen_ai.operation.name"


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # or "error"

    @property
    def kind(self) -> str:
        """invoke_agent, chat, execute_tool, approval or rollback."""
        return self.name.split(" ", 1)[0]

    @property
    def seconds(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9


class Tracer:
    def __init__(self, clock: Callable[[], int] = time.time_ns) -> None:
        self.clock = clock
        self.trace_id = secrets.token_hex(16)
        self.spans: list[Span] = []
        self._open: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = self._open[-1].span_id if self._open else None
        s = Span(
            name,
            self.trace_id,
            secrets.token_hex(8),
            parent,
            self.clock(),
            0,
            attributes,
        )
        self.spans.append(s)
        self._open.append(s)
        try:
            yield s
        except BaseException as e:
            s.status = "error"
            s.attributes["error.type"] = type(e).__name__
            raise
        finally:
            s.end_ns = self.clock()
            self._open.pop()

    def write_jsonl(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w") as f:
            f.writelines(json.dumps(asdict(s), default=str) + "\n" for s in self.spans)


def read_jsonl(path: str | Path) -> list[Span]:
    return [Span(**json.loads(line)) for line in Path(path).read_text().splitlines() if line]


def _quantile(values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile; small samples are reported with their n so nobody over-reads them."""
    ordered = sorted(values)
    rank = max(1, math.ceil(round(q * len(ordered), 9)))  # round: 0.95 * 20 is 19.000000000000004
    return ordered[rank - 1]


@dataclass(frozen=True)
class TraceSummary:
    status: str
    turns: int
    tool_calls: int
    decisions: dict[str, int]
    reviews: dict[str, int]  # approved / declined
    input_tokens: int  # all prompt tokens, cached or not
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float | None
    wall_s: float
    latency_s: dict[str, dict[str, float]]  # span kind -> n, p50, p95, max

    def lines(self) -> list[str]:
        cost = "unknown" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        cached = self.cache_read_tokens / self.input_tokens if self.input_tokens else 0.0
        out = [
            f"status {self.status}; {self.turns} turns, {self.tool_calls} tool calls, "
            f"{self.wall_s:.2f}s wall",
            "gate: " + ", ".join(f"{k} {v}" for k, v in sorted(self.decisions.items())),
            "reviews: "
            + (", ".join(f"{k} {v}" for k, v in sorted(self.reviews.items())) or "none"),
            f"tokens: {self.input_tokens} in ({cached:.0%} from cache), {self.output_tokens} out; "
            f"cost {cost}",
        ]
        for kind, s in self.latency_s.items():
            out.append(
                f"  {kind:<13} n={int(s['n']):<3} p50 {s['p50']:.3f}s  p95 {s['p95']:.3f}s  "
                f"max {s['max']:.3f}s"
            )
        return out


def summarize(spans: Sequence[Span]) -> TraceSummary:
    root = next(s for s in spans if s.parent_id is None)
    chats = [s for s in spans if s.kind == "chat"]
    tools = [s for s in spans if s.kind == "execute_tool"]
    decisions: dict[str, int] = {}
    for s in tools:
        d = str(s.attributes.get("agent.gate.decision", "?"))
        decisions[d] = decisions.get(d, 0) + 1
    reviews: dict[str, int] = {}
    for s in spans:
        if s.kind == "approval":
            r = "approved" if s.attributes.get("agent.review.approved") else "declined"
            reviews[r] = reviews.get(r, 0) + 1
    costs = [s.attributes.get("agent.cost_usd") for s in chats]
    latency: dict[str, dict[str, float]] = {}
    for kind in ("chat", "execute_tool", "approval"):
        secs = [s.seconds for s in spans if s.kind == kind]
        if secs:
            latency[kind] = {
                "n": len(secs),
                "p50": _quantile(secs, 0.50),
                "p95": _quantile(secs, 0.95),
                "max": max(secs),
            }
    return TraceSummary(
        status=str(root.attributes.get("agent.status", "?")),
        turns=len(chats),
        tool_calls=len(tools),
        decisions=decisions,
        reviews=reviews,
        input_tokens=sum(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) for s in chats),
        output_tokens=sum(int(s.attributes.get("gen_ai.usage.output_tokens", 0)) for s in chats),
        cache_read_tokens=sum(
            int(s.attributes.get("agent.usage.cache_read_tokens", 0)) for s in chats
        ),
        cost_usd=None if any(c is None for c in costs) else sum(float(c) for c in costs if c),
        wall_s=root.seconds,
        latency_s=latency,
    )


def timeline(spans: Sequence[Span]) -> list[str]:
    """The run as an indented tree, one line per span."""
    depth: dict[str | None, int] = {None: -1}
    t0 = min(s.start_ns for s in spans)
    out = []
    for s in spans:
        depth[s.span_id] = depth.get(s.parent_id, -1) + 1
        a = s.attributes
        detail = ""
        if s.kind == "chat":
            detail = (
                f"{a.get('gen_ai.usage.output_tokens', 0)} out, stop {a.get('agent.stop_reason')}"
            )
        elif s.kind == "execute_tool":
            why = a.get("agent.gate.reason") or a.get("agent.gate.rule", "")
            detail = f"{a.get('agent.gate.decision')}: {why}"
            if a.get("agent.tool.error"):
                detail += " [error]"
        elif s.kind == "approval":
            detail = "approved" if a.get("agent.review.approved") else "declined"
        elif s.kind == "invoke_agent":
            detail = f"{a.get('agent.status')}: {a.get('agent.reason') or ''}"
        out.append(
            f"{(s.start_ns - t0) / 1e9:7.3f}s {'  ' * depth[s.span_id]}{s.name}"
            f" ({s.seconds:.3f}s) {detail}".rstrip()
        )
    return out
