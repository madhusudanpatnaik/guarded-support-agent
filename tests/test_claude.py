"""The Claude model through the real `anthropic` SDK, over a mock HTTP transport: the request the
API receives, how the response is read, and that each turn's content goes back unchanged."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest

from toolagent.agent import Agent
from toolagent.approval import DenyAll
from toolagent.backoffice import SYSTEM, BackOffice
from toolagent.models import ClaudeModel
from toolagent.policy import Policy


def _message(content: list[dict[str, Any]], stop: str, usage: dict[str, int]) -> httpx2.Response:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage,
    }
    return httpx2.Response(200, json=body)


TOOL_TURN: list[dict[str, Any]] = [
    {"type": "thinking", "thinking": "Read the ticket before anything else.", "signature": "sig-1"},
    {"type": "tool_use", "id": "toolu_1", "name": "get_ticket", "input": {"ticket_id": "T-1"}},
]
FINAL_TURN: list[dict[str, Any]] = [
    {"type": "thinking", "thinking": "", "signature": "sig-2"},
    {"type": "text", "text": "Jane wants a refund for a cracked jug; nothing done yet."},
]


def _client(responses: list[httpx2.Response]) -> tuple[anthropic.Anthropic, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    queue = iter(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(
            {
                "url": str(request.url),
                "beta": request.headers.get("anthropic-beta"),
                "body": json.loads(request.content),
            }
        )
        return next(queue)

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    return anthropic.Anthropic(api_key="test-key", http_client=http, max_retries=0), sent


def _run(
    responses: list[httpx2.Response], capture: bool | None = None, **policy: Any
) -> tuple[Any, list[dict[str, Any]]]:
    client, sent = _client(responses)
    env = BackOffice()
    agent = Agent(
        ClaudeModel(client, effort="low"),
        env.toolbox(),
        approver=DenyAll(),
        policy=Policy(**policy),
        system=SYSTEM,
        capture_content=capture,
    )
    return agent.run("Handle ticket T-1."), sent


RESPONSES = [
    _message(TOOL_TURN, "tool_use",
             {"input_tokens": 1200, "output_tokens": 80, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 1000}),
    _message(FINAL_TURN, "end_turn",
             {"input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": 2200,
              "cache_creation_input_tokens": 100}),
]  # fmt: skip


def test_the_request_carries_the_documented_parameters() -> None:
    _, sent = _run(list(RESPONSES))
    first = sent[0]
    assert first["url"].endswith("/v1/messages?beta=true")
    assert first["beta"] == "server-side-fallback-2026-07-01"
    body = first["body"]
    assert body["model"] == "claude-opus-5-5" and body["system"] == SYSTEM
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "low"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["fallbacks"] == "default"
    assert "temperature" not in body and "tool_choice" not in body  # neither is allowed here
    tools = {t["name"]: t for t in body["tools"]}
    assert len(tools) == 12 and all(t["strict"] for t in tools.values())
    assert tools["issue_refund"]["input_schema"]["required"] == ["order_id", "amount", "reason"]


def test_each_turn_is_sent_back_unchanged_and_answered() -> None:
    result, sent = _run(list(RESPONSES))
    assert result.status == "completed"
    assert result.answer == "Jane wants a refund for a cracked jug; nothing done yet."
    second = sent[1]["body"]
    assert second["system"] == sent[0]["body"]["system"]  # system and tools never change
    assert second["tools"] == sent[0]["body"]["tools"]
    user, assistant, results = second["messages"]
    assert user == {"role": "user", "content": "Handle ticket T-1."}
    thinking, tool_use = assistant["content"]
    assert (thinking["type"], thinking["thinking"], thinking["signature"]) == (
        "thinking", "Read the ticket before anything else.", "sig-1",
    )  # fmt: skip
    assert (tool_use["id"], tool_use["name"], tool_use["input"]) == (
        "toolu_1", "get_ticket", {"ticket_id": "T-1"},
    )  # fmt: skip
    (r,) = results["content"]
    assert r["tool_use_id"] == "toolu_1" and r["is_error"] is False
    assert json.loads(r["content"])["customer_email"] == "jane.doe@example.com"


def test_usage_cost_and_reasoning_reach_the_result_and_the_trace() -> None:
    result, _ = _run(list(RESPONSES), capture=True)
    u = result.usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (
        1250, 100, 2200, 1100,
    )  # fmt: skip
    # turn 1: 1200*$4 + 80*$20 + 1000*$5 (cache write); turn 2: 50*$4 + 20*$20 + 2200*$0.20 + 100*$5
    assert result.cost_usd == pytest.approx((11400 + 1540) / 1e6)
    chat = [s for s in result.tracer.spans if s.kind == "chat"]
    assert chat[0].attributes["gen_ai.usage.input_tokens"] == 2200  # all prompt tokens
    assert chat[0].attributes["agent.reasoning"] == "Read the ticket before anything else."
    assert chat[1].attributes["agent.usage.cache_read_tokens"] == 2200
    assert chat[0].attributes["gen_ai.response.finish_reasons"] == ["tool_use"]


def test_a_refusal_ends_the_run_without_running_anything() -> None:
    refusal = _message(
        [{"type": "tool_use", "id": "toolu_9", "name": "close_account",
          "input": {"customer_id": "C-1001"}}],
        "refusal",
        {"input_tokens": 10, "output_tokens": 1},
    )  # fmt: skip
    result, sent = _run([refusal])
    assert result.status == "model_refused" and result.calls == [] and len(sent) == 1


def test_an_api_error_ends_the_run_as_an_error() -> None:
    err = httpx2.Response(
        529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    )
    result, _ = _run([err])
    assert result.status == "error" and "529" in result.reason


def test_reasoning_and_reasons_stay_out_of_traces_unless_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    result, _ = _run(list(RESPONSES))
    chat = [s for s in result.tracer.spans if s.kind == "chat"]
    tool = next(s for s in result.tracer.spans if s.kind == "execute_tool")
    root = result.tracer.spans[0]
    assert chat[0].attributes["agent.reasoning"] is None
    assert (
        tool.attributes["agent.gate.reason"] is None
        and tool.attributes["agent.gate.rule"] == "read"
    )
    assert (
        root.attributes["agent.reason"] is None and root.attributes["agent.status"] == "completed"
    )
    # usage and timing are not content: still recorded
    assert chat[0].attributes["gen_ai.usage.input_tokens"] == 2200
    # the standard OpenTelemetry switch turns capture on
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    result, _ = _run(list(RESPONSES))
    chat = [s for s in result.tracer.spans if s.kind == "chat"]
    assert chat[0].attributes["agent.reasoning"] == "Read the ticket before anything else."
