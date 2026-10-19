"""The τ²-bench adapter's core, without τ²-bench: which calls reach the environment, how a
write waits for the user's yes, and that the conversation stays well-formed."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from toolagent.models import ScriptedModel, ToolCall, Turn, Usage
from toolagent.tau2_agent import (
    NEEDS_CONFIRMATION,
    Reply,
    Tau2Policy,
    ToolResult,
    UserText,
    effect_for,
)
from toolagent.tools import Effect


class Reservation(BaseModel):
    reservation_id: str


TOOLS = [
    ("get_reservation_details", "Read a reservation.", Reservation),
    ("cancel_reservation", "Cancel a reservation.", Reservation),
]
GET = ("get_reservation_details", {"reservation_id": "ABC123"})
CANCEL = ("cancel_reservation", {"reservation_id": "ABC123"})


def _policy(steps: list[Any]) -> tuple[Tau2Policy, ScriptedModel]:
    model = ScriptedModel(steps)
    return Tau2Policy(model, TOOLS, "Airline policy: confirm before changing a booking."), model


def _names(reply: Reply) -> list[str]:
    return [c.name for c in reply.calls]


def _assert_well_formed(messages: list[dict[str, Any]]) -> None:
    """Every tool_use is answered in the next message, by id."""
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        ids = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
        if ids:
            nxt = messages[i + 1]["content"]
            answered = [b["tool_use_id"] for b in nxt if b.get("type") == "tool_result"]
            assert sorted(answered) == sorted(ids)


def test_a_write_waits_for_an_explicit_yes() -> None:
    p, _ = _policy([[GET], [CANCEL], "I'll cancel ABC123. Shall I go ahead?", [CANCEL], "Done."])
    first = p.step(UserText("Please cancel my reservation ABC123."))
    assert _names(first) == ["get_reservation_details"]  # reads go straight to the environment
    asked = p.step([ToolResult(first.calls[0].id, '{"status": "confirmed"}')])
    # the cancel was held back, the model was told why, and it asked the user
    assert asked == Reply("I'll cancel ABC123. Shall I go ahead?")
    held = p.messages[-2]["content"][0]
    assert held["is_error"] and held["content"] == NEEDS_CONFIRMATION
    go = p.step(UserText("Yes, go ahead."))
    assert _names(go) == ["cancel_reservation"]
    done = p.step([ToolResult(go.calls[0].id, "cancelled")])
    assert done.text == "Done."
    _assert_well_formed(p.messages)


def test_a_yes_covers_one_turn_of_writes_and_a_write_is_never_repeated() -> None:
    other = ("cancel_reservation", {"reservation_id": "XYZ789"})
    p, _ = _policy([[CANCEL], [other], "Should I cancel XYZ789 too?", [CANCEL], "ok"])
    go = p.step(UserText("yes"))
    assert _names(go) == ["cancel_reservation"]
    nxt = p.step([ToolResult(go.calls[0].id, "cancelled")])
    assert nxt.text == "Should I cancel XYZ789 too?"  # the second write needed a new yes
    again = p.step(UserText("yes"))  # the model repeats the first cancel instead
    assert again.text == "ok" and again.calls == []
    refused = p.messages[-2]["content"][0]["content"]
    assert "already done" in refused
    _assert_well_formed(p.messages)


def test_a_mixed_turn_sends_reads_and_answers_held_writes_with_the_results() -> None:
    p, _ = _policy([[GET, CANCEL], "Here are the details; confirm the cancellation?"])
    reply = p.step(UserText("Cancel ABC123."))
    assert _names(reply) == ["get_reservation_details"]
    p.step([ToolResult(reply.calls[0].id, "{}")])
    blocks = p.messages[2]["content"]
    assert [b["content"] for b in blocks] == [NEEDS_CONFIRMATION, "{}"]
    _assert_well_formed(p.messages)


def test_invalid_and_unknown_calls_are_answered_without_the_environment() -> None:
    p, _ = _policy([[("get_reservation_details", {})], [("delete_all", {})], "Sorry."])
    reply = p.step(UserText("hi"))
    assert reply == Reply("Sorry.")
    held = [m for m in p.messages if m["role"] == "user" and isinstance(m["content"], list)]
    errors = [m["content"][0]["content"] for m in held]
    assert errors[0].startswith("Not run: invalid arguments")
    assert errors[1].startswith("Not run: unknown tool 'delete_all'")


class _CutOff:
    name = "cut"

    def turn(self, system: str, tools: list[Any], messages: list[Any]) -> Turn:
        c = ToolCall("t1", "cancel_reservation", {"reservation_id": "ABC123"})
        content = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}]
        return Turn(content, [c], "", "max_tokens", Usage(), "cut")


def test_a_cut_off_turn_runs_nothing_and_its_calls_are_answered_before_the_next_user_turn() -> None:
    p = Tau2Policy(_CutOff(), TOOLS, "policy")
    assert p.step(UserText("yes")).calls == []
    p.model = ScriptedModel(["fine"])
    p.step(UserText("are you there?"))
    owed = p.messages[2]["content"]
    assert owed[0]["tool_use_id"] == "t1" and owed[0]["is_error"]
    assert owed[1] == {"type": "text", "text": "are you there?"}


@pytest.mark.parametrize(
    ("name", "effect"),
    [("get_user_details", Effect.READ), ("search_direct_flight", Effect.READ),
     ("calculate", Effect.READ), ("transfer_to_human_agents", Effect.READ),
     ("book_reservation", Effect.IRREVERSIBLE),
     ("modify_pending_order_items", Effect.IRREVERSIBLE)],
)  # fmt: skip
def test_tools_are_labelled_by_name(name: str, effect: Effect) -> None:
    assert effect_for(name) is effect


def test_the_tau2_wiring_imports_where_tau2_is_installed() -> None:
    pytest.importorskip("tau2")
    from toolagent.tau2_agent import gated_agent_class

    assert gated_agent_class().__name__ == "GatedAgent"
