"""The loop: what runs, what the model is told, how runs end, and what is undone."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from toolagent.agent import Agent, RunResult
from toolagent.approval import Approval, ApprovalRequest, ApproveAll, DenyAll
from toolagent.backoffice import SYSTEM, BackOffice
from toolagent.models import ScriptedModel, ToolCall, Turn, Usage
from toolagent.policy import Decision, Policy
from toolagent.tasks import find


class _Recorder:
    """A reviewer that answers from a list and remembers every request."""

    def __init__(self, *answers: bool) -> None:
        self.answers = list(answers)
        self.requests: list[ApprovalRequest] = []

    def review(self, request: ApprovalRequest) -> Approval:
        self.requests.append(request)
        ok = self.answers.pop(0)
        return Approval(ok, "test", "" if ok else "wrong amount")


def _run(
    steps: list[Any], reviewer: Any = None, task: str = "Handle ticket T-1.", **policy: Any
) -> tuple[RunResult, BackOffice, ScriptedModel]:
    env = BackOffice()
    model = ScriptedModel(steps)
    agent = Agent(
        model,
        env.toolbox(),
        approver=reviewer or ApproveAll(),
        policy=Policy(auto_approve=env.auto_rules(task), **policy),
        system=SYSTEM,
    )
    return agent.run(task), env, model


def _results(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in message["content"] if b["type"] == "tool_result"]


def test_a_careful_run_completes_and_the_conversation_is_well_formed() -> None:
    reviewer = _Recorder(True)
    result, env, model = _run(list(find("damaged-refund").careful), reviewer)
    assert result.status == "completed" and result.turns == 6
    assert env.effects() == [("refund", "O-5001", "129.00"), ("email", "jane.doe@example.com")]
    assert env.tickets["T-1"].status == "solved"
    # every tool_use is answered, in order, by a tool_result with its id in the next message
    msgs = result.messages
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            ids = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if ids:
                assert [r["tool_use_id"] for r in _results(msgs[i + 1])] == ids
    # append-only: what the model saw each turn is a prefix of what it saw next
    for before, after in zip(model.seen, model.seen[1:], strict=False):
        assert after[: len(before)] == before
    # the reviewer saw the source of the order id and the arguments
    (req,) = reviewer.requests
    assert req.tool == "issue_refund" and req.args["amount"] == "129.00"
    assert req.sources == {"order_id": 'get_ticket("T-1")'}
    assert "<- from get_ticket" in req.render()


def test_a_declined_call_does_not_run_and_the_model_is_told_why() -> None:
    refund = ("issue_refund", {"order_id": "O-5001", "amount": 129.0, "reason": "x"})
    steps = [[("get_ticket", {"ticket_id": "T-1"})], [refund], [refund], "Not refunded."]
    reviewer = _Recorder(False)
    result, env, _ = _run(steps, reviewer)
    assert env.refunds == [] and result.status == "completed"
    declined, retried = result.calls[1], result.calls[2]
    assert not declined.executed and "declined: wrong amount" in declined.observation
    assert "Do not try this action again" in declined.observation
    assert retried.decision is Decision.REFUSE and "already declined" in retried.observation
    assert len(reviewer.requests) == 1  # the same person is not asked twice


def test_refused_and_failed_calls_come_back_as_errors() -> None:
    steps = [
        [("issue_refund", {"order_id": "O-9", "amount": 1, "reason": "x"})],  # invented id
        [("get_order", {"order_id": "O-9"})],  # the tool's own error
        "Done.",
    ]
    result, _, model = _run(steps)
    first, second = (_results(model.seen[i][-1])[0] for i in (1, 2))
    assert first["is_error"] and first["content"].startswith("Not run: order_id='O-9'")
    assert second["is_error"] and second["content"] == "Error: no order O-9"


def test_a_stop_mid_turn_skips_the_rest_of_the_turn_and_rolls_back_writes() -> None:
    note = ("add_ticket_note", {"ticket_id": "T-1", "note": "working on it"})
    status = ("set_ticket_status", {"ticket_id": "T-1", "status": "pending"})
    read: tuple[str, dict[str, Any]] = ("list_open_tickets", {})
    steps = [[note, status], [read, read, read, ("get_ticket", {"ticket_id": "T-1"})]]
    result, env, _ = _run(steps, max_tool_calls=4)
    assert result.status == "stopped" and "tool-call budget of 4" in result.reason
    assert [c.decision for c in result.calls[-2:]] == [Decision.STOP, Decision.STOP]
    assert result.calls[-1].observation.startswith("Not run: the run stopped")
    assert env.tickets["T-1"].notes == [] and env.tickets["T-1"].status == "open"
    assert result.rolled_back == [
        'set_ticket_status("T-1", "pending")',
        'add_ticket_note("T-1", "working on it")',
    ]  # newest first


def test_irreversible_calls_are_not_undone_and_are_listed_as_unreviewed() -> None:
    steps = [
        [("get_ticket", {"ticket_id": "T-1"})],
        [("send_email", {"to": "jane.doe@example.com", "subject": "s", "body": "b"})],
        [("add_ticket_note", {"ticket_id": "T-1", "note": "emailed"})],
    ]
    result, env, _ = _run(steps, max_turns=3)
    assert result.status == "stopped" and result.reason == "turn budget of 3 spent"
    assert len(env.outbox) == 1 and env.tickets["T-1"].notes == []
    (sent,) = result.unreviewed
    assert sent.tool == "send_email" and sent.auto_approved


def test_rollback_can_be_turned_off() -> None:
    steps = [[("add_ticket_note", {"ticket_id": "T-1", "note": "x"})]]
    result, env, _ = _run(steps, max_turns=1, rollback_on_abort=False)
    assert result.status == "stopped" and env.tickets["T-1"].notes == ["x"]


@dataclass
class _Fixed:
    stop_reason: str
    calls: list[ToolCall] = field(default_factory=list)
    name: str = "fixed"
    usage: Usage = field(default_factory=Usage)
    model: str = "fixed"

    def turn(self, system: str, tools: list[Any], messages: list[Any]) -> Turn:
        content = [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in self.calls
        ]
        return Turn(content, self.calls, "", self.stop_reason, self.usage, self.model)


def _agent(model: Any, **policy: Any) -> tuple[Agent, BackOffice]:
    env = BackOffice()
    return Agent(model, env.toolbox(), approver=DenyAll(), policy=Policy(**policy)), env


def test_a_turn_cut_off_at_max_tokens_never_runs_its_tools() -> None:
    call = ToolCall("t1", "set_ticket_status", {"ticket_id": "T-1", "status": "solved"})
    agent, env = _agent(_Fixed("max_tokens", [call]))
    result = agent.run("x")
    assert result.status == "truncated" and result.calls == []
    assert env.tickets["T-1"].status == "open"


def test_a_model_refusal_ends_the_run() -> None:
    agent, _ = _agent(_Fixed("refusal"))
    assert agent.run("x").status == "model_refused"


def test_the_cost_budget_is_checked_before_each_model_call() -> None:
    call = ToolCall("t1", "list_open_tickets", {})
    pricey = _Fixed("tool_use", [call], usage=Usage(input_tokens=100_000), model="claude-opus-5-5")
    agent, _ = _agent(pricey, max_cost_usd=0.5)
    result = agent.run("x")  # $0.40 per turn: the second turn runs, the third does not
    assert result.status == "stopped" and result.turns == 2
    assert result.cost_usd == 0.8 and "cost budget of $0.50" in result.reason


def test_an_unknown_model_price_is_reported_as_unknown_not_guessed() -> None:
    model = _Fixed("end_turn", usage=Usage(input_tokens=10), model="some-other-model")
    agent, _ = _agent(model, max_cost_usd=None)
    assert agent.run("x").cost_usd is None


def test_long_tool_output_is_cut_before_the_model_sees_it() -> None:
    steps: list[Any] = [[("get_refund_policy", {})], "ok"]
    result, _, _ = _run(steps, max_observation_chars=50)
    obs = result.calls[0].observation
    assert obs.startswith("Refund policy") and "[truncated:" in obs and len(obs) < 120


def test_a_crashing_tool_ends_the_run_with_an_error_and_rolls_back() -> None:
    env = BackOffice()
    box = env.toolbox()

    def boom(_: Any) -> Any:
        raise RuntimeError("database is down")

    object.__setattr__(box["get_order"], "run", boom)
    steps = [[("add_ticket_note", {"ticket_id": "T-1", "note": "x"})],
             [("get_order", {"order_id": "O-5001"})]]  # fmt: skip
    agent = Agent(ScriptedModel(steps), box, approver=DenyAll())
    result = agent.run("x")
    assert result.status == "error" and "database is down" in result.reason
    assert env.tickets["T-1"].notes == [] and result.rolled_back
    execute = [s for s in result.tracer.spans if s.kind == "execute_tool"][-1]
    assert execute.status == "error"


def test_the_trace_records_one_span_per_turn_call_and_review() -> None:
    result, _, _ = _run(list(find("damaged-refund").careful), _Recorder(True))
    kinds = [s.kind for s in result.tracer.spans]
    assert kinds.count("invoke_agent") == 1 and kinds.count("chat") == 6
    assert kinds.count("execute_tool") == 6 and kinds.count("approval") == 1
    root = result.tracer.spans[0]
    assert all(s.trace_id == root.trace_id for s in result.tracer.spans)
    approval = next(s for s in result.tracer.spans if s.kind == "approval")
    parent = next(s for s in result.tracer.spans if s.span_id == approval.parent_id)
    assert parent.name == "execute_tool issue_refund"
    assert root.attributes["agent.status"] == "completed"


def test_an_unknown_price_stops_a_budgeted_run_instead_of_ignoring_the_budget() -> None:
    call = ToolCall("t1", "list_open_tickets", {})
    model = _Fixed("tool_use", [call], usage=Usage(input_tokens=10), model="some-other-model")
    agent, _ = _agent(model, max_cost_usd=1.0)
    result = agent.run("x")
    assert result.status == "stopped" and result.turns == 1
    assert "cannot be enforced" in result.reason


def test_only_a_normal_finish_counts_as_completed() -> None:
    for stop, status in [
        ("model_context_window_exceeded", "truncated"),
        ("pause_turn", "error"),
        ("stop_sequence", "completed"),
    ]:
        agent, _ = _agent(_Fixed(stop))
        assert agent.run("x").status == status, stop
