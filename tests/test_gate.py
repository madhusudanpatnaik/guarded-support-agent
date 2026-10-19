"""The gate's rules one at a time, each with the case that must pass and the case that must not."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from toolagent.backoffice import BackOffice
from toolagent.grounding import TASK, Ledger, mentions
from toolagent.policy import Decision, Gate, Policy
from toolagent.tools import Effect, Tool, Toolbox


def _gate(task: str = "Handle ticket T-1.", **policy: Any) -> tuple[Gate, BackOffice]:
    env = BackOffice()
    p = Policy(auto_approve=env.auto_rules(task), **policy)
    return Gate(env.toolbox(), p, Ledger(task)), env


REFUND = {"order_id": "O-5001", "amount": 129.0, "reason": "damaged"}


# ------------------------------------------------------------------ grounding


@pytest.mark.parametrize(
    ("text", "value", "found"),
    [
        ("refund order O-5001.", "O-5001", True),  # sentence-final full stop
        ("order O-50011", "O-5001", False),  # a longer id
        ("order XO-5001", "O-5001", False),
        ("write to mary.jane@example.com", "jane@example.com", False),  # a longer address
        ("write to jane@example.com, thanks", "jane@example.com", True),
        ("write to JANE@example.com", "jane@example.com", True),  # case does not matter
        ("ticket T-1", "", False),
    ],
)
def test_mentions_is_whole_token(text: str, value: str, found: bool) -> None:
    assert mentions(text, value) is found


def test_ledger_names_the_first_source_and_finds_values_inside_text() -> None:
    ledger = Ledger("Handle ticket T-1.")
    ledger.record('get_ticket("T-1")', {"customer_id": "C-1001", "body": "my order O-5001 broke"})
    ledger.record('list_orders("C-1001")', [{"order_id": "O-5001", "total": "129.00"}])
    assert ledger.source("T-1") == TASK
    assert ledger.source("C-1001") == 'get_ticket("T-1")'
    assert ledger.source("O-5001") == 'get_ticket("T-1")'  # mentioned in the body first
    assert ledger.source("129.00") == 'list_orders("C-1001")'
    assert ledger.source("o-5001") == 'get_ticket("T-1")'  # case-insensitive, like addresses
    assert ledger.source("O-5010") is None


# ------------------------------------------------------------------ decisions


def test_unknown_tool_and_bad_arguments_are_refused_with_a_reason() -> None:
    gate, _ = _gate()
    v = gate.check("delete_everything", {})
    assert v.decision is Decision.REFUSE and "available: get_ticket" in v.reason
    v = gate.check("set_ticket_status", {"ticket_id": "T-1", "status": "closed"})
    assert v.decision is Decision.REFUSE and v.reason.startswith("invalid arguments: status")
    v = gate.check("get_ticket", {"ticket_id": "T-1", "extra": 1})
    assert v.decision is Decision.REFUSE  # extra fields are not ignored


def test_an_invented_identifier_is_refused_before_anyone_is_asked() -> None:
    gate, _ = _gate()
    v = gate.check("issue_refund", REFUND)
    assert v.decision is Decision.REFUSE and "O-5001" in v.reason and "look it up" in v.reason


def test_irreversible_calls_go_to_a_human_unless_an_auto_rule_covers_them() -> None:
    gate, _ = _gate("Refund O-5001 and O-5002.")
    v = gate.check("issue_refund", REFUND)
    assert v.decision is Decision.ASK and v.sources == {"order_id": TASK}  # over $50
    small = {"order_id": "O-5002", "amount": 36, "reason": "change of mind, unopened"}
    v = gate.check("issue_refund", small)
    assert v.decision is Decision.ALLOW and v.auto_approved


@pytest.mark.parametrize(
    ("task", "refund", "auto"),
    [
        ("Handle ticket T-2.", {"order_id": "O-5002", "amount": 36, "reason": "unopened"}, True),
        # the amount the clause gives, and nothing else
        ("Handle ticket T-2.", {"order_id": "O-5002", "amount": 40, "reason": "unopened"}, False),
        # a reason that names no clause, or both clauses, goes to a person
        ("Handle ticket T-2.", {"order_id": "O-5002", "amount": 36, "reason": "asked"}, False),
        ("Handle ticket T-2.",
         {"order_id": "O-5002", "amount": 40, "reason": "unopened but damaged"}, False),
        # damage: the full price, if the order is in scope
        ("Handle ticket T-2.", {"order_id": "O-5002", "amount": 40, "reason": "damaged"}, True),
        # an order of a customer the task is not about
        ("Handle ticket T-1. Order O-5002.",
         {"order_id": "O-5002", "amount": 36, "reason": "unopened"}, True),
        ("Handle ticket T-1. See O-5002.",
         {"order_id": "O-5005", "amount": 15, "reason": "damaged"}, False),
    ],
)  # fmt: skip
def test_a_refund_runs_unreviewed_only_for_the_policy_amount_in_scope(
    task: str, refund: dict[str, Any], auto: bool
) -> None:
    gate, _ = _gate(task)
    gate.ledger.record("list_orders", [{"order_id": "O-5002"}, {"order_id": "O-5005"}])
    v = gate.check("issue_refund", refund)
    assert v.decision in (Decision.ALLOW, Decision.ASK) and v.auto_approved is auto


def test_an_email_runs_unreviewed_only_to_a_customer_the_task_is_about() -> None:
    gate, env = _gate("Reply to ticket T-6. Customers: jane.doe@example.com is cc'd elsewhere.")
    to_mia = {"to": "mia.chen@example.com", "subject": "s", "body": "b"}
    gate.ledger.record("get_ticket", {"customer_email": "mia.chen@example.com"})
    assert gate.check("send_email", to_mia).auto_approved
    gate2, _ = _gate("Reply to ticket T-6.")
    gate2.ledger.record("list_open_tickets", [{"customer_email": "jane.doe@example.com"}])
    to_jane = {"to": "jane.doe@example.com", "subject": "s", "body": "b"}
    assert gate2.check("send_email", to_jane).decision is Decision.ASK  # on file, not in scope
    assert env.in_scope("Reply to ticket T-6.") == {"C-1003"}


def test_reads_and_writes_run_without_review() -> None:
    gate, _ = _gate()
    assert gate.check("get_order", {"order_id": "O-9999"}).decision is Decision.ALLOW
    v = gate.check("add_ticket_note", {"ticket_id": "T-1", "note": "x"})
    assert v.decision is Decision.ALLOW and not v.auto_approved


def test_an_email_to_an_address_not_on_file_is_asked_even_when_grounded() -> None:
    gate, _ = _gate("Forward the receipt to accounts@partner.example.")
    v = gate.check("send_email", {"to": "accounts@partner.example", "subject": "s", "body": "b"})
    assert v.decision is Decision.ASK


def test_a_declined_call_is_never_asked_again_but_a_different_one_is() -> None:
    gate, _ = _gate("Refund O-5001.")
    v = gate.check("issue_refund", REFUND)
    gate.declined(v)
    again = gate.check("issue_refund", REFUND)
    assert again.decision is Decision.REFUSE and "declined" in again.reason
    other = gate.check("issue_refund", {**REFUND, "amount": 100.0})
    assert other.decision is Decision.ASK


def test_an_irreversible_call_that_ran_is_not_repeated() -> None:
    gate, _ = _gate("Refund O-5001.")
    gate.executed(gate.check("issue_refund", REFUND))
    v = gate.check("issue_refund", {**REFUND, "amount": "129.00"})  # same call, other spelling
    assert v.decision is Decision.REFUSE and "already done" in v.reason


def test_identical_calls_past_the_limit_are_refused_then_the_run_stops() -> None:
    gate, _ = _gate(max_repeats=2, max_refusals=2)
    call = ("get_order", {"order_id": "O-5003"})
    assert [gate.check(*call).decision for _ in range(2)] == [Decision.ALLOW] * 2
    assert gate.check(*call).decision is Decision.REFUSE
    assert gate.check(*call).decision is Decision.REFUSE
    v = gate.check(*call)
    assert v.decision is Decision.STOP and v.reason.startswith("stuck: 3 refused calls")


def test_a_call_that_runs_resets_the_refusal_streak() -> None:
    gate, _ = _gate(max_refusals=1)
    assert gate.check("nope", {}).decision is Decision.REFUSE
    gate.executed(gate.check("get_order", {"order_id": "O-5003"}))
    assert gate.check("nope", {}).decision is Decision.REFUSE  # not STOP: the streak restarted


def test_tool_call_budget_stops_the_run() -> None:
    gate, _ = _gate(max_tool_calls=2)
    for _ in range(2):
        gate.check("list_open_tickets", {})
    v = gate.check("list_open_tickets", {})
    assert v.decision is Decision.STOP and "budget" in v.reason


def test_gate_off_keeps_only_the_schema_check() -> None:
    gate, _ = _gate(enforce=False, max_tool_calls=1)
    for _ in range(3):  # no budget, no grounding, no review, no repeat limit
        assert gate.check("issue_refund", REFUND).decision is Decision.ALLOW
    assert gate.check("issue_refund", {"order_id": "O-1"}).decision is Decision.REFUSE


# ------------------------------------------------------------------ tool definitions


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: str


def test_a_write_tool_must_say_how_to_undo_it() -> None:
    with pytest.raises(ValueError, match="undo"):
        Tool("w", "d", _Args, Effect.WRITE, lambda a: None)


def test_grounded_params_must_exist() -> None:
    with pytest.raises(ValueError, match="grounded"):
        Tool("r", "d", _Args, Effect.IRREVERSIBLE, lambda a: None, grounded=("y",))


def test_tool_names_are_unique_and_schemas_are_strict() -> None:
    t = Tool("r", "d", _Args, Effect.READ, lambda a: None)
    with pytest.raises(ValueError, match="duplicate"):
        Toolbox([t, t])
    schema = t.schema()
    assert schema["strict"] is True and schema["input_schema"]["additionalProperties"] is False
