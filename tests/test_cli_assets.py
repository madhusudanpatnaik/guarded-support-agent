"""The command line, the download command (with a fake fetch: nothing leaves this machine in tests),
and the console reviewer."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from toolagent import assets
from toolagent.approval import ApprovalRequest, ConsoleApprover
from toolagent.backoffice import Refund
from toolagent.cli import main
from toolagent.tools import Effect


def test_demo_prints_the_run_and_writes_a_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace = tmp_path / "demo.jsonl"
    assert main(["demo", "--task", "damaged-refund", "--trajectory", "sloppy",
                 "--trace", str(trace)]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "(the sloppy script refunds an order id it made up" in out
    assert "issue_refund" in out and "-> refuse, not run" in out
    assert "-> ask [informed: yes], ran" in out
    assert "scored: success=True violations=0" in out
    assert main(["trace", str(trace)]) == 0
    assert "status completed; 7 turns, 6 tool calls" in capsys.readouterr().out


def test_eval_scripted_prints_the_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--scripted", "--tasks", "damaged-refund,tracking-loop"]) == 0
    out = capsys.readouterr().out
    rows = [ln for ln in out.splitlines() if ln.startswith(("careful", "sloppy"))]
    assert len(rows) == 8 and "test the harness, not a model" in out


def test_download_fetches_once_and_reports_a_digest(tmp_path: Path) -> None:
    fetched: list[str] = []

    def fake(url: str) -> bytes:
        fetched.append(url)
        return b"[]"

    lines = assets.download(["G1_tool"], tmp_path, fetch=fake)
    assert fetched == [assets.QUERY_URL.format(group="G1_tool")]
    assert (tmp_path / "G1_tool.json").read_bytes() == b"[]"
    assert lines[0].startswith(f"G1_tool: {tmp_path / 'G1_tool.json'} (sha256 4f53cda18c2baa0c")
    assets.download(["G1_tool"], tmp_path, fetch=fake)
    assert len(fetched) == 1  # already there: not fetched again
    with pytest.raises(ValueError, match="unknown group"):
        assets.download(["G9"], tmp_path, fetch=fake)


def test_download_list_names_every_group(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["download", "--list"]) == 0
    out = capsys.readouterr().out
    assert all(g in out for g in assets.GROUPS) and "raw.githubusercontent.com" in out


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        "Handle ticket T-1.",
        "issue_refund",
        Effect.IRREVERSIBLE,
        {"order_id": "O-5001", "amount": "129.00", "reason": "damaged"},
        {"order_id": 'get_ticket("T-1")'},
        "The jug is cracked; policy says full refund.",
    )


@pytest.mark.parametrize(
    ("typed", "approved", "note"),
    [("y\n", True, ""), ("YES\n", True, ""), ("\nnot yet\n", False, "not yet"),
     ("sure\nwrong order\n", False, "wrong order")],
)  # fmt: skip
def test_console_reviewer_approves_only_on_an_explicit_yes(
    typed: str, approved: bool, note: str
) -> None:
    out = io.StringIO()
    decision = ConsoleApprover(io.StringIO(typed), out).review(_request())
    assert (decision.approved, decision.note) == (approved, note)
    shown = out.getvalue()
    assert 'order_id = "O-5001"   <- from get_ticket("T-1")' in shown
    assert "why (model): The jug is cracked" in shown


def test_refund_amounts_are_in_cents() -> None:
    # validated from JSON-shaped arguments, as a model's tool call arrives
    assert str(Refund.model_validate({"order_id": "O-1", "amount": 36, "reason": "x"}).amount) == (
        "36.00"
    )
    with pytest.raises(ValidationError, match="two decimal places"):
        Refund.model_validate({"order_id": "O-1", "amount": "36.005", "reason": "x"})


def test_live_eval_passes_its_limits_and_reports_pass_hat_k(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from toolagent import cli, evaluate
    from toolagent.models import ScriptedModel
    from toolagent.tasks import find

    seen: list[float | None] = []
    real = evaluate.run_task

    def spy(task: Any, model: Any, **kw: Any) -> Any:
        seen.append(kw["max_cost_usd"])
        return real(task, ScriptedModel(find(task.id).careful), **kw)

    monkeypatch.setattr(cli, "run_task", spy)
    monkeypatch.setattr(cli, "_claude", lambda args: None)
    assert main(["eval", "--tasks", "order-status", "--trials", "2", "--max-cost", "0.25"]) == 0
    assert seen == [0.25, 0.25]
    out = capsys.readouterr().out
    assert "pass^k: k=1 1.00, k=2 1.00" in out


def test_run_with_nobody_on_call_declines_what_needs_review(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from toolagent import cli
    from toolagent.models import ScriptedModel
    from toolagent.tasks import find

    model = ScriptedModel(find("damaged-refund").careful)
    monkeypatch.setattr(cli, "_claude", lambda args: model)
    code = main(["run", "Handle ticket T-1.", "--reviewer", "unattended"])
    out = capsys.readouterr().out
    assert code == 0 and "[unattended: no], not run" in out
    assert "effects: email:jane.doe@example.com" in out  # the refund was not made


def test_a_missing_optional_library_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "tau2", None)
    monkeypatch.setitem(sys.modules, "tau2.registry", None)
    assert main(["tau2", "run", "--domain", "airline"]) == 2
    assert "pip install -r requirements-tau2.txt" in capsys.readouterr().err


def test_missing_credentials_are_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from toolagent import toolbench

    class _NoKeyJudge:
        def __init__(self, **kw: Any) -> None:
            pass

        def check_is_solved(self, query: Any, answer: str) -> Any:
            raise TypeError("Could not resolve authentication method. Expected one of api_key")

    queries = Path(__file__).parent / "fixtures" / "toolbench_sample.json"
    monkeypatch.setattr(toolbench, "ClaudeJudge", _NoKeyJudge)
    monkeypatch.setattr(toolbench, "run_query", _answered)
    monkeypatch.setattr("toolagent.cli._claude", lambda args: None)
    assert main(["toolbench", "--queries", str(queries), "--judge"]) == 2
    assert "set ANTHROPIC_API_KEY" in capsys.readouterr().err


def _answered(query: Any, model: Any, call: Any, policy: Any, capture: Any) -> Any:
    from toolagent.agent import RunResult
    from toolagent.models import Usage
    from toolagent.toolbench import QueryOutcome
    from toolagent.trace import Tracer

    outcome = QueryOutcome(query.query_id, "completed", True, None, 1, 0, 0, 0, 0.0)
    return outcome, RunResult("completed", "", "an answer", 1, [], Usage(), 0.0, [], [], Tracer())
