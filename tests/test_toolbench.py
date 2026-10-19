"""ToolBench: the file format, ToolBench's naming rules, the StableToolBench request, and scoring.
The fixture follows the published format of `solvable_queries/test_instruction/*.json` (checked
against G1_tool.json); its contents are made up."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from toolagent.models import ScriptedModel
from toolagent.policy import Policy
from toolagent.toolbench import (
    Api,
    VirtualApiServer,
    change_name,
    function_name,
    load_queries,
    run_query,
    standardize,
    toolbox_for,
)
from toolagent.tools import ToolError

SAMPLE = Path(__file__).parent / "fixtures" / "toolbench_sample.json"


def test_load_queries_reads_every_field() -> None:
    weather, joke = load_queries(SAMPLE)
    assert weather.query_id == 101 and joke.query == "Tell me a clean joke."
    assert len(weather.apis) == 3 and weather.apis[1].name == "5 Day Forecast"
    assert weather.apis[1].required[0] == {
        "name": "lat", "type": "NUMBER", "description": "Latitude", "default": "48.85",
    }  # fmt: skip
    assert weather.relevant == (
        ("Simple Weather", "Current Weather"), ("Simple Weather", "5 Day Forecast"),
    )  # fmt: skip


@pytest.mark.parametrize(
    ("raw", "standard"),
    [
        ("Simple Weather", "simple_weather"),
        ("Jokes API!", "jokes_api"),
        ("5 Day Forecast", "get_5_day_forecast"),  # a leading digit gets get_
        ("  a--b__c ", "a_b_c"),
        ("GetUser/{id}", "getuser_id"),
        ("x^y", "x^y"),  # ToolBench's pattern keeps ^
    ],
)
def test_standardize_matches_toolbench(raw: str, standard: str) -> None:
    assert standardize(raw) == standard


def test_change_name_and_function_name() -> None:
    assert [change_name(n) for n in ("from", "id", "fromage")] == ["is_from", "is_id", "fromage"]
    api = Api("Data", "Simple Weather", "Current Weather", "", "GET", (), ())
    assert function_name(api) == "current_weather_for_simple_weather"
    long = Api("Data", "A Very Long Tool Name For Testing Truncation", "Fetch The Full List Of "
               "Every Item", "", "GET", (), ())  # fmt: skip
    name = function_name(long)
    assert len(name) == 64 and name.endswith("_for_a_very_long_tool_name_for_testing_truncation")


def test_tools_keep_real_parameter_names_and_drop_duplicates() -> None:
    weather, _ = load_queries(SAMPLE)
    seen: list[tuple[str, dict[str, Any]]] = []

    def call(api: Api, args: dict[str, Any]) -> str:
        seen.append((api.name, args))
        return "ok"

    box, by_name = toolbox_for(weather, call)
    assert sorted(by_name) == ["current_weather_for_simple_weather",
                               "get_5_day_forecast_for_simple_weather"]  # fmt: skip
    forecast = box["get_5_day_forecast_for_simple_weather"]
    schema = forecast.schema()["input_schema"]
    assert sorted(schema["properties"]) == ["from", "lat", "lon"]  # `from` is a keyword in Python
    assert schema["required"] == ["lat", "lon"]
    assert "(example: 48.85)" in schema["properties"]["lat"]["description"]
    forecast.run(forecast.parse({"lat": 48.85, "lon": 2, "from": 1}))
    current = box["current_weather_for_simple_weather"]
    current.run(current.parse({"city": "Paris"}))
    assert seen == [
        ("5 Day Forecast", {"lat": 48.85, "lon": 2, "from": 1}),  # integers stay integers
        ("Current Weather", {"city": "Paris"}),  # unset optionals are not sent
    ]


def _server(handler: Any) -> tuple[VirtualApiServer, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append({"body": json.loads(request.content), "key": request.headers["toolbench_key"]})
        return handler(request)  # type: ignore[no-any-return]

    client = httpx.Client(transport=httpx.MockTransport(record))
    return VirtualApiServer("http://server/virtual", key="k1", client=client), sent


def test_the_server_request_matches_stabletoolbench() -> None:
    server, sent = _server(lambda r: httpx.Response(200, json={"error": "", "response": "sunny"}))
    api = Api("Data", "Simple Weather", "5 Day Forecast", "", "GET", (), ())
    assert server(api, {"lat": 1}) == "sunny"
    assert sent == [{
        "body": {"category": "Data", "tool_name": "simple_weather",
                 "api_name": "get_5_day_forecast", "tool_input": {"lat": 1},
                 "strip": "truncate", "toolbench_key": "k1"},
        "key": "k1",
    }]  # fmt: skip


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"error": "API not working error...", "response": ""}),
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
    ],
)
def test_server_failures_become_tool_errors(response: httpx.Response) -> None:
    server, _ = _server(lambda r: response)
    with pytest.raises(ToolError):
        server(Api("Data", "t", "a", "", "GET", (), ()), {})


def test_scoring_recall_and_give_up() -> None:
    weather, joke = load_queries(SAMPLE)
    model = ScriptedModel([[("current_weather_for_simple_weather", {"city": "Paris"})],
                           "It is sunny in Paris."])  # fmt: skip
    outcome, _ = run_query(weather, model, lambda api, args: {"temp": 21})
    assert outcome.answered and outcome.relevant_recall == 0.5 and outcome.turns == 2
    gave_up, _ = run_query(joke, ScriptedModel(["GIVE UP: the joke API is down."]), lambda a, b: "")
    assert not gave_up.answered and gave_up.relevant_recall == 0.0


def test_api_errors_count_as_tool_errors_not_refusals() -> None:
    _, joke = load_queries(SAMPLE)

    def down(api: Api, args: dict[str, Any]) -> Any:
        raise ToolError("API not working error...")

    model = ScriptedModel(
        [[("get_joke_for_jokes_api", {})], [("no_such_api", {})], "GIVE UP: down"]
    )
    outcome, result = run_query(joke, model, down, Policy(max_turns=5))
    assert (outcome.tool_errors, outcome.refused) == (1, 1)
    assert result.calls[0].observation == "Error: API not working error..."


# ------------------------------------------------------------------ Solvable Pass Rate


class _ScriptedJudge:
    def __init__(self, *statuses: str) -> None:
        self.statuses = list(statuses)
        self.asked: list[str] = []

    def check_is_solved(self, query: Any, answer: str) -> Any:
        self.asked.append(answer)
        return self.statuses.pop(0)


def _result(status: str, answer: str) -> Any:
    from toolagent.agent import RunResult
    from toolagent.models import Usage
    from toolagent.trace import Tracer

    return RunResult(status, "", answer, 1, [], Usage(), 0.0, [], [], Tracer())  # type: ignore[arg-type]


def test_pass_rate_follows_stabletoolbench_scoring() -> None:
    from toolagent.toolbench import solvable_pass_rate

    weather, joke = load_queries(SAMPLE)
    results = [(weather, _result("completed", "Sunny, 21C")), (joke, _result("stopped", ""))]
    judge = _ScriptedJudge("Solved", "Unsure", "Solved", "Unsolved")
    pr = solvable_pass_rate(results, judge, evaluate_times=4)
    # the unfinished run is Unsolved in every evaluation run, and the judge never sees it
    assert judge.asked == ["Sunny, 21C"] * 4
    assert pr.statuses["102"] == ["Unsolved"] * 4
    # per run: (1 + 0) / 2, (0.5 + 0) / 2, (1 + 0) / 2, (0 + 0) / 2
    assert pr.run_scores == [0.5, 0.25, 0.5, 0.0]
    assert (pr.pass_rate, pr.std) == (31.2, 20.7)  # mean 0.3125, population std 0.2073
    with pytest.raises(ValueError):
        solvable_pass_rate([], judge)


def test_claude_judge_asks_for_a_schema_constrained_verdict() -> None:
    import anthropic
    import httpx2

    from toolagent.toolbench import ClaudeJudge, JudgeError

    sent: list[dict[str, Any]] = []
    solved = '{"reason": "gives the weather and forecast", "status": "Solved"}'
    replies = iter([{"text": solved, "stop": "end_turn"}, {"text": "", "stop": "refusal"}])

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        r = next(replies)
        content = [{"type": "text", "text": r["text"]}] if r["text"] else []
        return httpx2.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": r["stop"], "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })  # fmt: skip

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    client = anthropic.Anthropic(api_key="k", http_client=http, max_retries=0)
    weather, _ = load_queries(SAMPLE)
    judge = ClaudeJudge(client)
    assert judge.check_is_solved(weather, "Sunny, 21C; then rain") == "Solved"
    body = sent[0]
    assert body["output_config"]["format"]["schema"]["properties"]["status"]["enum"] == [
        "Solved", "Unsure", "Unsolved",
    ]  # fmt: skip
    assert "Simple Weather / Current Weather" in body["messages"][0]["content"]
    with pytest.raises(JudgeError, match="refusal"):
        judge.check_is_solved(weather, "x")
