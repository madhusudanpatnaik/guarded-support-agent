"""ToolBench through StableToolBench: the agent on real multi-tool instructions.

ToolBench (Qin et al., 2023) pairs thousands of RapidAPI tools with instructions that need one
tool (G1), several tools of one category (G2) or of one collection (G3). The live APIs are
unreliable, so StableToolBench serves them from a response cache backed by a simulator behind one
endpoint, `POST /virtual`. This module reads StableToolBench's solvable test queries
(`solvable_queries/test_instruction/G1_instruction.json`, ...), turns each query's `api_list` into
tools named the way ToolBench names them, and calls that server.

What ToolBench measures here is planning and tool choice across unfamiliar APIs. It does not
exercise the guardrails: the simulator has no side effects, so every tool is a READ. The official
metric, Solvable Pass Rate, is computed by StableToolBench's procedure with Claude as the judge
(`solvable_pass_rate`, `ClaudeJudge`); next to it, the answer rate (finished with an answer, not
stopped and not giving up), recall of the APIs the annotators marked as relevant, and the harness
numbers (turns, refusals, tokens, cost).
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, create_model

from toolagent.agent import Agent, RunResult
from toolagent.approval import DenyAll
from toolagent.models import DEFAULT_MODEL, Model
from toolagent.policy import Decision, Policy
from toolagent.tools import Effect, Tool, Toolbox, ToolError

DEFAULT_SERVER = "http://localhost:8080/virtual"
GIVE_UP = "GIVE UP:"

SYSTEM = f"""\
You complete the user's request with the APIs provided as tools. Call them as needed; a result can
be empty, partial or an error, so try another API or other arguments before concluding. When you
have what the user asked for, reply with the final answer for them. If the request cannot be
completed with these tools, reply starting with "{GIVE_UP}" and say why."""


@dataclass(frozen=True)
class Api:
    category: str
    tool: str
    name: str
    description: str
    method: str
    required: tuple[dict[str, Any], ...]
    optional: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Query:
    query_id: int | str
    query: str
    apis: tuple[Api, ...]
    relevant: tuple[tuple[str, str], ...]  # (tool_name, api_name) the annotators marked as needed


def load_queries(path: str | Path) -> list[Query]:
    """A StableToolBench/ToolBench test file: a JSON list of {query, query_id, api_list,
    "relevant APIs"} with each api_list entry carrying its full documentation."""
    out = []
    for row in json.loads(Path(path).read_text()):
        apis = tuple(
            Api(
                category=a["category_name"],
                tool=a["tool_name"],
                name=a["api_name"],
                description=a.get("api_description") or "",
                method=a.get("method") or "GET",
                required=tuple(a.get("required_parameters") or ()),
                optional=tuple(a.get("optional_parameters") or ()),
            )
            for a in row["api_list"]
        )
        relevant = tuple((str(t), str(n)) for t, n in row.get("relevant APIs", []))
        out.append(Query(row["query_id"], row["query"], apis, relevant))
    return out


# ------------------------------------------------------------------ ToolBench's naming


def standardize(name: str) -> str:
    """ToolBench's `standardize`: anything but a letter, digit, underscore or CJK character becomes
    `_` (its pattern also keeps `^`), runs of `_` collapse, lowercased, `_` stripped from both
    ends, `get_` prefixed when it starts with a digit. The server applies the same function to
    find a tool's cache, so this must match it exactly."""
    s = re.sub(r"[^一-龥^a-zA-Z0-9_]", "_", name)
    s = re.sub(r"_{2,}", "_", s).lower().strip("_")
    if s and s[0].isdigit():
        s = "get_" + s
    return s


def change_name(name: str) -> str:
    """ToolBench's `change_name`: prefix reserved-looking words with `is_`."""
    return (
        "is_" + name if name in {"from", "class", "return", "false", "true", "id", "and"} else name
    )


def function_name(api: Api) -> str:
    """`{api}_for_{tool}`, keeping the *last* 64 characters, as ToolBench does."""
    return f"{change_name(standardize(api.name))}_for_{standardize(api.tool)}"[-64:]


_VALID_NAME = re.compile(r"[^a-zA-Z0-9_-]")


def _params_model(api: Api) -> type[BaseModel]:
    """A pydantic model for the API's parameters. Field names are positional with the API's own
    names as aliases: real parameter names include things like `from`, `json` and `q[]`."""
    kinds: dict[str, Any] = {"NUMBER": int | float, "BOOLEAN": bool}
    fields: dict[str, Any] = {}
    for i, (p, required) in enumerate(
        [(p, True) for p in api.required] + [(p, False) for p in api.optional]
    ):
        kind = kinds.get(str(p.get("type", "STRING")).upper(), str)
        description = str(p.get("description") or "")[:256]
        if p.get("default") not in (None, ""):
            description = f"{description} (example: {p['default']})".strip()
        alias = str(p["name"])
        if required:
            fields[f"p{i}"] = (kind, Field(alias=alias, description=description))
        else:
            fields[f"p{i}"] = (
                kind | None,
                Field(None, alias=alias, description=description),
            )
    config = ConfigDict(extra="forbid", serialize_by_alias=True, validate_by_name=False)
    return create_model("Params", __config__=config, **fields)


Call = Callable[[Api, dict[str, Any]], Any]


def toolbox_for(query: Query, call: Call) -> tuple[Toolbox, dict[str, Api]]:
    """The query's APIs as tools, and the tool name -> API map for scoring. Duplicate APIs (the
    same function name twice) keep the first."""
    tools: list[Tool] = []
    by_name: dict[str, Api] = {}
    for api in query.apis:
        name = _VALID_NAME.sub("_", function_name(api))  # the Messages API's name alphabet
        if name in by_name:
            continue
        by_name[name] = api
        description = (
            f'API "{api.name}" of the tool "{api.tool}" (category {api.category}). '
            f"{api.description[:512]}"
        ).strip()

        def run(args: BaseModel, api: Api = api) -> Any:
            return call(api, args.model_dump(mode="json", exclude_none=True))

        tools.append(Tool(name, description, _params_model(api), Effect.READ, run))
    return Toolbox(tools), by_name


# ------------------------------------------------------------------ the server


class VirtualApiServer:
    """A StableToolBench server: `POST {url}` with the tool's category, standardized tool and API
    names and the arguments; the reply is {"error": str, "response": ...}."""

    def __init__(
        self,
        url: str = DEFAULT_SERVER,
        *,
        key: str = "",
        strip: str = "truncate",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.url, self.key, self.strip = url, key, strip
        self.client = client or httpx.Client(timeout=timeout)

    def __call__(self, api: Api, args: dict[str, Any]) -> Any:
        payload = {
            "category": api.category,
            "tool_name": standardize(api.tool),
            "api_name": change_name(standardize(api.name)),
            "tool_input": args,
            "strip": self.strip,
            "toolbench_key": self.key,
        }
        try:
            r = self.client.post(self.url, json=payload, headers={"toolbench_key": self.key})
            r.raise_for_status()
            body = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise ToolError(f"API server error: {type(e).__name__}: {e}") from e
        if body.get("error"):
            raise ToolError(str(body["error"]))
        return body.get("response", "")


# ------------------------------------------------------------------ running and scoring


@dataclass(frozen=True)
class QueryOutcome:
    query_id: int | str
    status: str
    answered: bool  # completed with an answer that is not a give-up
    relevant_recall: float | None  # None when the query lists no relevant APIs
    turns: int
    tool_calls: int
    tool_errors: int
    refused: int
    cost_usd: float | None


def _key(tool: str, api: str) -> tuple[str, str]:
    return (standardize(tool), standardize(api))


def score(query: Query, result: RunResult, by_name: dict[str, Api]) -> QueryOutcome:
    called = {
        _key(by_name[c.tool].tool, by_name[c.tool].name)
        for c in result.calls
        if c.executed and c.tool in by_name
    }
    relevant = {_key(t, n) for t, n in query.relevant}
    recall = len(called & relevant) / len(relevant) if relevant else None
    return QueryOutcome(
        query_id=query.query_id,
        status=result.status,
        answered=result.status == "completed"
        and bool(result.answer.strip())
        and not result.answer.lstrip().upper().startswith(GIVE_UP),
        relevant_recall=recall,
        turns=result.turns,
        tool_calls=len(result.calls),
        tool_errors=sum(1 for c in result.calls if c.decision is Decision.ALLOW and c.error),
        refused=sum(1 for c in result.calls if c.decision is Decision.REFUSE),
        cost_usd=result.cost_usd,
    )


def run_query(
    query: Query,
    model: Model,
    call: Call,
    policy: Policy | None = None,
    capture_content: bool | None = None,
) -> tuple[QueryOutcome, RunResult]:
    toolbox, by_name = toolbox_for(query, call)
    agent = Agent(
        model,
        toolbox,
        approver=DenyAll(),
        policy=policy or Policy(),
        system=SYSTEM,
        capture_content=capture_content,
    )
    result = agent.run(query.query)
    return score(query, result, by_name), result


def summary(outcomes: Sequence[QueryOutcome]) -> dict[str, float | int | None]:
    n = len(outcomes)
    recalls = [o.relevant_recall for o in outcomes if o.relevant_recall is not None]
    costs = [o.cost_usd for o in outcomes]
    return {
        "queries": n,
        "answer_rate": sum(o.answered for o in outcomes) / n if n else 0.0,
        "relevant_api_recall": sum(recalls) / len(recalls) if recalls else None,
        "mean_turns": sum(o.turns for o in outcomes) / n if n else 0.0,
        "cost_usd": None if any(c is None for c in costs) else sum(c for c in costs if c),
    }


# ------------------------------------------------------------------ Solvable Pass Rate

AnswerStatus = Literal["Solved", "Unsure", "Unsolved"]
_POINTS: dict[str, float] = {"Solved": 1.0, "Unsure": 0.5, "Unsolved": 0.0}


class Judge(Protocol):
    def check_is_solved(self, query: Query, answer: str) -> AnswerStatus: ...


@dataclass(frozen=True)
class PassRate:
    """StableToolBench's Solvable Pass Rate: each evaluation run scores every query (Solved 1,
    Unsure 0.5, Unsolved 0) and averages them; the result is the mean over runs, in percent,
    with the population standard deviation across runs."""

    pass_rate: float
    std: float
    run_scores: list[float]
    statuses: dict[str, list[str]]  # query id -> one status per evaluation run


def solvable_pass_rate(
    results: Sequence[tuple[Query, RunResult]], judge: Judge, evaluate_times: int = 4
) -> PassRate:
    """StableToolBench's procedure (`eval_pass_rate.py`): a run that did not finish with an
    answer is Unsolved without asking the judge; every other answer is judged once per
    evaluation run (4 by default), each run scored on its own. The judge's prompt here is this
    project's, not StableToolBench's, so the number is SoPR-style rather than identical."""
    if not results or evaluate_times < 1:
        raise ValueError("need at least one result and one evaluation run")
    statuses: dict[str, list[str]] = {}
    for query, result in results:
        finished = result.status == "completed" and bool(result.answer.strip())
        statuses[str(query.query_id)] = [
            judge.check_is_solved(query, result.answer) if finished else "Unsolved"
            for _ in range(evaluate_times)
        ]
    run_scores = [
        sum(_POINTS[s[i]] for s in statuses.values()) / len(statuses) for i in range(evaluate_times)
    ]
    return PassRate(
        pass_rate=round(statistics.fmean(run_scores) * 100, 1),
        std=round(statistics.pstdev(run_scores) * 100, 1),
        run_scores=run_scores,
        statuses=statuses,
    )


class _Verdict(BaseModel):
    reason: str
    status: Literal["Solved", "Unsure", "Unsolved"]


JUDGE_SYSTEM = """\
You grade whether an assistant's final answer solves a user's request that needed external APIs.
Reply with a short reason and one status:
- Solved: the answer delivers what was asked — every part of the request, with concrete
  information — rather than a promise, a refusal, or instructions for the user to do it.
- Unsolved: the answer gives up, refuses, says it could not, leaves out part of the request, or
  holds only placeholders.
- Unsure: from the answer alone you cannot tell whether it is complete and real.
Judge what the answer contains against what the request asks; ignore style."""


class JudgeError(RuntimeError):
    pass


class ClaudeJudge:
    """The judge on Claude, its verdict constrained to the three statuses by a JSON schema."""

    def __init__(
        self, client: Any | None = None, *, model: str = DEFAULT_MODEL, effort: str = "low"
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client: Any = client
        self.model, self.effort = model, effort

    def check_is_solved(self, query: Query, answer: str) -> AnswerStatus:
        import anthropic

        apis = "\n".join(f"- {a.tool} / {a.name}: {a.description[:200]}" for a in query.apis)
        prompt = f"Request:\n{query.query}\n\nAPIs available:\n{apis}\n\nFinal answer:\n{answer}"
        r = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": anthropic.transform_schema(_Verdict)},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if r.stop_reason in ("refusal", "max_tokens"):
            raise JudgeError(f"judge stopped with {r.stop_reason} on query {query.query_id}")
        text = "".join(b.text for b in r.content if b.type == "text")
        return _Verdict.model_validate_json(text).status
