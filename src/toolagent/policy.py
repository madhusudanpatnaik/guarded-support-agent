"""The gate: every tool call the model asks for passes through `Gate.check` before anything runs.

Checks, in order — the first that fires decides:
  1. tool-call budget spent                      -> STOP
  2. unknown tool, or arguments fail the schema  -> REFUSE (the model can repair them)
  3. this exact call was declined by a human     -> REFUSE (never ask twice)
  4. this exact irreversible call already ran    -> REFUSE (no double refunds)
  5. this exact call repeated past the limit     -> REFUSE (the agent is looping)
  6. a grounded argument appears nowhere         -> REFUSE (an invented id or recipient)
  7. READ or WRITE                               -> ALLOW (WRITEs are journaled for rollback)
  8. IRREVERSIBLE with a matching auto rule      -> ALLOW, marked auto-approved
  9. IRREVERSIBLE                                -> ASK a human
Too many refusals in a row turns the next refusal into STOP: an agent that keeps proposing
refused calls is stuck, and more turns only cost money. `Policy(enforce=False)` keeps only the
schema check, for measuring what the rest is worth.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ValidationError

from toolagent.grounding import Ledger, canonical
from toolagent.tools import Effect, Toolbox, validation_message


@dataclass(frozen=True)
class Policy:
    max_turns: int = 12  # model calls per run
    max_tool_calls: int = 30
    max_cost_usd: float | None = 1.0  # checked before each model call
    max_repeats: int = 2  # identical calls allowed; the next one is refused
    max_refusals: int = 4  # refusals in a row before the run stops
    max_observation_chars: int = 4000  # longer tool output is cut before the model sees it
    rollback_on_abort: bool = True  # undo journaled WRITEs when a run does not complete
    enforce: bool = True  # False: run every schema-valid call (the no-gate ablation)
    # tool name -> rule; an irreversible call the rule accepts runs without a human
    auto_approve: Mapping[str, Callable[[BaseModel], bool]] = field(default_factory=dict)


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    REFUSE = "refuse"
    STOP = "stop"


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str
    key: str = ""  # the call, canonicalized: tool name + sorted JSON arguments
    args: BaseModel | None = None
    sources: Mapping[str, str] = field(default_factory=dict)  # grounded param -> where it came from
    auto_approved: bool = False
    rule: str = ""  # which check decided: a short code, safe to record when content is not


class Gate:
    def __init__(self, toolbox: Toolbox, policy: Policy, ledger: Ledger) -> None:
        self.toolbox = toolbox
        self.policy = policy
        self.ledger = ledger
        self.tool_calls = 0
        self._seen: Counter[str] = Counter()
        self._executed_irreversible: set[str] = set()
        self._declined: set[str] = set()
        self._refusals_in_a_row = 0

    def check(self, name: str, raw_args: object) -> Verdict:
        self.tool_calls += 1
        if self.policy.enforce and self.tool_calls > self.policy.max_tool_calls:
            return Verdict(
                Decision.STOP,
                f"tool-call budget of {self.policy.max_tool_calls} spent",
                rule="tool_budget",
            )
        verdict = self._check(name, raw_args)
        if verdict.decision is Decision.REFUSE:
            self._refusals_in_a_row += 1
            if self.policy.enforce and self._refusals_in_a_row > self.policy.max_refusals:
                why = f"stuck: {self._refusals_in_a_row} refused calls in a row ({verdict.reason})"
                return Verdict(Decision.STOP, why, verdict.key, verdict.args, rule="stuck")
        return verdict

    def _check(self, name: str, raw_args: object) -> Verdict:
        if name not in self.toolbox:
            known = ", ".join(t.name for t in self.toolbox)
            return Verdict(
                Decision.REFUSE, f"unknown tool {name!r}; available: {known}", rule="unknown_tool"
            )
        tool = self.toolbox[name]
        try:
            args = tool.parse(raw_args)
        except ValidationError as e:
            return Verdict(
                Decision.REFUSE, f"invalid arguments: {validation_message(e)}", rule="invalid_args"
            )
        key = f"{name} {json.dumps(canonical(args.model_dump()), sort_keys=True, default=str)}"
        if not self.policy.enforce:
            return Verdict(Decision.ALLOW, "gate off", key, args, rule="gate_off")

        if key in self._declined:
            return Verdict(
                Decision.REFUSE,
                "a human already declined this exact action",
                key,
                args,
                rule="declined_before",
            )
        if tool.effect is Effect.IRREVERSIBLE and key in self._executed_irreversible:
            return Verdict(
                Decision.REFUSE, "already done; not repeating it", key, args, rule="already_done"
            )
        self._seen[key] += 1
        if self._seen[key] > self.policy.max_repeats:
            why = f"same call made {self._seen[key] - 1} times already; use the earlier result"
            return Verdict(Decision.REFUSE, why, key, args, rule="repeat")

        sources: dict[str, str] = {}
        for param in tool.grounded:
            value = getattr(args, param)
            source = self.ledger.source(value)
            if source is None:
                why = (
                    f"{param}={value!r} does not appear in the task or in any tool result; "
                    "look it up instead of supplying it"
                )
                return Verdict(Decision.REFUSE, why, key, args, rule="ungrounded")
            sources[param] = source

        if tool.effect is not Effect.IRREVERSIBLE:
            return Verdict(
                Decision.ALLOW, tool.effect.value, key, args, sources, rule=tool.effect.value
            )
        rule = self.policy.auto_approve.get(name)
        if rule is not None and rule(args):
            return Verdict(
                Decision.ALLOW, "auto-approval rule", key, args, sources, True, rule="auto_rule"
            )
        return Verdict(
            Decision.ASK, "irreversible: needs a human", key, args, sources, rule="needs_human"
        )

    # the agent reports what happened, so the next check knows

    def executed(self, verdict: Verdict) -> None:
        self._refusals_in_a_row = 0
        tool = self.toolbox[verdict.key.split(" ", 1)[0]]
        if tool.effect is Effect.IRREVERSIBLE:
            self._executed_irreversible.add(verdict.key)

    def declined(self, verdict: Verdict) -> None:
        self._declined.add(verdict.key)
        self._refusals_in_a_row += 1
