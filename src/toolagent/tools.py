"""Tools the agent can call, each labelled with what calling it does to the world.

The label is the contract the gate enforces: a READ changes nothing, a WRITE changes something the
agent can undo (the tool supplies the undo), an IRREVERSIBLE call moves money, sends a message or
destroys data and goes to a human unless a narrow auto-approval rule covers it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError


class Effect(StrEnum):
    READ = "read"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


class ToolError(Exception):
    """A refusal by the tool itself (not found, not allowed by the business rules). The message goes
    back to the model as an error result so it can correct course."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: type[BaseModel]
    effect: Effect
    run: Callable[[Any], Any]
    # params whose value must already appear in the task or an earlier tool result: identifiers and
    # recipients, which a model can otherwise invent
    grounded: tuple[str, ...] = ()
    # WRITE tools: undo(args, result) reverses the call; required so a stopped run can roll back
    undo: Callable[[Any, Any], None] | None = None

    def __post_init__(self) -> None:
        if self.effect is Effect.WRITE and self.undo is None:
            raise ValueError(f"{self.name}: a WRITE tool must say how to undo it")
        unknown = set(self.grounded) - set(self.params.model_fields)
        if unknown:
            raise ValueError(f"{self.name}: grounded params {sorted(unknown)} are not params")

    def schema(self) -> dict[str, Any]:
        """The tool as the Messages API takes it. `strict` makes the API guarantee schema-valid
        input; the arguments are still validated here, because other models make no such promise."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": anthropic.transform_schema(self.params),
            "strict": True,
        }

    def parse(self, args: object) -> BaseModel:
        return self.params.model_validate(args)


class Toolbox:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            if t.name in self._tools:
                raise ValueError(f"duplicate tool name {t.name!r}")
            self._tools[t.name] = t

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self) -> Any:
        return iter(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]


def validation_message(e: ValidationError) -> str:
    """A short, model-readable account of what was wrong with the arguments."""
    parts = []
    for err in e.errors():
        where = ".".join(str(p) for p in err["loc"]) or "(arguments)"
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)
