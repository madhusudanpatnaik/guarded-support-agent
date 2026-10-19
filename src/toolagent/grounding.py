"""Where an argument value came from.

Identifiers and recipients are the arguments a model gets wrong by inventing them: an order id
that looks right, an address with a plausible domain. A tool marks those params as `grounded`, and
the gate refuses a value that appears neither in the task nor in anything a tool returned. The
source it finds is also what a human reviewer is shown: "order_id O-5001, from get_ticket".

This is a check on *existence*, not on intent: a value the agent saw for an unrelated reason
passes. That is why irreversible calls still go to a person.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

TASK = "task"


def _norm(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float | Decimal):
        try:
            return format(Decimal(str(value)).normalize(), "f")
        except InvalidOperation:
            return str(value)
    return str(value).strip().casefold()


def _scalars(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _scalars(v)]
    if isinstance(obj, list | tuple):
        return [s for v in obj for s in _scalars(v)]
    return [] if obj is None else [obj]


def mentions(text: str, value: Any) -> bool:
    """`value` occurs in `text` as a whole token: O-5001 is in "order O-5001." but not in
    "O-50011", and jane@example.com is not in "mary.jane@example.com"."""
    needle = str(value).strip()
    if not needle:
        return False
    pattern = rf"(?<![\w@.-]){re.escape(needle)}(?![\w@-]|\.\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


class Ledger:
    """Everything the agent has been shown, so each grounded value can be traced to a source."""

    def __init__(self, task: str) -> None:
        self._task = task
        # per tool result, in order: its scalars (normalized) and its strings with spaces in them
        self._records: list[tuple[str, set[str], list[str]]] = []

    def record(self, source: str, output: Any) -> None:
        scalars = _scalars(output)
        texts = [v for v in scalars if isinstance(v, str) and " " in v]
        self._records.append((source, {_norm(v) for v in scalars}, texts))

    def source(self, value: Any) -> str | None:
        """The first place `value` appeared — the task, then tool results in the order they
        came back, as a whole value or inside text; None if nowhere."""
        if mentions(self._task, value):
            return TASK
        key = _norm(value)
        for source, values, texts in self._records:
            if key in values or any(mentions(t, value) for t in texts):
                return source
        return None


def canonical(value: Any) -> Any:
    """The value with numbers written one way, so 129, 129.0 and "129.00" for a Decimal field
    compare equal when calls are matched."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [canonical(v) for v in value]
    if isinstance(value, Decimal | float) and not isinstance(value, bool):
        return _norm(value)
    return value
