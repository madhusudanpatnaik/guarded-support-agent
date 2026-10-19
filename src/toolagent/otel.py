"""Replay a run's spans into OpenTelemetry, so they land in Jaeger, Tempo, Langfuse, Phoenix or any
OTLP backend next to the rest of a service's traces.

This module needs only `opentelemetry-api`; the application chooses the SDK and exporter (the
`otel` extra installs both). With the standard environment variables:

    OTEL_SERVICE_NAME=toolagent OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
        toolagent demo --otlp

Spans keep their original start and end times; ids are the backend's own, with the run's trace
id kept as the `agent.trace_id` attribute.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from opentelemetry import trace

from toolagent.trace import Span

_Primitive = str | bool | int | float


def _attribute(value: Any) -> _Primitive | list[_Primitive] | None:
    """OpenTelemetry attributes are primitives or flat lists of them; anything else becomes JSON."""
    if value is None:
        return None
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple) and all(
        isinstance(v, str | bool | int | float) for v in value
    ):
        return list(value)
    return json.dumps(value, default=str)


def export(spans: Sequence[Span], tracer: trace.Tracer) -> int:
    """Re-create `spans` (parents before children, as recorded) under `tracer`; returns how many."""
    made: dict[str, trace.Span] = {}
    for s in spans:
        parent = made.get(s.parent_id) if s.parent_id else None
        attributes = {"agent.trace_id": s.trace_id}
        for k, v in s.attributes.items():
            if (a := _attribute(v)) is not None:
                attributes[k] = a  # type: ignore[assignment]
        otel_span = tracer.start_span(
            s.name,
            context=trace.set_span_in_context(parent) if parent is not None else None,
            start_time=s.start_ns,
            attributes=attributes,
        )
        if s.status == "error":
            otel_span.set_status(trace.Status(trace.StatusCode.ERROR))
        made[s.span_id] = otel_span
    for s in reversed(spans):  # children end before their parents
        made[s.span_id].end(end_time=s.end_ns)
    return len(made)


def export_with_sdk(
    spans: Sequence[Span],
) -> int:  # pragma: no cover - needs the otel extra
    """Export through an OTLP/HTTP exporter configured from the standard OTEL_* variables."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    n = export(spans, provider.get_tracer("toolagent"))
    provider.shutdown()  # flushes
    return n
