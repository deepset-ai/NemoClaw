"""OpenTelemetry -> OTLP/JSON file tracing for Haystack.

Adapted from `agentic-rag/tracing_setup.py`. Spans are written to disk as
**OTLP/JSON** using the official OpenTelemetry encoder (``encode_spans``), so the
on-disk data is the real OTLP data model — `resourceSpans -> scopeSpans -> spans`,
typed attribute arrays, `unixNano` timestamps. One `ExportTraceServiceRequest`
object is written per line (OTLP/JSON-lines).
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.semconv.resource import ResourceAttributes
from google.protobuf.json_format import MessageToDict


def _b64_to_hex(value: str) -> str:
    return base64.b64decode(value).hex()


def _hexify_ids(otlp: dict[str, Any]) -> None:
    """protobuf-JSON encodes trace/span IDs as base64; the OTLP/JSON spec mandates
    lowercase hex. Rewrite them in place so the output is spec-compliant."""
    for resource_spans in otlp.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                for key in ("traceId", "spanId", "parentSpanId"):
                    if key in span:
                        span[key] = _b64_to_hex(span[key])
                for link in span.get("links", []):
                    for key in ("traceId", "spanId"):
                        if key in link:
                            link[key] = _b64_to_hex(link[key])


class OTLPFileSpanExporter(SpanExporter):
    """Write finished spans to `path` as OTLP/JSON, one request object per line."""

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        request = encode_spans(spans)  # -> ExportTraceServiceRequest (protobuf)
        otlp = MessageToDict(request)  # OTLP/JSON (camelCase, base64 ids)
        _hexify_ids(otlp)              # -> spec-compliant hex ids
        self._fh.write(json.dumps(otlp, ensure_ascii=False) + "\n")
        self._fh.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # noqa: ARG002
        self._fh.flush()
        return True


def setup_tracing(
    trace_file: str | os.PathLike,
    service_name: str = "security-agent",
    resource_attributes: Mapping[str, Any] | None = None,
) -> TracerProvider:
    """Point the global OpenTelemetry provider at `trace_file` (OTLP/JSON) and wire
    Haystack's tracer to it. Call once, at process start, before running the agent.

    `resource_attributes` are merged onto the OTLP Resource, so anything passed here
    (e.g. ``{"security_agent.config_hash": config_hash()}``) appears on every span.
    Returns the provider so the caller can ``force_flush()`` before exit.
    """
    attributes: dict[str, Any] = {ResourceAttributes.SERVICE_NAME: service_name}
    if resource_attributes:
        attributes.update(resource_attributes)
    resource = Resource(attributes=attributes)
    provider = TracerProvider(resource=resource)
    # SimpleSpanProcessor exports on every span end -> nothing lost if the run crashes.
    provider.add_span_processor(SimpleSpanProcessor(OTLPFileSpanExporter(trace_file)))
    trace.set_tracer_provider(provider)

    from haystack import tracing as hs_tracing
    from haystack_integrations.tracing.opentelemetry import OpenTelemetryTracer

    hs_tracing.enable_tracing(OpenTelemetryTracer(trace.get_tracer(service_name)))
    return provider
