"""Reconstruct chat conversations from the OTLP JSONL trace exporter.

Completed Haystack Agent runs contain their lossless message history on the
``haystack.agent.run`` span. With ``--include-incomplete``, the latest exported
LLM step is also used to recover conversations whose root span was never
written (for example, when an evaluation process was interrupted).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any, Iterable, Iterator

ROOT_SPAN = "haystack.agent.run"
LLM_SPAN = "haystack.agent.step.llm"
ROOT_OUTPUT = "haystack.agent.output"
LLM_INPUT = "haystack.agent.step.llm.input"
LLM_OUTPUT = "haystack.agent.step.llm.output"


@dataclass
class ReconstructionStats:
    lines: int = 0
    spans: int = 0
    completed: int = 0
    incomplete: int = 0
    malformed_lines: int = 0
    malformed_spans: int = 0


def _otlp_value(value: Any) -> Any:
    """Decode an OTLP AnyValue represented by protobuf's JSON mapping."""
    if not isinstance(value, dict):
        return value
    scalar_fields = (
        "stringValue",
        "intValue",
        "doubleValue",
        "boolValue",
        "bytesValue",
    )
    for field in scalar_fields:
        if field in value:
            return value[field]
    if "arrayValue" in value:
        return [_otlp_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            item["key"]: _otlp_value(item.get("value"))
            for item in value["kvlistValue"].get("values", [])
            if "key" in item
        }
    return value


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    return {
        item["key"]: _otlp_value(item.get("value"))
        for item in span.get("attributes", [])
        if "key" in item
    }


def _spans(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            yield from scope_spans.get("spans", [])


def _json_object(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{field} is not a JSON object")
    return value


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _message_text(parts: Any) -> tuple[list[str], list[Any], list[dict], list[dict], list[Any]]:
    if isinstance(parts, str):
        return [parts], [], [], [], []
    if not isinstance(parts, list):
        return [], [], [], [], [parts]

    texts: list[str] = []
    reasoning: list[Any] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    unknown: list[Any] = []
    for part in parts:
        if not isinstance(part, dict):
            unknown.append(part)
        elif "text" in part:
            texts.append(_json_text(part["text"]))
        elif "reasoning" in part:
            reasoning.append(part["reasoning"])
        elif "tool_call" in part:
            tool_calls.append(part["tool_call"])
        elif "tool_call_result" in part:
            tool_results.append(part["tool_call_result"])
        else:
            unknown.append(part)
    return texts, reasoning, tool_calls, tool_results, unknown


def _tool_call_id(call: dict[str, Any], index: int) -> str:
    extra = call.get("extra") if isinstance(call.get("extra"), dict) else {}
    return str(call.get("id") or extra.get("call_id") or f"call_{index}")


def to_openai_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Haystack's content-block messages to an OpenAI-style message list.

    Reasoning blocks are retained under a non-standard ``reasoning`` key. Use the
    ``haystack`` output format when byte-for-byte preservation of every message
    field is more important than OpenAI API compatibility.
    """
    converted: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            converted.append({"role": "unknown", "content": _json_text(raw)})
            continue

        role = str(raw.get("role") or "unknown")
        texts, reasoning, calls, results, unknown = _message_text(raw.get("content", []))

        if role == "tool" and results:
            for index, result in enumerate(results):
                origin = result.get("origin") if isinstance(result.get("origin"), dict) else {}
                message: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": _tool_call_id(origin, index),
                    "content": _json_text(result.get("result", "")),
                }
                if origin.get("tool_name"):
                    message["name"] = str(origin["tool_name"])
                if result.get("error"):
                    message["error"] = True
                converted.append(message)
            continue

        content_parts = [*texts, *(_json_text(item) for item in unknown)]
        message = {
            "role": role,
            "content": "\n".join(content_parts) if content_parts else (None if calls else ""),
        }
        if raw.get("name") is not None:
            message["name"] = raw["name"]
        if reasoning:
            message["reasoning"] = reasoning[0] if len(reasoning) == 1 else reasoning
        if calls:
            message["tool_calls"] = [
                {
                    "id": _tool_call_id(call, index),
                    "type": "function",
                    "function": {
                        "name": str(call.get("tool_name") or ""),
                        "arguments": _json_text(call.get("arguments", {})),
                    },
                }
                for index, call in enumerate(calls)
            ]
        converted.append(message)
    return converted


def _format_messages(messages: list[dict[str, Any]], messages_format: str) -> list[dict[str, Any]]:
    if messages_format == "haystack":
        return messages
    return to_openai_messages(messages)


def _record_from_root(
    span: dict[str, Any], attributes: dict[str, Any], messages_format: str
) -> dict[str, Any]:
    state = _json_object(attributes[ROOT_OUTPUT], field=ROOT_OUTPUT)
    messages = state.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{ROOT_OUTPUT}.messages is not a list")
    return {
        "trace_id": span.get("traceId"),
        "status": "completed",
        "messages": _format_messages(messages, messages_format),
        "metadata": {
            "step_count": state.get("step_count"),
            "token_usage": state.get("token_usage"),
            "tool_call_counts": state.get("tool_call_counts"),
            "start_time_unix_nano": span.get("startTimeUnixNano"),
            "end_time_unix_nano": span.get("endTimeUnixNano"),
            "source_span": ROOT_SPAN,
        },
    }


def _record_from_step(
    trace_id: str,
    ended: str,
    input_value: Any,
    output_value: Any,
    messages_format: str,
) -> dict[str, Any]:
    llm_input = _json_object(input_value, field=LLM_INPUT)
    llm_output = _json_object(output_value, field=LLM_OUTPUT)
    messages = llm_input.get("messages")
    replies = llm_output.get("replies")
    if not isinstance(messages, list) or not isinstance(replies, list):
        raise ValueError("latest LLM input messages or output replies is not a list")
    reconstructed = [*messages, *replies]
    return {
        "trace_id": trace_id,
        "status": "incomplete",
        "messages": _format_messages(reconstructed, messages_format),
        "metadata": {
            "step_count": None,
            "token_usage": None,
            "tool_call_counts": None,
            "end_time_unix_nano": ended,
            "source_span": LLM_SPAN,
        },
    }


def _resolve_input(path: Path) -> Path:
    if path.is_dir():
        path = path / "traces.json"
    if not path.is_file():
        raise FileNotFoundError(f"Trace file not found: {path}")
    return path


def _open_input(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _write_record(output: IO[str], record: dict[str, Any]) -> None:
    output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def reconstruct(
    input_path: Path,
    output: IO[str],
    *,
    messages_format: str = "openai",
    include_incomplete: bool = False,
    strict: bool = False,
    progress_every: int = 0,
    diagnostics: IO[str] = sys.stderr,
) -> ReconstructionStats:
    """Stream trace records into one conversation record per JSONL output line."""
    input_path = _resolve_input(input_path)
    stats = ReconstructionStats()
    seen_roots: set[str] = set()

    pending_file = tempfile.NamedTemporaryFile(
        prefix="secagent-trace-messages-", suffix=".sqlite3", delete=False
    ) if include_incomplete else None
    pending_path = Path(pending_file.name) if pending_file else None
    if pending_file:
        pending_file.close()

    database_context = (
        sqlite3.connect(str(pending_path)) if pending_path is not None else nullcontext(None)
    )
    try:
        with database_context as database, _open_input(input_path) as source:
            if database is not None:
                database.execute(
                    "CREATE TABLE pending "
                    "(trace_id TEXT PRIMARY KEY, ended TEXT, input TEXT, output TEXT)"
                )

            for line_number, line in enumerate(source, start=1):
                stats.lines = line_number
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats.malformed_lines += 1
                    if strict:
                        raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
                    if stats.malformed_lines <= 10:
                        print(
                            f"warning: skipping malformed JSON at {input_path}:{line_number}: {exc}",
                            file=diagnostics,
                        )
                    continue

                for span in _spans(payload):
                    stats.spans += 1
                    name = span.get("name")
                    if name not in (ROOT_SPAN, LLM_SPAN):
                        continue
                    trace_id = str(span.get("traceId") or "")
                    if not trace_id:
                        stats.malformed_spans += 1
                        continue
                    attributes = _attributes(span)

                    if name == ROOT_SPAN and ROOT_OUTPUT in attributes:
                        if trace_id in seen_roots:
                            continue
                        try:
                            record = _record_from_root(span, attributes, messages_format)
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                            stats.malformed_spans += 1
                            if strict:
                                raise ValueError(f"invalid root span for trace {trace_id}: {exc}") from exc
                            print(
                                f"warning: skipping invalid root span for trace {trace_id}: {exc}",
                                file=diagnostics,
                            )
                            continue
                        seen_roots.add(trace_id)
                        if database is not None:
                            database.execute("DELETE FROM pending WHERE trace_id = ?", (trace_id,))
                        _write_record(output, record)
                        stats.completed += 1
                        continue

                    if (
                        database is not None
                        and name == LLM_SPAN
                        and trace_id not in seen_roots
                        and LLM_INPUT in attributes
                        and LLM_OUTPUT in attributes
                    ):
                        ended = str(span.get("endTimeUnixNano") or "")
                        database.execute(
                            "INSERT INTO pending(trace_id, ended, input, output) VALUES (?, ?, ?, ?) "
                            "ON CONFLICT(trace_id) DO UPDATE SET "
                            "ended=excluded.ended, input=excluded.input, output=excluded.output "
                            "WHERE CAST(excluded.ended AS INTEGER) >= CAST(pending.ended AS INTEGER)",
                            (
                                trace_id,
                                ended or "0",
                                _json_text(attributes[LLM_INPUT]),
                                _json_text(attributes[LLM_OUTPUT]),
                            ),
                        )

                if progress_every and line_number % progress_every == 0:
                    print(
                        f"processed {line_number:,} lines; "
                        f"completed={stats.completed:,}",
                        file=diagnostics,
                    )

            if database is not None:
                for trace_id, ended, input_value, output_value in database.execute(
                    "SELECT trace_id, ended, input, output FROM pending ORDER BY ended, trace_id"
                ):
                    try:
                        record = _record_from_step(
                            trace_id, ended, input_value, output_value, messages_format
                        )
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        stats.malformed_spans += 1
                        if strict:
                            raise ValueError(
                                f"invalid incomplete trace {trace_id}: {exc}"
                            ) from exc
                        print(
                            f"warning: skipping invalid incomplete trace {trace_id}: {exc}",
                            file=diagnostics,
                        )
                        continue
                    _write_record(output, record)
                    stats.incomplete += 1
    finally:
        if pending_path is not None:
            pending_path.unlink(missing_ok=True)

    return stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct conversations from a secagent OTLP JSONL trace. "
            "The output is JSONL with one {trace_id, status, messages, metadata} record per run."
        )
    )
    parser.add_argument(
        "trace",
        type=Path,
        help="traces.json, traces.json.gz, or a trace directory containing traces.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output JSONL path (default: stdout)",
    )
    parser.add_argument(
        "--messages-format",
        choices=("openai", "haystack"),
        default="openai",
        help="normalize messages for OpenAI APIs or preserve raw Haystack blocks (default: openai)",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="recover interrupted runs from their latest completed LLM step",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of warning when a JSON line or relevant span is malformed",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        metavar="N",
        help="report progress to stderr every N input lines; 0 disables it (default: 10000)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the final reconstruction summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = _resolve_input(args.trace)
    if args.output is not None and input_path.resolve() == args.output.resolve():
        raise SystemExit("Refusing to overwrite the input trace file.")

    output_context = args.output.open("w", encoding="utf-8") if args.output else nullcontext(sys.stdout)
    try:
        with output_context as output:
            stats = reconstruct(
                input_path,
                output,
                messages_format=args.messages_format,
                include_incomplete=args.include_incomplete,
                strict=args.strict,
                progress_every=args.progress_every,
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if not args.quiet:
        print(
            "reconstruction: "
            + ", ".join(f"{key}={value:,}" for key, value in asdict(stats).items()),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
