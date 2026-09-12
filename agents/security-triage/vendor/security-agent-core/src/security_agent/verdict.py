"""Parse the triage agent's final answer into a normalized verdict.

The PrimeVul agent is configured with structured outputs (a strict JSON schema lives in
the target's ``chat_generator.generation_kwargs.output_config`` in ``seeds/primevul.yaml``),
so its final message is guaranteed-valid JSON matching :class:`VulnerabilityVerdict`.
``benchmarks.primevul`` calls :func:`parse_verdict` to turn that message into the plain
dict the scorer (``security_agent.evaluate``) consumes.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError


def normalize_cwe(raw: Any) -> Optional[str]:
    """'787' / 'cwe-787' / 'CWE 787' -> 'CWE-787'; empty/None/no-digits -> None."""
    if raw is None:
        return None
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return f"CWE-{digits}" if digits else None


# `VulnerabilityVerdict` is the single source of truth for the strict JSON schema the
# target agent is constrained to emit. `strict_schema()` mirrors it into
# `seeds/primevul.yaml`'s `output_config`; `tests/test_seeds.py` guards the two against
# drift. The class docstring below is intentionally short: it becomes the schema's
# top-level `description`, which is sent to the model.
class VulnerabilityVerdict(BaseModel):
    """A security-triage verdict for a single source-code function."""

    is_vulnerable: bool = Field(
        description="Whether the function contains a security vulnerability."
    )
    cwe: Optional[str] = Field(
        description="CWE identifier like 'CWE-89'; null when not vulnerable."
    )
    vulnerable_lines: list[int] = Field(
        description="1-based line numbers; empty when not vulnerable."
    )
    rationale: str = Field(description="One or two sentence justification.")


def strict_schema() -> dict:
    """The JSON schema Anthropic's structured outputs require: the model's schema with
    ``additionalProperties: false``. Embedded verbatim in ``seeds/primevul.yaml``."""
    return {**VulnerabilityVerdict.model_json_schema(), "additionalProperties": False}


def parse_verdict(text: str) -> dict:
    """Return a normalized verdict dict.

    On success: ``{is_vulnerable, cwe, vulnerable_lines, rationale, parse_error: False}``.
    Structured outputs make the agent's final message schema-valid JSON, so validation
    only fails for an empty or errored task (e.g. the runner recorded ``answer=""`` after
    an exception); such a verdict is scored as wrong via ``parse_error`` rather than
    crashing the batch.
    """
    try:
        verdict = VulnerabilityVerdict.model_validate_json(text or "")
    except ValidationError:
        return _failed(text)
    return {
        "is_vulnerable": verdict.is_vulnerable,
        "cwe": normalize_cwe(verdict.cwe),
        "vulnerable_lines": verdict.vulnerable_lines,
        "rationale": verdict.rationale.strip(),
        "parse_error": False,
    }


def extract_json_object(text: str) -> Optional[str]:
    """Pull the verdict JSON object out of a free-text answer, or return None.

    Tries, in order: the whole string; the contents of the last fenced block; the last
    balanced ``{...}`` span. "Last" rather than "first" because a model that reasons before
    answering tends to put the real verdict at the end — an early brace is more often an
    example it is quoting than its actual answer.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    candidates: list[str] = []
    # ```json ... ``` or plain ``` ... ```
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        if block.startswith("{"):
            candidates.append(block)

    # Balanced-brace scan, so nested objects inside the verdict do not truncate it.
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])

    return candidates[-1] if candidates else None


def parse_verdict_lenient(text: str) -> dict:
    """Parse a verdict that is not guaranteed to be bare JSON.

    NeMo Gym drops the Responses API's ``text.format`` when it converts to chat/completions, so
    an agent running under Gym is *asked* for the verdict schema rather than constrained to it.
    This accepts the same answers :func:`parse_verdict` does, plus a JSON object wrapped in prose
    or a code fence. Anything with no recoverable object is still a ``parse_error`` — the model
    failing to produce the output format is a real result, not something to paper over.
    """
    direct = parse_verdict(text)
    if not direct.get("parse_error"):
        return direct
    candidate = extract_json_object(text or "")
    if candidate is None:
        return _failed(text)
    recovered = parse_verdict(candidate)
    return recovered if not recovered.get("parse_error") else _failed(text)


def _failed(text: str) -> dict:
    return {
        "is_vulnerable": None,
        "cwe": None,
        "vulnerable_lines": [],
        "rationale": "",
        "parse_error": True,
        "raw": (text or "")[:500],
    }


# --- YES/NO option parsing, for PrimeVul's own published prompt format -------------------------
#
# Adapted from the parser in the PrimeVul Gym environment on `bogdankostic/Gym@primevul_env`, whose
# design note explains the trap: "no" is ordinary English, so a reply like
# ``"YES. There is no bounds check."`` contains two candidate tokens. Taking the *last* match reads
# the wrong one and silently inverts the verdict -- corrupting only the positive detections,
# inflating the benign rate and depressing recall, and never raising a parse error. So this is a
# priority ladder over increasingly weak evidence, stopping at the strongest tier that matches.

_REASONING_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)

_VERDICT_TIERS = (
    # (pattern, take_last, is_numbered), strongest evidence first.
    (re.compile(r"\(([12])\)"), True, True),                                   # "(1)" / "(2)"
    (re.compile(r"\b(YES|NO)\s*:", re.IGNORECASE), True, False),               # "YES: ..." labelled option
    (re.compile(r"(?:^|[.!?]\s+)\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE), False, False),
    (re.compile(r"\b(YES|NO)\b"), True, False),                                # bare token, case-sensitive
    (re.compile(r"\A\W*([12])\W*\Z"), True, True),                             # the whole reply is "1"
)


def parse_yes_no(text: str) -> dict:
    """Parse PrimeVul's two-option answer into the same shape as :func:`parse_verdict`."""
    cleaned = _REASONING_BLOCK.sub("", text or "")
    for pattern, take_last, numbered in _VERDICT_TIERS:
        tokens = pattern.findall(cleaned)
        if tokens:
            token = tokens[-1] if take_last else tokens[0]
            is_vuln = token == "1" if numbered else token.upper() == "YES"
            return {"is_vulnerable": is_vuln, "cwe": None, "vulnerable_lines": [],
                    "rationale": "", "parse_error": False}
    return {"is_vulnerable": None, "cwe": None, "vulnerable_lines": [], "rationale": "",
            "parse_error": True, "raw": (text or "")[:500]}


def parse_verdict_any(text: str) -> dict:
    """Parse either the JSON verdict or PrimeVul's YES/NO option, whichever the text carries.

    JSON is tried first so this is a strict superset of `parse_verdict_lenient`: an agent asked for
    the JSON schema is scored exactly as before, and only a reply with no recoverable object falls
    through to the option ladder. That lets one scoring path serve output-format ablations without
    a flag that could silently select the wrong parser.
    """
    verdict = parse_verdict_lenient(text)
    if not verdict.get("parse_error"):
        return verdict
    return parse_yes_no(text)
