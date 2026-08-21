"""Parse the triage agent's final answer into a normalized verdict.

The PrimeVul agent is configured with structured outputs (a strict JSON schema lives in
the target's ``chat_generator.generation_kwargs.output_config`` in ``seeds/primevul.yaml``),
so its final message is guaranteed-valid JSON matching :class:`VulnerabilityVerdict`.
``benchmarks.primevul`` calls :func:`parse_verdict` to turn that message into the plain
dict the scorer (``security_agent.evaluate``) consumes.
"""
from __future__ import annotations

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


def _failed(text: str) -> dict:
    return {
        "is_vulnerable": None,
        "cwe": None,
        "vulnerable_lines": [],
        "rationale": "",
        "parse_error": True,
        "raw": (text or "")[:500],
    }
