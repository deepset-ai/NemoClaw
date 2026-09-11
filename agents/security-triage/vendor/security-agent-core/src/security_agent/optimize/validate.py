"""Load and validate serialized Agent configs behind Haystack's deserialization allowlist.

Wraps `Agent.from_dict` so that (a) this project's packages are on the allowlist and
(b) failures come back as concise, actionable messages the meta-agent can fix,
instead of raw tracebacks.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

# Construction (not execution) only: a placeholder lets `load_agent` deserialize an Agent (whose
# chat generator reads ANTHROPIC_API_KEY) without a real key present — e.g. when the tool catalog
# reconstructs tools from a seed offline, or in tests. The value is never used to call the API; a
# real key from .env wins at runtime. This is THE deserialization path, so setting it here covers
# every caller.
#
# The cost of that convenience: importing this module makes both variables *truthy* process-wide, so
# `if os.environ.get("ANTHROPIC_API_KEY")` stops answering "can a live call succeed?" and starts
# answering "has this module been imported?". Anything gating real API traffic — most of all a test's
# skip guard — must ask `has_real_api_key`, or a keyless run stops skipping and fails on auth.
ANTHROPIC_PLACEHOLDER_KEY = "sk-ant-placeholder-for-yaml-generation"
OPENAI_PLACEHOLDER_KEY = "sk-placeholder-for-yaml-generation"
PLACEHOLDER_API_KEYS = frozenset({ANTHROPIC_PLACEHOLDER_KEY, OPENAI_PLACEHOLDER_KEY})

os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_PLACEHOLDER_KEY)
os.environ.setdefault("OPENAI_API_KEY", OPENAI_PLACEHOLDER_KEY)


def has_real_api_key(variable: str) -> bool:
    """True when `variable` holds a key that could actually reach the provider."""
    value = (os.environ.get(variable) or "").strip()
    return bool(value) and value not in PLACEHOLDER_API_KEYS


def load_real_credentials(env_path) -> None:
    """Load `.env`, replacing only the placeholders this module installed.

    `load_dotenv` will not overwrite a variable that is already set, and importing this module sets
    both key variables — so a plain `load_dotenv` after this import silently keeps the placeholder.
    Callers used to work around that by loading `.env` at *import* time, which mutated the
    environment of anything that merely imported them; under pytest that turned a keyless run into
    a live, billable API call. This replaces placeholders and leaves real values (from the shell,
    or already loaded) alone.
    """
    from dotenv import dotenv_values

    for name, value in dotenv_values(env_path).items():
        if not value:
            continue
        current = os.environ.get(name)
        if current is None or current in PLACEHOLDER_API_KEYS:
            os.environ[name] = value

from haystack.components.agents import Agent  # noqa: E402
from haystack.core.serialization_security import allow_deserialization_module  # noqa: E402

AGENT_TYPE = "haystack.components.agents.agent.Agent"

# Modules (beyond Haystack's defaults) that serialized configs may reference. One
# pattern covers everything under the package: the agent's CWE components
# (security_agent.components), the engine's tool functions, and generated components
# (security_agent.optimize.generated.*).
ALLOWLIST_PATTERNS = ("security_agent",)


def register_allowlist() -> None:
    """Extend the process-wide deserialization allowlist with this project's packages."""
    for pattern in ALLOWLIST_PATTERNS:
        allow_deserialization_module(pattern)


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def message(self) -> str:
        if self.ok:
            return "Config is valid: it deserializes into a runnable Agent."
        return "Config is INVALID:\n" + "\n".join(f"- {e}" for e in self.errors)


def _concise(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 600 else text[:600] + " ..."


def load_agent(config: dict) -> Agent:
    """Deserialize a config into an Agent, with the project allowlist registered."""
    register_allowlist()
    return Agent.from_dict(copy.deepcopy(config))


def validate_config(config: dict) -> ValidationReport:
    """Check that a config is a serialized Agent that deserializes cleanly."""
    errors: list[str] = []

    if not isinstance(config, dict):
        return ValidationReport(ok=False, errors=["Config must be a mapping."])
    if config.get("type") != AGENT_TYPE:
        errors.append(f"Top-level 'type' must be '{AGENT_TYPE}', got {config.get('type')!r}.")
    if not isinstance(config.get("init_parameters"), dict):
        errors.append("'init_parameters' must be a mapping.")
    if errors:
        return ValidationReport(ok=False, errors=errors)

    try:
        load_agent(config)
    except Exception as e:  # noqa: BLE001 - every failure must become a report, not a crash
        return ValidationReport(ok=False, errors=[_concise(e)])
    return ValidationReport(ok=True)
