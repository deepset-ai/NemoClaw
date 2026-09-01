"""Structured patching of a serialized Agent config.

The meta-agent never edits YAML text. It proposes a list of ops against whitelisted
logical paths; everything else — `type:` fields, callables, hooks, state schema —
is rejected before deserialization is even attempted. Tool entries only ever enter
the config by splicing pre-serialized dicts from the tool catalog.

This is layer 1 of the defense in depth. Layers 2 and 3 are Haystack's
deserialization allowlist (`security_agent.optimize.validate`) and subprocess execution
(`security_agent.optimize.harness.runner`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from haystack.tools import Tool

from security_agent.optimize.tool_catalog import serialized_tool


class PatchError(ValueError):
    """A patch op was rejected. The message is written for the meta-agent to act on."""


@dataclass(frozen=True)
class PatchPolicy:
    approved_models: tuple[str, ...] = ("gpt-5.4-nano", "gpt-5.4-mini")
    max_agent_steps_range: tuple[int, int] = (1, 50)
    tool_concurrency_range: tuple[int, int] = (1, 8)
    temperature_range: tuple[float, float] = (0.0, 2.0)
    # Bounds for the coding agent's output-token limit. Upper bound above the SEC-bench seed's 16k so
    # the meta-agent can tune it (the OpenAI Responses API kwarg is `max_output_tokens`) rather than
    # being forced to cap it below the seed value.
    max_tokens_range: tuple[int, int] = (16, 32000)
    top_p_range: tuple[float, float] = (0.0, 1.0)


GENERATION_KWARG_KEYS = ("temperature", "max_output_tokens", "top_p")

# Logical paths the meta-agent may `set`, with a short human/LLM-readable constraint.
def mutable_paths(policy: PatchPolicy) -> dict[str, str]:
    lo, hi = policy.max_agent_steps_range
    clo, chi = policy.tool_concurrency_range
    tlo, thi = policy.temperature_range
    mlo, mhi = policy.max_tokens_range
    plo, phi = policy.top_p_range
    return {
        "system_prompt": "string or null",
        "user_prompt": "string or null",
        "exit_conditions": 'list of strings; each must be "text" or the name of an enabled tool',
        "max_agent_steps": f"integer in [{lo}, {hi}]",
        "tool_concurrency_limit": f"integer in [{clo}, {chi}]",
        "raise_on_tool_invocation_failure": "boolean",
        "chat_generator.model": f"one of: {', '.join(policy.approved_models)}",
        "chat_generator.generation_kwargs.temperature": f"number in [{tlo}, {thi}]",
        "chat_generator.generation_kwargs.max_output_tokens": f"integer in [{mlo}, {mhi}]",
        "chat_generator.generation_kwargs.top_p": f"number in [{plo}, {phi}]",
    }


def describe_ops(policy: PatchPolicy) -> dict[str, Any]:
    """A machine-readable description of the patch interface, for the meta-agent."""
    return {
        "ops": {
            "set": {"path": "one of the mutable paths", "value": "the new value"},
            "enable_tool": {"name": "a tool name from the catalog"},
            "disable_tool": {"name": "a currently enabled tool name"},
            "set_tool_description": {"name": "an enabled tool name", "value": "new description string"},
        },
        "mutable_paths": mutable_paths(policy),
    }


def _init_params(config: dict) -> dict:
    if not isinstance(config, dict) or "init_parameters" not in config:
        raise PatchError("Config is not a serialized Agent: missing 'init_parameters'.")
    return config["init_parameters"]


def _tools_list(config: dict) -> list[dict]:
    tools = _init_params(config).setdefault("tools", [])
    if tools is None:
        tools = []
        _init_params(config)["tools"] = tools
    if not isinstance(tools, list):
        raise PatchError(
            "This config stores tools as a Toolset; only plain tool lists can be patched."
        )
    return tools


def enabled_tool_names(config: dict) -> list[str]:
    names = []
    for item in _tools_list(config):
        data = item.get("data", {}) if isinstance(item, dict) else {}
        name = data.get("name")
        if name:
            names.append(name)
    return names


def _check_number(value: Any, bounds: tuple[float, float], path: str, integer: bool) -> None:
    lo, hi = bounds
    allowed_types = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, allowed_types):
        kind = "an integer" if integer else "a number"
        raise PatchError(f"'{path}' must be {kind}, got {value!r}.")
    if not lo <= value <= hi:
        raise PatchError(f"'{path}' must be in [{lo}, {hi}], got {value!r}.")


def _apply_set(config: dict, path: str, value: Any, policy: PatchPolicy) -> None:
    ip = _init_params(config)

    if path in ("system_prompt", "user_prompt"):
        if value is not None and not isinstance(value, str):
            raise PatchError(f"'{path}' must be a string or null, got {type(value).__name__}.")
        ip[path] = value
    elif path == "exit_conditions":
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise PatchError("'exit_conditions' must be a non-empty list of strings.")
        ip[path] = value
    elif path == "max_agent_steps":
        _check_number(value, policy.max_agent_steps_range, path, integer=True)
        ip[path] = value
    elif path == "tool_concurrency_limit":
        _check_number(value, policy.tool_concurrency_range, path, integer=True)
        ip[path] = value
    elif path == "raise_on_tool_invocation_failure":
        if not isinstance(value, bool):
            raise PatchError(f"'{path}' must be a boolean, got {value!r}.")
        ip[path] = value
    elif path == "chat_generator.model":
        if value not in policy.approved_models:
            raise PatchError(
                f"Model {value!r} is not approved. Approved models: {', '.join(policy.approved_models)}."
            )
        ip["chat_generator"]["init_parameters"]["model"] = value
    elif path.startswith("chat_generator.generation_kwargs."):
        key = path.rsplit(".", 1)[1]
        if key not in GENERATION_KWARG_KEYS:
            raise PatchError(
                f"Unknown generation kwarg '{key}'. Allowed: {', '.join(GENERATION_KWARG_KEYS)}."
            )
        if key == "temperature":
            _check_number(value, policy.temperature_range, path, integer=False)
        elif key == "max_output_tokens":
            _check_number(value, policy.max_tokens_range, path, integer=True)
        elif key == "top_p":
            _check_number(value, policy.top_p_range, path, integer=False)
        gen_ip = ip["chat_generator"]["init_parameters"]
        if gen_ip.get("generation_kwargs") is None:
            gen_ip["generation_kwargs"] = {}
        gen_ip["generation_kwargs"][key] = value
    else:
        raise PatchError(
            f"Path '{path}' is not mutable. Mutable paths: {', '.join(mutable_paths(policy))}."
        )


def _find_tool(config: dict, name: str) -> dict:
    for item in _tools_list(config):
        if isinstance(item, dict) and item.get("data", {}).get("name") == name:
            return item
    raise PatchError(
        f"Tool '{name}' is not enabled in this config. Enabled tools: "
        f"{', '.join(enabled_tool_names(config)) or '(none)'}."
    )


def apply_patch(
    config: dict,
    ops: list[dict],
    catalog: dict[str, Tool],
    policy: PatchPolicy | None = None,
) -> dict:
    """
    Apply patch ops to a serialized Agent config and return the patched copy.

    :param config: The serialized Agent (``{"type": ..., "init_parameters": {...}}``).
    :param ops: A list of op dicts, see :func:`describe_ops`.
    :param catalog: Available tools by name; the only source of new tool entries.
    :param policy: Bounds and approved values for scalar paths.
    :raises PatchError: If any op is invalid. The input config is never modified.
    """
    policy = policy or PatchPolicy()
    if not isinstance(ops, list) or not ops:
        raise PatchError("Expected a non-empty list of op objects.")

    patched = copy.deepcopy(config)
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or "op" not in op:
            raise PatchError(f"Op #{i} must be an object with an 'op' key, got {op!r}.")
        kind = op["op"]
        if kind == "set":
            if "path" not in op or "value" not in op:
                raise PatchError(f"Op #{i}: 'set' requires 'path' and 'value'.")
            _apply_set(patched, op["path"], op["value"], policy)
        elif kind == "enable_tool":
            name = op.get("name")
            if name in enabled_tool_names(patched):
                raise PatchError(f"Op #{i}: tool '{name}' is already enabled.")
            try:
                _tools_list(patched).append(serialized_tool(name, catalog))
            except KeyError as e:
                raise PatchError(f"Op #{i}: {e.args[0]}") from e
        elif kind == "disable_tool":
            entry = _find_tool(patched, op.get("name"))
            _tools_list(patched).remove(entry)
        elif kind == "set_tool_description":
            value = op.get("value")
            if not isinstance(value, str) or not value.strip():
                raise PatchError(f"Op #{i}: 'set_tool_description' requires a non-empty string 'value'.")
            _find_tool(patched, op.get("name"))["data"]["description"] = value
        else:
            raise PatchError(
                f"Op #{i}: unknown op '{kind}'. "
                "Allowed ops: set, enable_tool, disable_tool, set_tool_description."
            )

    _cross_check(patched)
    return patched


def _cross_check(config: dict) -> None:
    """Checks that span multiple fields, run after all ops are applied."""
    valid_exits = set(enabled_tool_names(config)) | {"text"}
    exit_conditions = _init_params(config).get("exit_conditions") or ["text"]
    invalid = [c for c in exit_conditions if c not in valid_exits]
    if invalid:
        raise PatchError(
            f"exit_conditions {invalid} do not match any enabled tool. "
            f"Valid values: {', '.join(sorted(valid_exits))}. "
            "Either enable the tool or update exit_conditions in the same patch."
        )
