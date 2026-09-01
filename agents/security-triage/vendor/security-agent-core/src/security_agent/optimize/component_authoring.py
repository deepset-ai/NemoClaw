"""Authoring workflow for meta-agent-generated custom components (M5).

The meta-agent cannot put code into a config — YAML only references classes by
import path. This module closes the loop the way deepset Cloud custom components do:
generated `@component` classes are written into the `security_agent.optimize.generated` package
(which is on the deserialization allowlist), validated, wrapped in a `ComponentTool`,
and registered as a catalog extension the meta-agent can then enable like any other
tool.

Safety layers for generated code:
1. a static denylist scan below (best-effort, fails fast with a clear message),
2. the benchmark subprocess is the only place candidate configs execute,
3. optional human approval of every `create_component` call via `ConfirmationHook`.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from haystack.tools.component_tool import ComponentTool

GENERATED_PACKAGE = "security_agent.optimize.generated"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CLASS_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")

# Generated components must be tiny and pure. Anything that touches process, file
# system, or network is rejected outright.
_FORBIDDEN_PATTERNS = (
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import shutil",
    "import pathlib",
    "import ctypes",
    "importlib",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "getattr(",
    "setattr(",
    "globals(",
    "locals(",
    "requests",
    "urllib",
    "httpx",
)


class ComponentAuthoringError(ValueError):
    """A generated component was rejected. The message is written for the meta-agent."""


@dataclass(frozen=True)
class ComponentExtension:
    """A registered catalog extension backed by a generated component."""

    module_name: str
    class_name: str
    tool_name: str
    description: str
    init_params: dict[str, Any] = field(default_factory=dict)


def _package_dir() -> Path:
    package = importlib.import_module(GENERATED_PACKAGE)
    return Path(package.__file__).parent


def check_source(source: str) -> None:
    """Static checks on generated component source. Best-effort, not a sandbox."""
    lowered = source.lower()
    hits = [p for p in _FORBIDDEN_PATTERNS if p in lowered]
    if hits:
        raise ComponentAuthoringError(
            f"Source rejected, it contains forbidden patterns: {', '.join(sorted(set(hits)))}. "
            "Generated components must be pure Python without file system, network, or process access."
        )
    if "@component" not in source:
        raise ComponentAuthoringError(
            "Source must define a class decorated with @component "
            "(from haystack import component)."
        )


def create_component(
    *,
    module_name: str,
    class_name: str,
    tool_name: str,
    description: str,
    source: str,
    init_params: dict[str, Any] | None = None,
    existing_tool_names: set[str] | None = None,
) -> tuple[ComponentExtension, dict]:
    """
    Validate, install, and wrap a generated component as a catalog extension.

    Writes the module into the `security_agent.optimize.generated` package, imports it, instantiates
    the class, wraps it in a `ComponentTool`, and round-trips that tool through
    to_dict/from_dict to prove the resulting config entry deserializes.

    :returns: The extension record and the serialized ComponentTool dict.
    :raises ComponentAuthoringError: With an actionable message on any failure.
    """
    init_params = init_params or {}
    if not _NAME_RE.match(module_name):
        raise ComponentAuthoringError(
            f"module_name '{module_name}' must match {_NAME_RE.pattern} (snake_case)."
        )
    if not _CLASS_RE.match(class_name):
        raise ComponentAuthoringError(f"class_name '{class_name}' must be a CamelCase identifier.")
    if not _NAME_RE.match(tool_name):
        raise ComponentAuthoringError(f"tool_name '{tool_name}' must match {_NAME_RE.pattern}.")
    if existing_tool_names and tool_name in existing_tool_names:
        raise ComponentAuthoringError(f"tool_name '{tool_name}' already exists in the catalog.")
    if not description.strip():
        raise ComponentAuthoringError("description must not be empty.")
    check_source(source)

    module_path = _package_dir() / f"{module_name}.py"
    if module_path.exists() and module_path.read_text() != source:
        raise ComponentAuthoringError(
            f"Module '{module_name}' already exists with different content. Pick a new module_name."
        )
    module_path.write_text(source)

    qualified = f"{GENERATED_PACKAGE}.{module_name}"
    try:
        if qualified in sys.modules:
            module = importlib.reload(sys.modules[qualified])
        else:
            module = importlib.import_module(qualified)
    except Exception as e:
        module_path.unlink(missing_ok=True)
        raise ComponentAuthoringError(f"Module does not import: {type(e).__name__}: {e}") from e

    try:
        component_cls = getattr(module, class_name)
    except AttributeError:
        module_path.unlink(missing_ok=True)
        raise ComponentAuthoringError(
            f"Module '{module_name}' defines no class named '{class_name}'."
        ) from None

    from security_agent.optimize.validate import register_allowlist

    register_allowlist()  # the round-trip below deserializes from the generated package
    try:
        instance = component_cls(**init_params)
        component_tool = ComponentTool(component=instance, name=tool_name, description=description)
        serialized = component_tool.to_dict()
        ComponentTool.from_dict(serialized)
    except Exception as e:
        module_path.unlink(missing_ok=True)
        raise ComponentAuthoringError(
            f"Class '{class_name}' is not a usable Haystack component: {type(e).__name__}: {e}. "
            "It must be decorated with @component and its run() must declare typed parameters."
        ) from e

    extension = ComponentExtension(
        module_name=module_name,
        class_name=class_name,
        tool_name=tool_name,
        description=description,
        init_params=init_params,
    )
    return extension, serialized
