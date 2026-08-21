"""Tool catalog for the target agent — the tools the meta-agent may enable/disable/re-describe.

The catalog is derived from the benchmark's **seed** (`seeds/<name>.yaml`), the single source of
truth: it is the seed agent's own tools, fully wired (component `type` + `outputs_to_string` +
description), reconstructed via the same `Agent.from_dict` path that eval and `enable_tool` trust.
Sourcing from the seed (an immutable baseline) — not the mutable champion — means a tool the
meta-agent has disabled can always be re-enabled.

The meta-agent never writes tool code into a config: it can only enable, disable, or re-describe
catalog entries (`security_agent.optimize.config_patch`), or — when explicitly allowed
(`optimizer.allow_create_component`) — extend the catalog with a generated `@component` class via
`security_agent.optimize.component_authoring`.
"""

import importlib
from typing import TYPE_CHECKING

from haystack.tools import Tool
from haystack.tools.component_tool import ComponentTool

from security_agent.optimize.validate import load_agent

if TYPE_CHECKING:
    from security_agent.optimize.component_authoring import ComponentExtension


def get_catalog(
    seed_config: dict, extensions: list["ComponentExtension"] | None = None
) -> dict[str, Tool]:
    """Return all available tools by name for a benchmark, keyed by tool name.

    :param seed_config: The benchmark's serialized seed agent (``seeds/<name>.yaml`` loaded as a
        dict). Its tools ARE the catalog's baseline entries.
    :param extensions: Component extensions registered by the meta-agent, loaded from the run
        store's extensions manifest (only used when create_component is allowed).
    """
    catalog: dict[str, Tool] = {tool.name: tool for tool in (load_agent(seed_config).tools or [])}
    for ext in extensions or []:
        module = importlib.import_module(f"security_agent.optimize.generated.{ext.module_name}")
        component_cls = getattr(module, ext.class_name)
        instance = component_cls(**ext.init_params)
        catalog[ext.tool_name] = ComponentTool(
            component=instance, name=ext.tool_name, description=ext.description
        )
    return catalog


def serialized_tool(name: str, catalog: dict[str, Tool]) -> dict:
    """Return the pre-serialized dict for a catalog tool, ready to splice into a config."""
    if name not in catalog:
        raise KeyError(f"Unknown tool '{name}'. Available tools: {', '.join(sorted(catalog))}")
    return catalog[name].to_dict()
