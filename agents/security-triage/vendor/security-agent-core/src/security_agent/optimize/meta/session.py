"""Per-iteration session shared between the meta-agent's tools and hooks.

The meta-agent's tools are module-level functions (so the meta-agent itself is
serializable); they reach run-scoped resources through the current session, which
the optimizer installs before each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from haystack.tools import Tool

from security_agent.benchmarks.base import Benchmark, Task
from security_agent.optimize.config_patch import PatchPolicy
from security_agent.optimize.settings import Settings
from security_agent.optimize.store import RunStore


@dataclass
class MetaSession:
    store: RunStore
    settings: Settings
    catalog: dict[str, Tool]
    policy: PatchPolicy
    working_config: dict
    champion_hash: str
    champion_score: float
    smoke_tasks: list[Task]
    benchmark: Benchmark
    # The benchmark's seed agent (seeds/<name>.yaml as a dict) — the source the tool catalog is
    # rebuilt from (e.g. after create_component adds an extension).
    seed_config: dict = field(default_factory=dict)

    applied_ops: list[dict] = field(default_factory=list)
    validated: bool = False
    submitted: bool = False
    submitted_summary: str | None = None
    smoke_score: float | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


_current: MetaSession | None = None


def set_session(session: MetaSession | None) -> None:
    global _current
    _current = session


def current_session() -> MetaSession:
    if _current is None:
        raise RuntimeError(
            "No active MetaSession. Meta-agent tools can only run inside an optimizer iteration."
        )
    return _current
