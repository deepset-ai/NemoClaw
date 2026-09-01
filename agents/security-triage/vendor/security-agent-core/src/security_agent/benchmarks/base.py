"""Benchmark abstraction for the security agent.

A `Benchmark` bundles everything that varies per evaluation target: how tasks are
loaded (with train/holdout splits) and how the agent's output is scored into the scalar
the optimizer hill-climbs. The agent, the optimizer engine, and the CLI are all
benchmark-agnostic — they pick a benchmark by name via `security_agent.benchmarks.get_benchmark`.

TWO STYLES OF BENCHMARK share this interface:

- CLASSIFICATION-STYLE (e.g. PrimeVul): the agent is run once per task (one user message in,
  its final text out) by the *shared* runner in `security_agent.optimize.harness`, and `score`
  is a pure function of (task, agent output). Such a benchmark implements only `load_tasks` +
  `score`.

- EXECUTION-STYLE (e.g. SEC-bench): the agent must act inside a sandbox (write a patch) and
  success is judged by *running* the artifacts against Docker images with sanitizers. Such a
  benchmark additionally implements the optional `run(config, tasks, settings)` method and
  thereby OWNS its runner. `security_agent.optimize.harness.runner.run_tasks` dispatches to
  `run` when it is present (duck-typed, like the optional `build` classmethod) and falls back
  to the shared classification runner otherwise. `score` then verifies the produced artifacts
  (for SEC-bench, a Docker run → resolve rate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Task:
    """One benchmark task. `question` is the ready-to-run agent input (already rendered
    from the raw sample); `gold` holds benchmark-specific labels used only by `score`."""

    id: str
    question: str
    split: str = "train"  # "train" (drives hill-climbing) or "holdout" (report only)
    gold: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """The generic output of running the agent on one task (produced by the shared
    runner in `optimize/harness/runner.py`; consumed by a benchmark's `score`)."""

    task_id: str
    answer: str
    error: str | None = None
    steps: int | None = None
    seconds: float | None = None
    token_usage: object = None


@dataclass
class ScoreReport:
    """A scored run: the scalar objective plus the full metrics and per-task detail."""

    score: float
    metrics: dict[str, Any]
    mean_steps: float
    individual: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"score={self.score:.3f} mean_steps={self.mean_steps:.1f}"]
        failed = [r for r in self.individual if not r.get("match")]
        if failed:
            lines.append(f"failed tasks ({len(failed)}/{len(self.individual)}):")
            lines += [
                f"  {r['task_id']}: expected {r.get('expected')}, got {r.get('answer')!r}"
                + (f" (error: {r['error']})" if r.get("error") else "")
                for r in failed
            ]
        return "\n".join(lines)


@runtime_checkable
class Benchmark(Protocol):
    """A classification-style benchmark (see module docstring for the boundary)."""

    name: str

    def load_tasks(self) -> list[Task]:
        """All tasks across every split, each with `question` rendered and `gold` set."""
        ...

    def score(
        self,
        tasks: list[Task],
        results: list[TaskResult],
        *,
        step_cost_lambda: float = 0.0,
        max_agent_steps: int = 50,
    ) -> ScoreReport:
        """Turn aligned (tasks, agent results) into metrics + the scalar objective."""
        ...


def split(tasks: list[Task], name: str) -> list[Task]:
    return [t for t in tasks if t.split == name]


def smoke(tasks: list[Task], size: int) -> list[Task]:
    """A small train subset for the meta-agent's indicative smoke test."""
    return split(tasks, "train")[:size]
