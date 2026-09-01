"""Benchmark registry. Select a benchmark by name; add one by implementing the
`Benchmark` protocol (see `base.py`) and registering it in `_REGISTRY`."""

from __future__ import annotations

from .base import Benchmark, ScoreReport, Task, TaskResult, smoke, split
from .primevul import PrimeVul
from .secbench import SecBench

_REGISTRY: dict[str, type] = {
    "primevul": PrimeVul,
    "secbench": SecBench,
}


def get_benchmark(name: str, **kwargs) -> Benchmark:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {', '.join(sorted(_REGISTRY))}."
        ) from None
    return cls(**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Benchmark",
    "ScoreReport",
    "Task",
    "TaskResult",
    "get_benchmark",
    "available",
    "smoke",
    "split",
]
