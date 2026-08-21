"""Campaign settings loaded from the project-root `config.yaml`.

`config.yaml` is a per-benchmark map: cross-cutting defaults (`target`, `optimizer`) plus a
`benchmarks:` block per benchmark holding only what varies. The active benchmark is chosen by
`load_settings(..., benchmark=...)` (the CLI's `--benchmark`) or falls back to `default_benchmark`.
Every per-benchmark artifact path — seed, runs store, promoted pipeline, meta-agent program — is
*derived* from the active name by convention (`seeds/<name>.yaml`, `runs/<name>/`,
`pipelines/<name>.yaml`, `programs/<name>.md`) so two benchmarks can never overwrite one another and
the paths can't drift out of sync with the selected benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    project_root: Path

    # target agent
    seed_config: Path = Path("seeds/primevul.yaml")
    # Approved target models the optimizer may switch between (a single value pins the model, so
    # the optimizer works on prompt, tools, and step budget only). Set per-benchmark in config.yaml
    # (secbench -> gpt-5.4-mini, primevul -> claude-sonnet-5); this default is the no-config fallback.
    approved_models: tuple[str, ...] = ("claude-sonnet-5",)
    max_agent_steps_range: tuple[int, int] = (1, 8)

    # benchmark
    benchmark: str = "primevul"  # name resolved via security_agent.benchmarks.get_benchmark
    smoke_size: int = 4
    per_task_timeout: float = 180.0
    step_cost_lambda: float = 0.0

    # optimizer
    epsilon: float = 0.02
    iterations: int = 5
    meta_model: str = "gpt-5.4"
    meta_max_steps: int = 25
    hitl: bool = False
    # Gate the meta-agent's create_component tool. Off for the first security
    # campaigns: keep the optimization surface to prompt + tool descriptions + steps.
    allow_create_component: bool = False

    # per-benchmark artifact paths (derived from `benchmark` in load_settings)
    runs_dir: Path = Path("runs/primevul")
    program_file: Path = Path("programs/primevul.md")
    pipeline: Path = Path("pipelines/primevul.yaml")

    def path(self, p: Path) -> Path:
        """Resolve a settings path relative to the project root."""
        return p if p.is_absolute() else self.project_root / p


def load_settings(
    project_root: Path, config_file: Path | None = None, benchmark: str | None = None
) -> Settings:
    """Resolve settings for the active benchmark.

    :param benchmark: The benchmark to activate (the CLI's `--benchmark`). When omitted, falls
        back to `config.yaml`'s `default_benchmark` (or a legacy `benchmark.name`), then to the
        `Settings` default. Selecting a benchmark selects its whole config block *and* all of its
        derived artifact paths.
    """
    config_file = config_file or project_root / "config.yaml"
    raw: dict = {}
    if config_file.exists():
        raw = yaml.safe_load(config_file.read_text()) or {}

    target = raw.get("target", {}) or {}
    optimizer = raw.get("optimizer", {}) or {}
    benchmarks = raw.get("benchmarks", {}) or {}

    defaults = Settings(project_root=project_root)
    # Legacy fallback: a pre-map config nested the name under `benchmark.name`.
    legacy_name = (raw.get("benchmark", {}) or {}).get("name")
    active = benchmark or raw.get("default_benchmark") or legacy_name or defaults.benchmark
    block = benchmarks.get(active, {}) or {}

    def pick(key: str, section: dict, default):
        """A per-benchmark block value overrides a cross-cutting section value overrides the default."""
        return block.get(key, section.get(key, default))

    # Artifact paths are derived from the active name unless a block explicitly overrides them.
    program_file = Path(block.get("program", f"programs/{active}.md"))
    seed_config = Path(block.get("seed_config", f"seeds/{active}.yaml"))
    runs_dir = Path(block.get("runs_dir", f"runs/{active}"))
    pipeline = Path(block.get("pipeline", f"pipelines/{active}.yaml"))

    return Settings(
        project_root=project_root,
        seed_config=seed_config,
        approved_models=tuple(pick("approved_models", target, defaults.approved_models)),
        max_agent_steps_range=tuple(pick("max_agent_steps_range", target, defaults.max_agent_steps_range)),
        benchmark=str(active),
        smoke_size=int(pick("smoke_size", target, defaults.smoke_size)),
        per_task_timeout=float(pick("per_task_timeout", target, defaults.per_task_timeout)),
        step_cost_lambda=float(pick("step_cost_lambda", target, defaults.step_cost_lambda)),
        epsilon=float(pick("epsilon", optimizer, defaults.epsilon)),
        iterations=int(pick("iterations", optimizer, defaults.iterations)),
        meta_model=str(pick("meta_model", optimizer, defaults.meta_model)),
        meta_max_steps=int(pick("meta_max_steps", optimizer, defaults.meta_max_steps)),
        hitl=bool(pick("hitl", optimizer, defaults.hitl)),
        allow_create_component=bool(
            pick("allow_create_component", optimizer, defaults.allow_create_component)
        ),
        runs_dir=runs_dir,
        program_file=program_file,
        pipeline=pipeline,
    )
