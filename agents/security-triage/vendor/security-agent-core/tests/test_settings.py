"""`load_settings` resolves the active benchmark from the per-benchmark YAML map and derives all
per-benchmark artifact paths by convention (seeds/<name>.yaml, runs/<name>/, pipelines/<name>.yaml,
programs/<name>.md). The benchmark is the single selector: `--benchmark` (the `benchmark=` arg) wins
over `default_benchmark`, and choosing it flips every knob and every path together.
"""

import importlib
from pathlib import Path

import pytest

from security_agent.optimize.settings import load_settings


def test_config_module_is_gone():
    """config.py was dissolved into paths.py + the seed YAMLs; nothing should import it."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("security_agent.config")

CONFIG = """
default_benchmark: secbench

target:
  approved_models: ["gpt-5.4-mini"]
optimizer:
  meta_model: gpt-5.4
  epsilon: 0.05
  iterations: 7

benchmarks:
  primevul:
    max_agent_steps_range: [1, 8]
    smoke_size: 4
    per_task_timeout: 180
    step_cost_lambda: 0.0
  secbench:
    max_agent_steps_range: [10, 60]
    smoke_size: 2
    per_task_timeout: 1800
    epsilon: 0.02
"""


def _write_config(tmp_path: Path) -> Path:
    (tmp_path / "config.yaml").write_text(CONFIG)
    return tmp_path


def test_default_benchmark_selects_secbench_block_and_paths(tmp_path):
    root = _write_config(tmp_path)
    s = load_settings(root)  # no --benchmark -> default_benchmark
    assert s.benchmark == "secbench"
    # per-benchmark knobs from the secbench block
    assert s.max_agent_steps_range == (10, 60)
    assert s.smoke_size == 2
    assert s.per_task_timeout == 1800.0
    # block overrides the optimizer-section epsilon; iterations falls back to the section value
    assert s.epsilon == 0.02
    assert s.iterations == 7
    # derived paths keyed by the active name
    assert s.seed_config == Path("seeds/secbench.yaml")
    assert s.runs_dir == Path("runs/secbench")
    assert s.pipeline == Path("pipelines/secbench.yaml")
    assert s.program_file == Path("programs/secbench.md")


def test_benchmark_arg_overrides_default_and_flips_every_path(tmp_path):
    root = _write_config(tmp_path)
    s = load_settings(root, benchmark="primevul")  # --benchmark wins over default_benchmark
    assert s.benchmark == "primevul"
    assert s.max_agent_steps_range == (1, 8)
    assert s.smoke_size == 4
    assert s.per_task_timeout == 180.0
    # primevul has no epsilon in its block -> optimizer-section default
    assert s.epsilon == 0.05
    assert s.seed_config == Path("seeds/primevul.yaml")
    assert s.runs_dir == Path("runs/primevul")
    assert s.pipeline == Path("pipelines/primevul.yaml")
    assert s.program_file == Path("programs/primevul.md")


def test_cross_cutting_defaults_apply_to_every_benchmark(tmp_path):
    root = _write_config(tmp_path)
    for name in ("primevul", "secbench"):
        s = load_settings(root, benchmark=name)
        assert s.approved_models == ("gpt-5.4-mini",)
        assert s.meta_model == "gpt-5.4"


def test_paths_resolve_against_project_root(tmp_path):
    root = _write_config(tmp_path)
    s = load_settings(root, benchmark="secbench")
    assert s.path(s.pipeline) == root / "pipelines" / "secbench.yaml"
    assert s.path(s.runs_dir) == root / "runs" / "secbench"


def test_missing_config_falls_back_to_defaults(tmp_path):
    s = load_settings(tmp_path)  # no config.yaml present
    assert s.benchmark == "primevul"
    assert s.seed_config == Path("seeds/primevul.yaml")
    assert s.pipeline == Path("pipelines/primevul.yaml")


def test_legacy_benchmark_name_key_still_loads(tmp_path):
    # A pre-map config nested the name under benchmark.name; keep it loadable.
    (tmp_path / "config.yaml").write_text("benchmark:\n  name: secbench\n")
    s = load_settings(tmp_path)
    assert s.benchmark == "secbench"
    assert s.runs_dir == Path("runs/secbench")
