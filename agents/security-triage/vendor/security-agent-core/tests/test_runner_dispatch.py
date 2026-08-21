"""The dispatch seam: `run_tasks` delegates to a benchmark's own `run` when present,
otherwise falls back to the shared classification runner (`run_candidate`)."""

from types import SimpleNamespace

from security_agent.benchmarks.base import TaskResult
from security_agent.optimize.harness import runner


def test_run_tasks_delegates_to_benchmark_run():
    sentinel = [TaskResult(task_id="x", answer="diff")]

    class ExecBenchmark:
        name = "exec"

        def run(self, config, tasks, settings):
            assert config == {"k": "v"}
            return sentinel

    out = runner.run_tasks(ExecBenchmark(), {"k": "v"}, ["t"], SimpleNamespace(per_task_timeout=1.0))
    assert out is sentinel


def test_run_tasks_falls_back_to_run_candidate(monkeypatch):
    captured = {}

    def fake_run_candidate(config, tasks, per_task_timeout):
        captured["args"] = (config, tasks, per_task_timeout)
        return ["fallback"]

    monkeypatch.setattr(runner, "run_candidate", fake_run_candidate)

    class ClassificationBenchmark:
        name = "clf"  # no `run` method

    out = runner.run_tasks(
        ClassificationBenchmark(), {"c": 1}, ["a", "b"], SimpleNamespace(per_task_timeout=42.0)
    )
    assert out == ["fallback"]
    assert captured["args"] == ({"c": 1}, ["a", "b"], 42.0)
