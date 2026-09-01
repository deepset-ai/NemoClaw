"""Run a candidate config over benchmark tasks in an isolated subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from security_agent.benchmarks.base import Task, TaskResult


class HarnessError(RuntimeError):
    """The benchmark subprocess failed as a whole (crash or timeout)."""


def run_tasks(benchmark, config: dict, tasks: list[Task], settings) -> list[TaskResult]:
    """Produce aligned `TaskResult`s for `tasks`, letting a benchmark own its runner.

    If the benchmark defines `run` (an *execution-style* benchmark such as SEC-bench, which
    drives the agent inside a Docker sandbox), delegate to it. Otherwise use the shared
    classification runner below: one user message in, the agent's final text out, isolated in
    a subprocess. This is the single dispatch point used by the optimizer, the meta-agent's
    smoke test, and the CLI eval.
    """
    runner = getattr(benchmark, "run", None)
    if callable(runner):
        return runner(config, tasks, settings)
    return run_candidate(config, tasks, settings.per_task_timeout)


def run_candidate(config: dict, tasks: list[Task], per_task_timeout: float) -> list[TaskResult]:
    """
    Execute the target agent described by `config` on each task, in a subprocess.

    :raises HarnessError: If the subprocess crashes or exceeds the batch timeout.
    """
    with tempfile.TemporaryDirectory(prefix="security_agent.optimize-run-") as tmp:
        tmp_path = Path(tmp)
        config_file = tmp_path / "config.yaml"
        tasks_file = tmp_path / "tasks.json"
        out_file = tmp_path / "out.json"
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        tasks_file.write_text(json.dumps([{"id": t.id, "question": t.question} for t in tasks]))

        timeout = per_task_timeout * len(tasks) + 60
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "security_agent.optimize.harness.subprocess_runner",
                    str(config_file),
                    str(tasks_file),
                    str(out_file),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise HarnessError(f"Benchmark subprocess timed out after {timeout:.0f}s.") from e
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.splitlines()[-15:])
            raise HarnessError(f"Benchmark subprocess failed (exit {proc.returncode}):\n{tail}")

        raw = json.loads(out_file.read_text())

    by_id = {r["task_id"]: r for r in raw}
    return [
        TaskResult(
            task_id=t.id,
            answer=by_id.get(t.id, {}).get("answer", ""),
            error=by_id.get(t.id, {}).get("error"),
            steps=by_id.get(t.id, {}).get("steps"),
            seconds=by_id.get(t.id, {}).get("seconds"),
            token_usage=by_id.get(t.id, {}).get("token_usage"),
        )
        for t in tasks
    ]
