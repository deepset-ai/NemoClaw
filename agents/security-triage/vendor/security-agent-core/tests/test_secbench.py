"""Offline unit tests for the SecBench benchmark: task loading and score aggregation.

Docker verification is stubbed (`docker_eval.verify_patch`), so these run without a daemon.
"""

import json

import pytest

from security_agent.benchmarks.base import Task, TaskResult
from security_agent.benchmarks.secbench import SecBench

_ROW = {
    "instance_id": "njs.cve-2022-32414",
    "repo": "nginx/njs",
    "project_name": "njs",
    "lang": "c++",
    "work_dir": "/src/njs",
    "sanitizer": "address",
    "base_commit": "deadbeef",
    "bug_description": "SEGV in the interpreter on crafted input.",
    "exit_code": 0,
    "gold_patch": "--- a\n+++ b\n",
}


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_tasks_renders_question_and_gold(tmp_path):
    data = _write_jsonl(
        tmp_path / "secbench.jsonl",
        [dict(_ROW, split="train"), dict(_ROW, instance_id="mruby.cve-2022-0240", split="holdout")],
    )
    tasks = SecBench(data_path=data).load_tasks()
    assert [t.split for t in tasks] == ["train", "holdout"]

    t = tasks[0]
    assert t.id == "njs.cve-2022-32414"
    assert t.gold["work_dir"] == "/src/njs"
    assert t.gold["exit_code"] == 0
    assert t.gold["sanitizer"] == "address"
    # The question surfaces the instance context but never leaks the gold patch.
    assert "njs" in t.question and "address" in t.question and "/src/njs" in t.question
    assert "SEGV in the interpreter" in t.question
    assert "+++ b" not in t.question


def test_load_tasks_rejects_bad_split(tmp_path):
    data = _write_jsonl(tmp_path / "secbench.jsonl", [dict(_ROW, split="validation")])
    with pytest.raises(ValueError, match="split"):
        SecBench(data_path=data).load_tasks()


def test_load_tasks_missing_file():
    with pytest.raises(FileNotFoundError, match="build-benchmark"):
        SecBench(data_path="/nonexistent/secbench.jsonl").load_tasks()


def _task(iid, split="train"):
    return Task(id=iid, question="q", split=split, gold={"instance_id": iid, "work_dir": "/src", "exit_code": 0})


def test_score_resolve_rate(monkeypatch):
    calls = []

    def fake_verify(instance_id, diff, gold, *, mode="strict"):
        calls.append(instance_id)
        # Only the first instance's patch "resolves".
        ok = instance_id == "a"
        return ok, ("resolved" if ok else "sanitizer still triggers after patch"), "logs"

    monkeypatch.setattr("security_agent.docker_eval.verify_patch", fake_verify)

    tasks = [_task("a"), _task("b")]
    results = [
        TaskResult(task_id="a", answer="--- patch a", steps=7),
        TaskResult(task_id="b", answer="--- patch b", steps=9),
    ]
    report = SecBench().score(tasks, results, max_agent_steps=20)

    assert report.metrics == {"n": 2, "resolved": 1, "resolve_rate": 0.5, "mode": "medium"}
    assert report.score == 0.5
    assert report.mean_steps == 8.0
    assert {r["task_id"]: r["match"] for r in report.individual} == {"a": True, "b": False}
    assert calls == ["a", "b"]
    # verify_patch is timed (both tasks had a patch, so verify ran for both).
    assert all(r["verify_seconds"] is not None for r in report.individual)


def test_score_short_circuits_on_error_without_patch(monkeypatch):
    def fake_verify(*a, **k):
        raise AssertionError("verify_patch must not run when there is no patch")

    monkeypatch.setattr("security_agent.docker_eval.verify_patch", fake_verify)

    tasks = [_task("a")]
    results = [TaskResult(task_id="a", answer="", error="RuntimeError: image pull failed")]
    report = SecBench().score(tasks, results)
    assert report.metrics["resolved"] == 0
    assert report.individual[0]["match"] is False
    assert "image pull failed" in report.individual[0]["answer"]
    # No patch -> verify_patch never ran -> no verify time recorded.
    assert report.individual[0]["verify_seconds"] is None


def test_sample_split_deterministic_and_filters_unavailable():
    from security_agent.benchmarks.secbench import _sample_split

    rows = [{"instance_id": f"proj.cve-2020-{i:04d}"} for i in range(30)]
    dead = {"proj.cve-2020-0003", "proj.cve-2020-0007"}

    def is_available(iid):
        return iid not in dead

    train, holdout, _skipped = _sample_split(rows, 3, 2, seed=42, is_available=is_available)
    assert len(train) == 3 and len(holdout) == 2
    chosen = {r["instance_id"] for r in train + holdout}
    assert not (chosen & dead)  # unavailable instances are never chosen
    # Same seed -> same sample (the JSONL is built once and must be reproducible).
    t2, h2, _ = _sample_split(rows, 3, 2, seed=42, is_available=is_available)
    assert [r["instance_id"] for r in train + holdout] == [r["instance_id"] for r in t2 + h2]


def test_sample_split_handles_shortfall_when_few_images_available():
    from security_agent.benchmarks.secbench import _sample_split

    rows = [{"instance_id": f"p.cve-{i}"} for i in range(5)]
    train, holdout, skipped = _sample_split(
        rows, 3, 1, seed=0, is_available=lambda iid: iid == "p.cve-2"
    )
    assert len(train) + len(holdout) == 1  # only one runnable instance
    assert set(skipped) == {"p.cve-0", "p.cve-1", "p.cve-3", "p.cve-4"}


def test_score_step_cost_penalty(monkeypatch):
    monkeypatch.setattr(
        "security_agent.docker_eval.verify_patch",
        lambda *a, **k: (True, "resolved", ""),
    )
    tasks = [_task("a")]
    results = [TaskResult(task_id="a", answer="p", steps=20)]
    report = SecBench().score(tasks, results, step_cost_lambda=0.5, max_agent_steps=20)
    # resolve_rate 1.0 - 0.5 * (20/20) = 0.5
    assert report.score == 0.5
