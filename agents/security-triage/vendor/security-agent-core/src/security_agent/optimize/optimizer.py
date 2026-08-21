"""The deterministic outer loop: hill-climbing over target-agent configs.

The meta-agent proposes one candidate per iteration; this module — never the LLM —
scores candidates on the selected benchmark's train split, applies the epsilon
acceptance rule, promotes champions, and journals every step. Resume is journal-based:
rerunning the campaign continues from the recorded iteration count.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import yaml

from haystack.dataclasses import ChatMessage

from security_agent.benchmarks import get_benchmark
from security_agent.benchmarks.base import Benchmark, ScoreReport, Task, smoke, split
from security_agent.optimize.config_patch import PatchPolicy
from security_agent.optimize.harness.runner import HarnessError, run_tasks
from security_agent.optimize.meta.agent_factory import build_meta_agent
from security_agent.optimize.meta.session import MetaSession, set_session
from security_agent.optimize.settings import Settings
from security_agent.optimize.store import RunStore
from security_agent.optimize.tool_catalog import get_catalog


def make_policy(settings: Settings) -> PatchPolicy:
    return PatchPolicy(
        approved_models=settings.approved_models,
        max_agent_steps_range=settings.max_agent_steps_range,
    )


def get_bench(settings: Settings) -> Benchmark:
    return get_benchmark(settings.benchmark)


def load_seed_config(settings: Settings) -> dict:
    """The benchmark's committed seed agent (seeds/<name>.yaml) as a dict — the source of
    truth the tool catalog is built from."""
    seed_path = settings.path(settings.seed_config)
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Seed config {seed_path} not found. Commit a seed at seeds/{settings.benchmark}.yaml."
        )
    return yaml.safe_load(seed_path.read_text())


def load_train_tasks(settings: Settings, benchmark: Benchmark | None = None) -> list[Task]:
    benchmark = benchmark or get_bench(settings)
    return split(benchmark.load_tasks(), "train")


def score_config(
    config: dict, tasks: list[Task], settings: Settings, benchmark: Benchmark
) -> ScoreReport:
    results = run_tasks(benchmark, config, tasks, settings)
    return benchmark.score(
        tasks,
        results,
        step_cost_lambda=settings.step_cost_lambda,
        max_agent_steps=config.get("init_parameters", {}).get("max_agent_steps", 50),
    )


def ensure_champion(store: RunStore, settings: Settings) -> dict[str, Any]:
    """Score and install the seed config as champion if no champion exists yet."""
    champion = store.champion()
    if champion is not None:
        return champion
    benchmark = get_bench(settings)
    config = load_seed_config(settings)
    digest = store.save_config(config)
    report = score_config(config, load_train_tasks(settings, benchmark), settings, benchmark)
    store.set_champion(digest, report.score, report.metrics)
    store.append_journal(
        {
            "type": "seed",
            "hash": digest,
            "score": report.score,
            "metrics": report.metrics,
            "benchmark": benchmark.name,
            "timestamp": time.time(),
        }
    )
    return store.champion()


def _iteration_context(session: MetaSession, program: str) -> str:
    return (
        f"{program}\n\n"
        "## Current state\n\n"
        f"Benchmark: {session.benchmark.name}. "
        f"Champion config hash: {session.champion_hash}, train score: {session.champion_score:.3f}.\n"
        "The working config starts as a copy of the champion. Improve it and submit a candidate.\n"
    )


def run_iteration(store: RunStore, settings: Settings, iteration: int) -> dict[str, Any]:
    champion = store.champion()
    benchmark = get_bench(settings)
    tasks = benchmark.load_tasks()
    train = split(tasks, "train")
    seed_config = load_seed_config(settings)
    session = MetaSession(
        store=store,
        settings=settings,
        catalog=get_catalog(seed_config, store.extensions()),
        policy=make_policy(settings),
        working_config=copy.deepcopy(store.load_champion_config()),
        champion_hash=champion["hash"],
        champion_score=champion["score"],
        smoke_tasks=smoke(tasks, settings.smoke_size),
        benchmark=benchmark,
        seed_config=seed_config,
    )
    set_session(session)
    try:
        meta_agent = build_meta_agent(settings)
        meta_agent.warm_up()
        program = settings.path(settings.program_file).read_text()
        result = meta_agent.run(messages=[ChatMessage.from_user(_iteration_context(session, program))])

        record: dict[str, Any] = {
            "type": "iteration",
            "iteration": iteration,
            "benchmark": benchmark.name,
            "champion_hash": session.champion_hash,
            "champion_score": session.champion_score,
            "ops": session.applied_ops,
            "summary": session.submitted_summary,
            "meta_steps": result.get("step_count"),
            "meta_token_usage": result.get("token_usage"),
            "trace": session.trace,
            "timestamp": time.time(),
        }

        if not session.submitted:
            record.update(accepted=False, reason="no candidate submitted within the step budget")
            store.append_journal(record)
            return record

        candidate = session.working_config
        digest = store.save_config(candidate)
        record["hash"] = digest
        try:
            report = score_config(candidate, train, settings, benchmark)
        except HarnessError as e:
            record.update(accepted=False, reason=f"benchmark failed: {e}")
            store.append_journal(record)
            return record

        accepted = report.score > session.champion_score + settings.epsilon
        record.update(
            accepted=accepted,
            score=report.score,
            metrics=report.metrics,
            mean_steps=report.mean_steps,
            failed_tasks=[
                f"{r['task_id']}: expected {r['expected']}, got {r['answer']!r}"
                for r in report.individual
                if not r["match"]
            ],
            # Per-task cost, so `optimize status` shows where wall-clock goes (agent loop vs. the
            # untraced verify build). Keys are absent for benchmarks that don't record them -> None.
            timings=[
                {
                    "task_id": r.get("task_id"),
                    "agent_s": r.get("seconds"),
                    "verify_s": r.get("verify_seconds"),
                }
                for r in report.individual
            ],
        )
        if accepted:
            store.set_champion(digest, report.score, report.metrics)
        store.append_journal(record)
        return record
    finally:
        set_session(None)


def run_campaign(store: RunStore, settings: Settings, iterations: int | None = None) -> list[dict[str, Any]]:
    ensure_champion(store, settings)
    iterations = iterations or settings.iterations
    start = store.next_iteration()
    records = []
    for i in range(start, start + iterations):
        record = run_iteration(store, settings, iteration=i)
        status = "ACCEPTED" if record.get("accepted") else "rejected"
        timings = record.get("timings") or []
        agent_s = sum(t["agent_s"] for t in timings if t.get("agent_s") is not None)
        verify_s = sum(t["verify_s"] for t in timings if t.get("verify_s") is not None)
        cost = f" | agent~{agent_s:.0f}s verify~{verify_s:.0f}s" if (agent_s or verify_s) else ""
        print(
            f"[iteration {i}] {status} score={record.get('score')} "
            f"(champion {record.get('champion_score')}) reason={record.get('reason', '')}{cost}"
        )
        records.append(record)
    return records


def holdout_report(store: RunStore, settings: Settings) -> str:
    """Score seed and champion on the holdout split; the overfitting check for the campaign."""
    benchmark = get_bench(settings)
    holdout = split(benchmark.load_tasks(), "holdout")
    journal = store.read_journal()
    seed_records = [r for r in journal if r.get("type") == "seed"]
    if not seed_records:
        raise RuntimeError("No seed record in the journal. Run `secagent optimize init` first.")
    champion = store.champion()

    lines = [f"Holdout report on '{benchmark.name}' (never used for hill-climbing):", ""]
    for label, digest in (("seed", seed_records[0]["hash"]), ("champion", champion["hash"])):
        report = score_config(store.load_config(digest), holdout, settings, benchmark)
        lines.append(f"{label} ({digest}):")
        lines.append("  " + report.summary().replace("\n", "\n  "))
        lines.append("")
    return "\n".join(lines)
