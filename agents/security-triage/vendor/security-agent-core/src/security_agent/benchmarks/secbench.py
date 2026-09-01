"""SEC-bench: an *execution-style* vulnerability-patching benchmark.

Given a real C/C++ CVE checked out in a Docker container, the agent edits the source to fix
the vulnerability; success is decided by *running* the patch against SEC-bench's prebuilt image
(compile + the known PoC no longer trips the sanitizer). SEC-bench **owns its runner**:

- `run`  — per task: start the instance's `:patch` container, run the container-coding agent
           (`security_agent.container_tools`) in-process with the live container bound, and
           extract its edits as a git diff. Returns one `TaskResult` per task (`answer` = diff).
- `score` — per task: run the diff against a fresh container via `security_agent.docker_eval`
           and aggregate the pass/fail verdicts into a **resolve rate** (the optimizer's scalar).

`security_agent.optimize.harness.runner.run_tasks` dispatches to `run` because this class defines
it; benchmarks without a `run` method keep using the shared classification subprocess runner.

Tasks are served from a pre-sampled JSONL (`benchmark_data/secbench.jsonl`, built once by
`SecBench.build`) so every optimizer iteration scores the same instances and no HF access is
needed at scoring time. `build` **randomly samples** instances from the HF `cve` split (200
instances), deterministic per `--seed`, and keeps only those with an available prebuilt `:patch`
image so the sampled set is runnable — keep the count small because each instance is a large image.
"""

from __future__ import annotations

import json
import random
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

from security_agent import paths

from .base import ScoreReport, Task, TaskResult

SECBENCH_HF_DATASET = "SEC-bench/SEC-bench"
# Read the CVE split's raw JSONL directly (robust to datasets-server availability).
SECBENCH_HF_DATA_FILE = "data/eval-cve.jsonl"

# Fields persisted to the JSONL (gold_patch is for offline debugging only — never shown to the agent).
_JSONL_FIELDS = (
    "instance_id",
    "repo",
    "project_name",
    "lang",
    "work_dir",
    "sanitizer",
    "base_commit",
    "bug_description",
    "exit_code",
    "gold_patch",
)

_QUESTION_TEMPLATE = """Fix the security vulnerability in the {project_name} project ({repo}).

- Language: {lang}
- Sanitizer that detects the bug: {sanitizer}
- Repository root (your working directory): {work_dir}

Vulnerability description:
{bug_description}

Investigate the source, find the root cause, and apply a MINIMAL fix by editing the affected \
file(s) with your tools. The project must still compile and the known crashing input must no \
longer trip the {sanitizer} sanitizer. Reply with a short summary when you are done."""


class SecBench:
    name = "secbench"

    def __init__(self, data_path: Path | None = None, mode: str = "medium") -> None:
        self.data_path = Path(data_path) if data_path else paths.BENCHMARK_DATA / "secbench.jsonl"
        # Patch-verdict strictness (see docker_eval._interpret_patch): strict | medium | generous.
        # Default `medium` = repro exits with the dataset's recorded post-fix `exit_code` and no
        # sanitizer fires. A correct fix often exits NON-zero (the program now cleanly rejects the
        # malicious input), so `strict` (demand exit 0) wrongly fails such instances — confirmed
        # against the gold patches (mruby/openjpeg fix -> exit 1, njs fix -> exit 0).
        self.mode = mode

    # --- loading (used by run + score) ---

    def load_tasks(self) -> list[Task]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"No benchmark data at {self.data_path}. "
                "Run `secagent build-benchmark --benchmark secbench` first."
            )
        tasks: list[Task] = []
        for line_no, line in enumerate(self.data_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            split = record.get("split", "train")
            if split not in ("train", "holdout"):
                raise ValueError(f"{self.data_path}:{line_no}: split must be 'train' or 'holdout'.")
            tasks.append(
                Task(
                    id=str(record["instance_id"]),
                    question=_render_question(record),
                    split=split,
                    gold={
                        "instance_id": record["instance_id"],
                        "work_dir": record.get("work_dir"),
                        "exit_code": record.get("exit_code"),
                        "sanitizer": record.get("sanitizer"),
                        "gold_patch": record.get("gold_patch"),
                    },
                )
            )
        if not tasks:
            raise ValueError(f"No tasks in {self.data_path}.")
        return tasks

    # --- running the agent (SEC-bench owns its runner) ---

    def run(self, config_dict: dict, tasks: list[Task], settings) -> list[TaskResult]:
        """Run the container-coding agent on each task and return its patch as the answer.

        Runs the agent in-process (not the shared subprocess) because the live Docker handle
        can't cross a serialized-config boundary; the container is the real isolation boundary.
        """
        from haystack.dataclasses import ChatMessage

        from security_agent import docker_eval
        from security_agent.container_tools import use_container
        from security_agent.optimize.validate import load_agent

        results: list[TaskResult] = []
        for task in tasks:
            gold = task.gold
            started = time.monotonic()
            diff, steps, error = "", None, None
            try:
                image = docker_eval.ensure_image(gold["instance_id"], "patch")
                with docker_eval.start_container(image, gold["work_dir"]) as handle:
                    agent = load_agent(config_dict)
                    agent.warm_up()
                    with use_container(handle):
                        try:
                            out = agent.run(messages=[ChatMessage.from_user(task.question)])
                            steps = out.get("step_count")
                        except Exception as e:  # noqa: BLE001 - a task failure must not abort the run
                            error = f"{type(e).__name__}: {e}"
                    diff = handle.git_diff()
            except Exception as e:  # noqa: BLE001 - Docker/image failures become a failed task
                error = f"{type(e).__name__}: {e}"
            results.append(
                TaskResult(
                    task_id=task.id,
                    answer=diff,
                    error=error,
                    steps=steps,
                    seconds=round(time.monotonic() - started, 2),
                )
            )
        return results

    # --- scoring (the objective) ---

    def score(
        self,
        tasks: list[Task],
        results: list[TaskResult],
        *,
        step_cost_lambda: float = 0.0,
        max_agent_steps: int = 50,
    ) -> ScoreReport:
        from security_agent import docker_eval

        if len(tasks) != len(results):
            raise ValueError("tasks and results must be aligned.")

        individual: list[dict] = []
        resolved = 0
        for task, result in zip(tasks, results, strict=True):
            gold = task.gold
            diff = result.answer or ""
            verify_seconds = None
            if result.error and not diff.strip():
                success, reason = False, result.error
            else:
                started = time.monotonic()
                success, reason, _logs = docker_eval.verify_patch(
                    gold["instance_id"], diff, gold, mode=self.mode
                )
                # verify_patch runs patch->build->repro in a FRESH container; this is the Docker cost
                # the OTLP tracer can't see (it only wraps the in-process agent run).
                verify_seconds = round(time.monotonic() - started, 1)
            resolved += int(success)
            individual.append(
                {
                    "task_id": task.id,
                    "expected": "resolved",
                    "answer": reason,
                    "match": success,
                    "steps": result.steps,
                    "seconds": result.seconds,  # agent wall-clock (from SecBench.run)
                    "verify_seconds": verify_seconds,
                    "error": result.error,
                }
            )

        n = len(tasks)
        resolve_rate = resolved / n if n else 0.0
        steps = [r.steps for r in results if r.steps is not None]
        mean_steps = sum(steps) / len(steps) if steps else 0.0
        penalty = step_cost_lambda * (mean_steps / max_agent_steps) if max_agent_steps else 0.0
        metrics = {"n": n, "resolved": resolved, "resolve_rate": resolve_rate, "mode": self.mode}
        return ScoreReport(
            score=max(0.0, resolve_rate - penalty),
            metrics=metrics,
            mean_steps=mean_steps,
            individual=individual,
        )

    # --- building the pre-sampled JSONL from the Hugging Face Hub ---

    @classmethod
    def build(
        cls,
        *,
        train_pairs: int,
        holdout_pairs: int,
        seed: int = 0,
        out_path: Path | None = None,
        verify_images: bool = True,
    ) -> Path:
        """Randomly sample instances from the HF `cve` split and write the task JSONL.

        `train_pairs`/`holdout_pairs` are interpreted as instance *counts* (SEC-bench is not paired):
        `train_pairs` sampled instances go to train, the next `holdout_pairs` to holdout. The sample
        is a deterministic function of `seed` (same seed -> same instances). When `verify_images`
        (default), an instance is kept only if its prebuilt `:patch` image is available on the
        registry — sampled instances without an image are skipped so the built set is runnable
        (missing images would otherwise just score 0 every iteration).
        """
        out_path = Path(out_path) if out_path else paths.BENCHMARK_DATA / "secbench.jsonl"
        rows = _load_all_rows()
        if not rows:
            raise RuntimeError(
                f"No instances found in {SECBENCH_HF_DATASET} ({SECBENCH_HF_DATA_FILE})."
            )
        is_available: Callable[[str], bool] = _image_available if verify_images else (lambda _iid: True)
        train, holdout, skipped = _sample_split(rows, train_pairs, holdout_pairs, seed, is_available)

        if skipped:
            warnings.warn(
                f"Skipped {len(skipped)} sampled instance(s) with no available :patch image: "
                f"{sorted(skipped)}"
            )
        got, want = len(train) + len(holdout), train_pairs + holdout_pairs
        if got == 0:
            raise RuntimeError(
                "No sampled instances had an available :patch image; try a different --seed "
                "or pass verify_images=False to keep them anyway."
            )
        if got < want:
            warnings.warn(
                f"Only {got} of {want} requested instances had an available image; built with {got}."
            )
        records = [_record(r, "train") for r in train] + [_record(r, "holdout") for r in holdout]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return out_path


def _render_question(record: dict) -> str:
    return _QUESTION_TEMPLATE.format(
        project_name=record.get("project_name") or record.get("instance_id"),
        repo=record.get("repo") or "?",
        lang=record.get("lang") or "c",
        sanitizer=record.get("sanitizer") or "the",
        work_dir=record.get("work_dir") or "/src",
        bug_description=(record.get("bug_description") or "").strip() or "(no description provided)",
    )


def _record(row: dict, split: str) -> dict:
    record = {key: row.get(key) if key != "gold_patch" else row.get("patch") for key in _JSONL_FIELDS}
    record["split"] = split
    return record


def _load_all_rows() -> list[dict]:
    """Load every row of the HF `cve` split (read the raw JSONL directly)."""
    from datasets import load_dataset  # heavy import; only needed at build time

    ds = load_dataset(SECBENCH_HF_DATASET, data_files=SECBENCH_HF_DATA_FILE, split="train")
    return [dict(row) for row in ds]


def _image_available(instance_id: str) -> bool:
    """Whether the instance's prebuilt `:patch` image exists on the registry (no pull, just a
    manifest lookup). Any lookup failure is treated as unavailable so the built set stays runnable."""
    from security_agent import docker_eval

    try:
        docker_eval._client().images.get_registry_data(docker_eval.image_ref(instance_id, "patch"))
        return True
    except Exception:  # noqa: BLE001 - NotFound / auth / network all mean "don't sample this one"
        return False


def _sample_split(
    rows: list[dict],
    train_pairs: int,
    holdout_pairs: int,
    seed: int,
    is_available: Callable[[str], bool],
) -> tuple[list[dict], list[dict], list[str]]:
    """Deterministically shuffle `rows` by `seed`, keep instances whose image `is_available`, and
    split the first `train_pairs + holdout_pairs` kept rows into (train, holdout). Also returns the
    instance ids skipped for having no image (only those encountered before the quota was filled)."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    want = train_pairs + holdout_pairs
    chosen: list[dict] = []
    skipped: list[str] = []
    for row in shuffled:
        if len(chosen) >= want:
            break
        if is_available(row.get("instance_id")):
            chosen.append(row)
        else:
            skipped.append(row.get("instance_id"))
    return chosen[:train_pairs], chosen[train_pairs : train_pairs + holdout_pairs], skipped
