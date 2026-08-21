"""PrimeVul: a function-level vulnerability *classification* benchmark.

Given one C/C++ function, the agent predicts `is_vulnerable` + CWE class and emits a
JSON verdict. Scored by **paired accuracy** — PrimeVul's `paired` config lists each
vulnerable function next to its fixed counterpart, and a pair counts only when both are
classified correctly (this defeats the "always say vulnerable" degenerate policy);
falls back to binary F1 when a task set has no complete pairs.

Tasks are served from a pre-sampled JSONL file (``benchmark_data/primevul.jsonl``, built
once by `PrimeVul.build`) so every optimizer iteration scores the same tasks and no
network is needed at scoring time.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from security_agent import paths
from security_agent.evaluate import aggregate, task_result
from security_agent.verdict import parse_verdict

from .base import ScoreReport, Task, TaskResult

# PrimeVul on the Hugging Face Hub. The `paired` config ships each vulnerable function
# immediately followed by its fixed counterpart; our split names map to HF's.
PRIMEVUL_HF_DATASET = "colin/PrimeVul"
PRIMEVUL_HF_CONFIG = "paired"
_PRIMEVUL_HF_SPLIT = {"val": "validation", "heldout": "test", "train": "train"}

_JSONL_FIELDS = ("id", "code", "lang", "gold_is_vulnerable", "gold_cwe", "pair_id")

# Framing of the per-task user message (benchmark task rendering, not agent shaping).
# `{lang}` and `{code}` are filled per task. The agent's own system prompt lives in the seed YAML.
INPUT_TEMPLATE = """Analyze the following {lang} function for security \
vulnerabilities and respond with your triage verdict as JSON.

```{lang}
{code}
```"""


def _input_message(code: str, lang: str = "c") -> str:
    """Render the per-task user message for one PrimeVul function."""
    return INPUT_TEMPLATE.format(lang=lang, code=code)


class PrimeVul:
    name = "primevul"

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else paths.BENCHMARK_DATA / "primevul.jsonl"

    # --- loading (used by the harness + eval) ---

    def load_tasks(self) -> list[Task]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"No benchmark data at {self.data_path}. "
                "Run `secagent build-benchmark --benchmark primevul` first."
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
                    id=str(record["id"]),
                    question=_input_message(record["code"], record.get("lang", "c")),
                    split=split,
                    gold={
                        "is_vulnerable": bool(record["gold_is_vulnerable"]),
                        "cwe": record.get("gold_cwe"),
                        "pair_id": record.get("pair_id"),
                    },
                )
            )
        if not tasks:
            raise ValueError(f"No tasks in {self.data_path}.")
        return tasks

    # --- scoring (the objective) ---

    def score(
        self,
        tasks: list[Task],
        results: list[TaskResult],
        *,
        step_cost_lambda: float = 0.0,
        max_agent_steps: int = 50,
    ) -> ScoreReport:
        if len(tasks) != len(results):
            raise ValueError("tasks and results must be aligned.")

        scored_items: list[dict] = []
        individual: list[dict] = []
        for task, result in zip(tasks, results, strict=True):
            verdict = parse_verdict(result.answer or "")
            gold = {
                "gold_is_vulnerable": task.gold.get("is_vulnerable"),
                "gold_cwe": task.gold.get("cwe"),
            }
            scored = task_result(verdict, gold)
            scored_items.append({"pair_id": task.gold.get("pair_id"), "scored": scored})

            pred_v = scored["pred_is_vulnerable"]
            match = pred_v is not None and bool(pred_v) == bool(task.gold.get("is_vulnerable"))
            individual.append(
                {
                    "task_id": task.id,
                    "expected": _gold_str(task),
                    "answer": _pred_str(verdict),
                    "match": match,
                    "steps": result.steps,
                    "error": result.error,
                }
            )

        agg = aggregate(scored_items)
        binary = agg.get("binary") or {}
        paired = binary.get("paired_accuracy")
        objective = paired if paired is not None else binary.get("f1", 0.0)

        steps = [r.steps for r in results if r.steps is not None]
        mean_steps = sum(steps) / len(steps) if steps else 0.0
        score = objective - step_cost_lambda * (mean_steps / max_agent_steps)

        return ScoreReport(
            score=max(0.0, score), metrics=agg, mean_steps=mean_steps, individual=individual
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
    ) -> Path:
        out_path = Path(out_path) if out_path else paths.BENCHMARK_DATA / "primevul.jsonl"
        records = _records("train", "train", train_pairs, seed) + _records(
            "holdout", "val", holdout_pairs, seed
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return out_path


def _gold_str(task: Task) -> str:
    if not task.gold.get("is_vulnerable"):
        return "benign"
    return f"vulnerable/{task.gold.get('cwe') or 'CWE-?'}"


def _pred_str(verdict: dict) -> str:
    if verdict.get("parse_error"):
        return "PARSE-ERROR"
    if not verdict.get("is_vulnerable"):
        return "benign"
    return f"vulnerable/{verdict.get('cwe') or 'CWE-?'}"


def _records(out_split: str, primevul_split: str, pairs: int, seed: int) -> list[dict]:
    raw = _load_primevul(primevul_split)
    raw = _sample_pairs(raw, pairs, seed)
    out = []
    for task in raw:
        record = {key: task.get(key) for key in _JSONL_FIELDS}
        record["split"] = out_split
        out.append(record)
    return out


def _load_primevul(split: Optional[str]) -> list[dict]:
    """Load the PrimeVul *paired* benchmark from the Hugging Face Hub. Consecutive rows
    form one (vulnerable, fixed) pair; `i // 2` groups them under one `pair_id`."""
    from datasets import load_dataset  # heavy import; only needed at build time

    our_split = split or "val"
    hf_split = _PRIMEVUL_HF_SPLIT.get(our_split, our_split)
    ds = load_dataset(PRIMEVUL_HF_DATASET, PRIMEVUL_HF_CONFIG, split=hf_split)

    tasks: list[dict] = []
    for i, row in enumerate(ds):
        cwe = row.get("cwe")
        if isinstance(cwe, (list, tuple)):  # PrimeVul ships `cwe` as a sequence.
            cwe = cwe[0] if cwe else None
        is_vuln = bool(row.get("target"))
        tasks.append(
            {
                "id": str(row.get("idx", i)),
                "pair_id": f"primevul-{our_split}-{i // 2}",
                "lang": "c",
                "code": row.get("func", ""),
                "gold_is_vulnerable": is_vuln,
                "gold_cwe": (cwe if is_vuln else None),
                "split": our_split,
            }
        )
    return tasks


def _sample_pairs(tasks: list[dict], n_pairs: int, seed: int) -> list[dict]:
    """Reproducibly pick `n_pairs` whole pairs (grouped by `pair_id`), keeping both
    members so the paired metric stays valid. Original ordering is preserved."""
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for t in tasks:
        groups.setdefault(t["pair_id"], []).append(t)
    pair_ids = list(groups.keys())
    if n_pairs >= len(pair_ids):
        return tasks
    chosen = set(random.Random(seed).sample(pair_ids, n_pairs))
    return [t for pid in pair_ids if pid in chosen for t in groups[pid]]
