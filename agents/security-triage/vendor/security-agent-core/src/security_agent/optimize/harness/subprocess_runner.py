"""Entry point executed inside the benchmark subprocess.

Usage: ``python -m security_agent.optimize.harness.subprocess_runner <config.yaml> <tasks.json> <out.json>``

This process is the only place candidate configs (which may reference generated
components) are executed. The parent enforces a wall-clock timeout and treats any
crash as a failed run; per-task exceptions become score-0 results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

from haystack.dataclasses import ChatMessage

from security_agent.optimize.validate import load_agent


def _jsonable(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def main() -> int:
    config_path, tasks_path, out_path = sys.argv[1:4]
    config = yaml.safe_load(Path(config_path).read_text())
    tasks = json.loads(Path(tasks_path).read_text())

    agent = load_agent(config)
    agent.warm_up()

    results = []
    for task in tasks:
        started = time.monotonic()
        record: dict = {"task_id": task["id"]}
        try:
            output = agent.run(messages=[ChatMessage.from_user(task["question"])])
            record["answer"] = output["last_message"].text or ""
            record["steps"] = output.get("step_count")
            record["token_usage"] = _jsonable(output.get("token_usage"))
        except Exception as e:  # noqa: BLE001 - a task failure must not kill the batch
            record["answer"] = ""
            record["error"] = f"{type(e).__name__}: {e}"
        record["seconds"] = round(time.monotonic() - started, 2)
        results.append(record)

    Path(out_path).write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
