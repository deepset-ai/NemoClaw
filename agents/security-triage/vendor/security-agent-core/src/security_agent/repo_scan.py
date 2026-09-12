"""Scan one code repository for security defects with a YAML-configured Haystack Agent.

This is the deployable shape of `scripts/audit_repo.py`: the same reviewer prompt and the same
`run_bash` / `run_python` / `read_file` / `write_file` tools, but with the agent read from a YAML
file (`seeds/repo-scan.yaml`) and the tools bound to `local_sandbox.LocalSandbox` — because the
intended host is a NemoClaw/OpenShell sandbox, which is itself the isolation boundary and cannot
start Docker containers. Two optional additions: `search_security_kb` over HTTP
(`kb_client.RemoteKbSearch`, to the host-side KB server), and the `code-triage` skill's
`load_skill` / `run_skill_script` tools when a skills directory is given.

The YAML is the thing a meta-agent edits and a human approves; everything in this module is fixed
plumbing. A scan writes one JSON record — parsed report, raw final answer, full trajectory, step
count, timing — so every claimed finding can be checked against the code afterwards.

    python -m security_agent.repo_scan --repo /sandbox/work/myrepo --out /sandbox/.security-triage/reports
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from security_agent import paths, sandbox
from security_agent.container_tools import use_container
from security_agent.local_sandbox import LocalSandbox
from security_agent.optimize.validate import load_agent, register_allowlist
from security_agent.verdict import extract_json_object

DEFAULT_CONFIG = paths.PROJECT_ROOT / "seeds" / "repo-scan.yaml"
CONFIG_ENV = "SECURITY_SCAN_CONFIG"
MODEL_ENV = ("NEMOCLAW_MODEL", "SECURITY_SCAN_MODEL")

SOURCE_SUFFIXES = {".py", ".c", ".h", ".cc", ".cpp", ".hpp", ".go", ".java", ".js", ".ts", ".tsx", ".rs", ".rb", ".php"}
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "venv"}

USER_PROMPT = (
    "Audit the repository `{name}` at {repo} ({n_files} source files, {n_lines} lines). "
    "Report only findings you verified in the code, with file and line."
)


def load_config(path: Optional[Path] = None) -> dict:
    """The agent YAML: an explicit path, else `$SECURITY_SCAN_CONFIG`, else the packaged seed."""
    chosen = path or (Path(os.environ[CONFIG_ENV]) if os.environ.get(CONFIG_ENV) else DEFAULT_CONFIG)
    return yaml.safe_load(Path(chosen).read_text())


def apply_overrides(config: dict, *, model: Optional[str] = None, max_steps: Optional[int] = None) -> dict:
    """Deployment-time overrides that must not be baked into the YAML: the served model name
    (NemoClaw injects it as `NEMOCLAW_MODEL`) and the step budget."""
    config = copy.deepcopy(config)
    params = config["init_parameters"]
    model = model or next((os.environ[name] for name in MODEL_ENV if os.environ.get(name)), None)
    if model:
        params["chat_generator"]["init_parameters"]["model"] = model
    if max_steps:
        params["max_agent_steps"] = max_steps
    return config


def count_sources(repo: Path) -> tuple[int, int]:
    n_files = n_lines = 0
    for path in repo.rglob("*"):
        if any(part in EXCLUDE_DIRS for part in path.relative_to(repo).parts) or not path.is_file():
            continue
        if path.suffix in SOURCE_SUFFIXES:
            n_files += 1
            try:
                n_lines += sum(1 for _ in path.open(errors="replace"))
            except OSError:
                pass
    return n_files, n_lines


def parse_report(text: str) -> Optional[dict]:
    candidate = extract_json_object(text or "")
    if not candidate:
        return None
    try:
        report = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return report if isinstance(report, dict) and "findings" in report else None


def skill_tools(skills_dir: Path) -> list[Any]:
    """`load_skill`/`read_skill_file` plus the constrained `run_skill_script`, when a skills dir exists."""
    from haystack.skill_stores.file_system import FileSystemSkillStore
    from haystack.tools import SkillToolset

    from security_agent.skill_script_tool import make_skill_script_tool

    return [SkillToolset(FileSystemSkillStore(skills_dir)), make_skill_script_tool(skills_dir, auto_venv=False)]


def scan(
    config: dict,
    repo: Path,
    *,
    skills_dir: Optional[Path] = None,
    name: Optional[str] = None,
) -> dict:
    """Run one scan and return the JSON-able record. Never raises for agent-side failures."""
    repo = repo.resolve()
    name = name or repo.name
    started = time.monotonic()
    record: dict[str, Any] = {"repo": name, "path": str(repo), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    n_files, n_lines = count_sources(repo)
    record.update(n_files=n_files, n_lines=n_lines)

    register_allowlist()
    params = config["init_parameters"]
    params["system_prompt"] = params["system_prompt"].replace("{repo}", str(repo))
    agent = load_agent(config)
    if skills_dir and Path(skills_dir).is_dir():
        # Appended after deserialization: `run_skill_script` wraps a function, so it has no YAML form.
        agent.tools = [*(agent.tools or []), *skill_tools(Path(skills_dir))]
        record["skills_dir"] = str(skills_dir)
    collector = sandbox.TrajectoryCollector()
    collector.attach(agent)
    agent.warm_up()

    # The repo is the working directory, so relative paths in tool calls resolve against it; the
    # prompt sends scratch files to /tmp.
    handle = LocalSandbox(work_dir=str(repo))
    with use_container(handle):
        try:
            out = agent.run(
                messages=[_user_message(USER_PROMPT.format(name=name, repo=repo, n_files=n_files, n_lines=n_lines))]
            )
            if not sandbox.last_message_text(handle, out):
                out = sandbox.force_final_answer(agent, out, sandbox.FINAL_ANSWER_PROMPT)
                record["forced_final_answer"] = True
            record["steps"] = out.get("step_count")
            record["raw_answer"] = sandbox.last_message_text(handle, out)
            record["trajectory"] = sandbox.serialize_messages(out.get("messages"))
        except Exception as exc:  # noqa: BLE001 - a failed scan is a record with an error, not a crash
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["trajectory"] = collector.messages
    record["seconds"] = round(time.monotonic() - started, 1)
    record["report"] = parse_report(record.get("raw_answer", ""))
    record["model"] = params["chat_generator"]["init_parameters"].get("model")
    return record


def _user_message(text: str):
    from haystack.dataclasses import ChatMessage

    return ChatMessage.from_user(text)


def summary_line(record: dict) -> str:
    report = record.get("report") or {}
    findings = report.get("findings") or []
    status = record.get("error") or ("ok" if report else "NO-REPORT")
    detail = ", ".join(f"{f.get('severity')}:{f.get('title')}" for f in findings)[:300]
    return f"[{record['repo']}] {status}  steps={record.get('steps')}  {record.get('seconds', 0):.0f}s  findings={len(findings)} {detail}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True, help="repository to scan (read as evidence)")
    parser.add_argument("--out", type=Path, required=True, help="directory for <name>.json")
    parser.add_argument("--config", type=Path, default=None, help=f"agent YAML (default: ${CONFIG_ENV} or seeds/repo-scan.yaml)")
    parser.add_argument("--model", default=None, help="served model name (default: $NEMOCLAW_MODEL, else the YAML's)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skills-dir", type=Path, default=None, help="skills directory with the code-triage skill")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), model=args.model, max_steps=args.max_steps)
    record = scan(config, args.repo, skills_dir=args.skills_dir, name=args.name)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{record['repo']}.json"
    target.write_text(json.dumps(record, indent=2, default=str))
    print(summary_line(record), flush=True)
    print(f"report: {target}", file=sys.stderr, flush=True)
    return 0 if not record.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
