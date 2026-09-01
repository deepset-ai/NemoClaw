"""The meta-agent's tools.

All tools are module-level `@tool` functions so the meta-agent round-trips through
YAML like any other Haystack agent. They operate on the session's *working config*,
a copy of the champion that accumulates validated patches until it is submitted as
a candidate.
"""

import json
from typing import Annotated

import yaml

from haystack.tools import tool

from security_agent.optimize import config_patch
from security_agent.optimize.component_authoring import ComponentAuthoringError, create_component as _create_component
from security_agent.optimize.harness.runner import HarnessError, run_tasks
from security_agent.optimize.meta.session import current_session
from security_agent.optimize.validate import validate_config as _validate_config


@tool
def read_config() -> str:
    """Read the current working config (YAML) that your patches apply to."""
    session = current_session()
    return (
        f"# working config (champion {session.champion_hash}, score {session.champion_score:.3f},"
        f" {len(session.applied_ops)} ops applied)\n" + yaml.safe_dump(session.working_config, sort_keys=False)
    )


@tool
def list_mutable_paths() -> str:
    """List the patch operations and the whitelisted config paths with their bounds."""
    session = current_session()
    return json.dumps(config_patch.describe_ops(session.policy), indent=2)


@tool
def list_tool_catalog() -> str:
    """List all tools in the catalog with their descriptions and whether they are enabled."""
    session = current_session()
    enabled = set(config_patch.enabled_tool_names(session.working_config))
    lines = []
    for name, catalog_tool in sorted(session.catalog.items()):
        status = "enabled" if name in enabled else "disabled"
        lines.append(f"- {name} [{status}]: {catalog_tool.description}")
    return "\n".join(lines)


@tool
def propose_patch(
    ops_json: Annotated[
        str,
        'JSON array of ops, e.g. \'[{"op": "set", "path": "system_prompt", "value": "..."}]\'. '
        "See list_mutable_paths for allowed ops and paths.",
    ],
) -> str:
    """Apply patch operations to the working config. Invalid ops are rejected with the reason."""
    session = current_session()
    try:
        ops = json.loads(ops_json)
    except json.JSONDecodeError as e:
        return f"REJECTED: ops_json is not valid JSON ({e})."
    try:
        session.working_config = config_patch.apply_patch(
            session.working_config, ops, session.catalog, session.policy
        )
    except config_patch.PatchError as e:
        return f"REJECTED: {e}"
    session.applied_ops.extend(ops)
    session.validated = False
    return f"APPLIED: {len(ops)} op(s). The working config now has {len(session.applied_ops)} op(s) in total."


@tool
def validate_working_config() -> str:
    """Validate that the working config deserializes into a runnable Agent. Run this before smoke tests."""
    session = current_session()
    report = _validate_config(session.working_config)
    session.validated = report.ok
    return report.message()


@tool
def run_smoke_test() -> str:
    """Run the working config on a few train tasks and report per-task answers and the score."""
    session = current_session()
    if not session.validated:
        report = _validate_config(session.working_config)
        session.validated = report.ok
        if not report.ok:
            return "Cannot smoke test, the working config is invalid.\n" + report.message()
    try:
        results = run_tasks(
            session.benchmark, session.working_config, session.smoke_tasks, session.settings
        )
    except HarnessError as e:
        return f"SMOKE TEST FAILED: {e}"
    report = session.benchmark.score(
        session.smoke_tasks,
        results,
        step_cost_lambda=session.settings.step_cost_lambda,
    )
    session.smoke_score = report.score
    lines = ["Smoke test results (small train subset, indicative only):"]
    for r in report.individual:
        mark = "PASS" if r["match"] else "FAIL"
        lines.append(f"  [{mark}] {r['task_id']}: expected {r['expected']}, got {r['answer']!r}")
    lines.append(f"smoke score: {report.score:.3f}")
    return "\n".join(lines)


@tool
def read_journal(last_n: Annotated[int, "How many recent iterations to show."] = 5) -> str:
    """Read the most recent optimization journal entries: past patches, scores, and accept/reject decisions."""
    session = current_session()
    records = [r for r in session.store.read_journal() if r.get("type") == "iteration"][-last_n:]
    if not records:
        return "The journal is empty; this is the first iteration."
    lines = []
    for r in records:
        lines.append(
            f"- iteration {r['iteration']}: {'ACCEPTED' if r['accepted'] else 'rejected'} "
            f"score={r.get('score')} (champion was {r.get('champion_score')}) ops={json.dumps(r.get('ops', []))}"
        )
        for task in r.get("failed_tasks", []):
            lines.append(f"    failed: {task}")
    return "\n".join(lines)


@tool
def create_component(
    module_name: Annotated[str, "snake_case module name, e.g. 'calculator'."],
    class_name: Annotated[str, "CamelCase class name, e.g. 'Calculator'."],
    tool_name: Annotated[str, "snake_case tool name the target agent will see, e.g. 'calculate'."],
    description: Annotated[str, "One-sentence tool description shown to the target agent."],
    source_code: Annotated[
        str,
        "Complete Python source of the module. It must define the class decorated with "
        "@component (from haystack import component), with a run() method that has typed "
        "parameters and returns a dict declared via @component.output_types(...). "
        "Pure Python only: no imports of os/sys/subprocess, no file or network access.",
    ],
) -> str:
    """Create a new tool for the target agent by writing a small custom Haystack component.

    Use this only when benchmark tasks need a capability no catalog tool provides.
    On success the tool is registered in the catalog; enable it with propose_patch.
    """
    session = current_session()
    try:
        extension, _ = _create_component(
            module_name=module_name,
            class_name=class_name,
            tool_name=tool_name,
            description=description,
            source=source_code,
            existing_tool_names=set(session.catalog),
        )
    except ComponentAuthoringError as e:
        return f"REJECTED: {e}"
    session.store.add_extension(extension)
    session.store.save_component_source(module_name, source_code)
    from security_agent.optimize.tool_catalog import get_catalog

    session.catalog = get_catalog(session.seed_config, session.store.extensions())
    return (
        f"CREATED: tool '{tool_name}' backed by security_agent.optimize.generated.{module_name}.{class_name}. "
        f'It is registered but not yet enabled; enable it with {{"op": "enable_tool", "name": "{tool_name}"}}.'
    )


@tool
def submit_candidate(
    summary: Annotated[str, "One or two sentences: what you changed and why you expect it to score higher."],
) -> str:
    """Submit the working config as this iteration's candidate. Ends the iteration.

    The harness then scores the candidate on the full train split and accepts it only
    if it beats the champion.
    """
    session = current_session()
    if not session.applied_ops:
        return "REJECTED: no patches applied; submit only after changing the config."
    report = _validate_config(session.working_config)
    session.validated = report.ok
    if not report.ok:
        return "REJECTED: the working config is invalid.\n" + report.message()
    session.submitted = True
    session.submitted_summary = summary
    return "SUBMITTED. The harness will benchmark this candidate."


META_TOOLS = [
    read_config,
    list_mutable_paths,
    list_tool_catalog,
    propose_patch,
    validate_working_config,
    run_smoke_test,
    read_journal,
    create_component,
    submit_candidate,
]
