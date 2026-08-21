"""Integration tests: Haystack's SkillToolset serves the code-triage skill, and an
Agent given a bring-your-own execution tool can actually run its bundled scripts.

`tests/test_code_triage_skill.py` covers the scripts in isolation. These tests cover
the *Haystack wiring* instead:

- the read-only `SkillToolset` path (discovery, `load_skill`, `read_skill_file`) —
  pure Haystack, no LLM, no venv, always runs;
- `make_skill_script_tool`, the execution surface Haystack deliberately omits
  (the skill store is read-only by design), including its path/extension guardrails;
- the full agentic loop (Agent loads the skill, then runs its scripts through the
  tool) — gated on ANTHROPIC_API_KEY so CI without a key still passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from security_agent.skill_script_tool import make_skill_script_tool

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
SKILL_DIR = SKILLS_DIR / "code-triage"

pytest.importorskip("haystack", reason="haystack-ai not installed")

from haystack.skill_stores.file_system import FileSystemSkillStore  # noqa: E402
from haystack.tools import SkillToolset  # noqa: E402


@pytest.fixture
def toolset() -> SkillToolset:
    ts = SkillToolset(FileSystemSkillStore(SKILLS_DIR))
    ts.warm_up()
    return ts


@pytest.fixture
def vuln_repo(tmp_path: Path) -> Path:
    """A tiny repo with one obvious command-injection sink for sweep to flag."""
    (tmp_path / "svc.py").write_text(
        "import subprocess\n"
        "def run_cmd(cmd):\n"
        "    subprocess.run(cmd)\n"
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# Read-only SkillToolset path (no LLM, no venv)
# --------------------------------------------------------------------------- #

def test_skilltoolset_discovers_skill_and_bakes_catalog(toolset: SkillToolset):
    assert "code-triage" in toolset.skills
    load_skill = next(t for t in toolset if t.name == "load_skill")
    # The catalog is baked into the tool description (no system-prompt injection).
    assert "code-triage" in load_skill.description


def test_load_skill_returns_body_and_bundled_script_manifest(toolset: SkillToolset):
    load_skill = next(t for t in toolset if t.name == "load_skill")
    body = load_skill.invoke(name="code-triage")
    # Instructions come back, and the bundled scripts are auto-listed in the manifest —
    # i.e. the scripts ARE discoverable; only their execution is left to the caller.
    assert "sweep.py" in body and "deepen.py" in body


def test_read_skill_file_serves_bundled_script_source(toolset: SkillToolset):
    read_file = next(t for t in toolset if t.name == "read_skill_file")
    src = read_file.invoke(name="code-triage", path="scripts/sweep.py")
    assert isinstance(src, str) and src.lstrip().startswith("#!")


# --------------------------------------------------------------------------- #
# The bring-your-own execution tool (Haystack omits this by design)
# --------------------------------------------------------------------------- #

def test_skill_script_tool_runs_bundled_script(vuln_repo: Path):
    # sweep.py is stdlib-only, so run it under the current interpreter (no venv build).
    tool = make_skill_script_tool(SKILLS_DIR, auto_venv=False)
    out = tool.invoke(skill="code-triage", script="scripts/sweep.py", args=str(vuln_repo))
    result = json.loads(out)
    assert any(f["path"] == "svc.py" for f in result["files"])


def test_skill_script_tool_rejects_paths_outside_the_skill(tmp_path: Path):
    tool = make_skill_script_tool(SKILLS_DIR, auto_venv=False)
    # escape attempt via traversal
    assert tool.invoke(
        skill="code-triage", script="../../pyproject.toml", args=""
    ).startswith("REJECTED")
    # unknown skill
    assert tool.invoke(skill="no-such-skill", script="scripts/sweep.py", args="").startswith("REJECTED")
    # non-Python file inside the skill
    assert tool.invoke(skill="code-triage", script="SKILL.md", args="").startswith("REJECTED")


# --------------------------------------------------------------------------- #
# Full agentic loop (needs a live model)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_agent_loads_skill_and_runs_its_scripts(toolset: SkillToolset, vuln_repo: Path):
    pytest.importorskip("haystack_integrations.components.generators.anthropic")
    from haystack.components.agents import Agent
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.anthropic.chat.chat_generator import (
        AnthropicChatGenerator,
    )

    agent = Agent(
        # claude-sonnet-5 defaults to adaptive thinking; the anthropic-haystack replay
        # currently orders thinking blocks last in assistant turns, which the API rejects
        # ("final block ... cannot be `thinking`") on the next tool iteration. Disable it.
        chat_generator=AnthropicChatGenerator(
            model="claude-sonnet-5",
            generation_kwargs={"adaptive_thinking_effort": "none"},
        ),
        tools=[toolset, make_skill_script_tool(SKILLS_DIR, auto_venv=False)],
        system_prompt=(
            "You are a security triage assistant. When a task matches an available "
            "skill, load it with load_skill first and follow its instructions exactly. "
            "Run any script a skill bundles via the run_skill_script tool."
        ),
    )
    agent.warm_up()
    result = agent.run(messages=[ChatMessage.from_user(
        f"Triage the repo at {vuln_repo} for vulnerabilities using the code-triage skill."
    )])

    tool_names = [tc.tool_name for m in result["messages"] for tc in (m.tool_calls or [])]
    assert "load_skill" in tool_names, tool_names
    assert "run_skill_script" in tool_names, tool_names
