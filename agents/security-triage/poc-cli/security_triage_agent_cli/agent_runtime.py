# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security-triage Agent construction and one-shot execution.

Wires deepset Haystack's `SkillToolset` (read-only skill discovery: `load_skill`,
`read_skill_file`) together with `make_skill_script_tool` -- the constrained
script-execution tool `security_agent.skill_script_tool` provides, since
`SkillToolset` deliberately omits execution (see security-agent-core's PR
"constrained execution tool for Haystack SkillToolset skills"). The Agent loads
the bundled `code-triage` skill, then runs its `sweep.py`/`deepen.py` scripts
through `run_skill_script`.

    from haystack.components.agents import Agent
    from haystack.dataclasses import ChatMessage
    agent = Agent(chat_generator=..., system_prompt=..., tools=[...], exit_conditions=["text"])
    agent.run(messages=[ChatMessage.from_user(text)])["last_message"].text

The chat generator is Anthropic's `claude-sonnet-5` via `AnthropicChatGenerator`,
which honours `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` through the underlying
`anthropic` SDK's own env-var resolution (there is no `base_url=` constructor
param on the Haystack wrapper itself -- it falls through to the SDK, which reads
`ANTHROPIC_BASE_URL` when no `base_url` kwarg is passed), so a NemoClaw managed
Anthropic-compatible inference endpoint can be injected without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Baked into the image (see Dockerfile): the vendored security-agent-core's
# skills directory, containing the code-triage skill this agent runs.
DEFAULT_SKILLS_DIR = Path(
    os.environ.get("SECURITY_TRIAGE_SKILLS_DIR", "/opt/security-agent-core/skills")
)

SYSTEM_PROMPT = (
    "You are a security triage assistant. When a task matches an available "
    "skill, load it with load_skill first and follow its instructions exactly. "
    "Run any script a skill bundles via the run_skill_script tool."
)


def build_agent(skills_dir: Path | None = None) -> Any:
    from haystack.components.agents import Agent
    from haystack.skill_stores.file_system import FileSystemSkillStore
    from haystack.tools import SkillToolset
    from haystack_integrations.components.generators.anthropic.chat.chat_generator import (
        AnthropicChatGenerator,
    )
    from security_agent.skill_script_tool import make_skill_script_tool

    skills_dir = skills_dir or DEFAULT_SKILLS_DIR
    toolset = SkillToolset(FileSystemSkillStore(skills_dir))

    model = os.environ.get("NEMOCLAW_MODEL") or os.environ.get("SECURITY_TRIAGE_MODEL") or "claude-sonnet-5"

    return Agent(
        chat_generator=AnthropicChatGenerator(
            model=model,
            # claude-sonnet-5 defaults to adaptive thinking; anthropic-haystack's
            # message replay currently orders thinking blocks last in assistant
            # turns, which the API rejects on the next tool iteration.
            generation_kwargs={"adaptive_thinking_effort": "none"},
            # Bound per-call latency: the underlying `anthropic` SDK's own
            # defaults (unset here) can leave a single request retrying for a
            # very long time against transient upstream slowness, which reads
            # as a hang with no error -- a triage run should fail loud and
            # fast instead, so an operator/watchdog sees a real error.
            timeout=120.0,
            max_retries=3,
        ),
        tools=[
            toolset,
            # The skill's own venv (tree-sitter/bandit) is provisioned at image
            # build time -- auto_venv=False so a run never tries a PyPI install.
            make_skill_script_tool(skills_dir, auto_venv=False),
        ],
        system_prompt=SYSTEM_PROMPT,
    )


def run_once(repo_path: str, skills_dir: Path | None = None) -> tuple[str, dict]:
    """Run one triage turn. Returns (final_text, meta).

    `meta["tool_results"]` carries every tool call + its raw result verbatim, in
    call order -- this is what verification should diff against a known-good
    `sweep.py`/`deepen.py` JSON, since the LLM's prose summary is not
    deterministic but the tool JSON is.
    """
    from haystack.dataclasses import ChatMessage

    agent = build_agent(skills_dir)
    agent.warm_up()

    result = agent.run(
        messages=[
            ChatMessage.from_user(
                f"Triage the repo at {repo_path} for vulnerabilities using the code-triage skill."
            )
        ]
    )

    tool_results: list[dict] = []
    tool_names: list[str] = []
    for m in result["messages"]:
        for tc in m.tool_calls or []:
            tool_names.append(tc.tool_name)
        for tr in m.tool_call_results or []:
            tool_results.append({"tool_name": tr.origin.tool_name, "result": tr.result})

    last = result.get("last_message")
    text = (getattr(last, "text", None) or "") if last is not None else ""
    meta = {"tool_calls": tool_names, "tool_results": tool_results, "step_count": result.get("step_count")}
    return text, meta
