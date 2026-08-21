"""Build the meta-agent: a Haystack Agent whose job is improving the target config."""

from __future__ import annotations

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator

from security_agent.optimize.meta.hooks import IterationBudgetHook, JournalHook
from security_agent.optimize.meta.tools import META_TOOLS
from security_agent.optimize.settings import Settings

META_SYSTEM_PROMPT = """\
You are AutoAgent, a meta-agent that improves another AI agent by editing its
serialized configuration. You never edit YAML text directly; you use tools.

The task the target agent must solve, how it is scored, and what "better" means are
defined in the directive below (the user message). Read it carefully — it is the
objective you optimize. read_config and list_tool_catalog show the concrete surface you
may change for this target.

Work in this order, every iteration:
1. read_journal and read_config to understand where you are and what failed before.
2. Decide ONE focused improvement (1-3 patch ops). list_mutable_paths and
   list_tool_catalog show what you can change.
3. propose_patch, then validate_working_config, then run_smoke_test.
4. If the smoke test looks better (or you have a clear reason), submit_candidate.
   If a patch is rejected or validation fails, read the message and fix it.

Rules:
- Small steps beat big rewrites; you get many iterations.
- The system_prompt is usually the highest-leverage surface: how it tells the target to
  approach the task, which tools to use and when, and what a correct final answer is.
- set_tool_description shapes when the target reaches for each tool. Keep any output or
  answer contract the objective depends on intact.
- Always finish by calling submit_candidate.
"""


def build_meta_agent(settings: Settings) -> Agent:
    hooks: dict = {
        "on_exit": [IterationBudgetHook()],
        "before_tool": [JournalHook()],
    }
    if settings.hitl:
        from haystack.human_in_the_loop import (
            AlwaysAskPolicy,
            BlockingConfirmationStrategy,
            ConfirmationHook,
            RichConsoleUI,
        )

        # Append so the JournalHook already registered on before_tool is preserved.
        hooks.setdefault("before_tool", []).append(
            ConfirmationHook(
                confirmation_strategies={
                    "create_component": BlockingConfirmationStrategy(
                        confirmation_policy=AlwaysAskPolicy(), confirmation_ui=RichConsoleUI()
                    )
                }
            )
        )

    # Guardrail: create_component lets the meta-agent write new tool code. It is off by
    # default for the security campaigns (settings.allow_create_component), keeping the
    # optimization surface to prompt + tool descriptions + step budget.
    tools = (
        META_TOOLS
        if settings.allow_create_component
        else [t for t in META_TOOLS if t.name != "create_component"]
    )

    return Agent(
        chat_generator=OpenAIResponsesChatGenerator(model=settings.meta_model),
        tools=tools,
        system_prompt=META_SYSTEM_PROMPT,
        exit_conditions=["submit_candidate"],
        max_agent_steps=settings.meta_max_steps,
        hooks=hooks,
    )
