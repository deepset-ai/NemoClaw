"""Hooks for the meta-agent, built on Haystack's Agent hook points.

- `IterationBudgetHook` (on_exit): keeps the meta-agent working until it has
  submitted a candidate, by setting the `continue_run` control flag. The loop stays
  bounded by the Agent's own `max_agent_steps` (on_exit hooks do not run when the
  step budget stops the run).
- `JournalHook` (before_tool): records every tool call the meta-agent makes into the
  session trace, giving each iteration an audit trail. This Haystack build has no
  post-tool hook point (valid points are before_llm, before_tool, on_exit), so the
  trace records the pending call (tool + arguments) at invocation time; per-tool
  results are still surfaced through the tools' own return values and the journal.
"""

from __future__ import annotations

from typing import Any

from haystack.components.agents.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from security_agent.optimize.meta.session import current_session

NUDGE = (
    "You have not submitted a candidate yet. Continue: apply at least one patch, "
    "validate the working config, run the smoke test, then call submit_candidate. "
    "If your last action was rejected, read the rejection reason and fix the op."
)


class IterationBudgetHook:
    allowed_hook_points = ("on_exit",)

    def run(self, state: State) -> None:
        if current_session().submitted:
            return
        state.set("messages", [ChatMessage.from_system(NUDGE)])
        state.set("continue_run", True)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IterationBudgetHook":
        return default_from_dict(cls, data)


class JournalHook:
    allowed_hook_points = ("before_tool",)

    def run(self, state: State) -> None:
        session = current_session()
        messages = state.data.get("messages", [])
        if not messages:
            return
        # `before_tool` fires after the model requests tool calls, before they run.
        # The pending calls live on the last (assistant) message.
        for tool_call in getattr(messages[-1], "tool_calls", None) or []:
            session.trace.append(
                {
                    "step": state.get("step_count"),
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JournalHook":
        return default_from_dict(cls, data)
