"""Offline test doubles.

`ScriptedChatGenerator` is a fully serializable chat generator that answers from a
question->answer mapping. It lets the whole harness (subprocess execution, scoring,
store, optimizer plumbing) run end-to-end without an API key or network access.
"""

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.tools import Toolset
from haystack.tools.tool import Tool


@component
class ScriptedChatGenerator:
    """Returns a canned reply for each user question; used for offline harness tests."""

    def __init__(self, replies: dict[str, str] | None = None, default_reply: str = "I do not know."):
        self.replies = replies or {}
        self.default_reply = default_reply

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        tools: list[Tool | Toolset] | Toolset | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        question = next((m.text for m in reversed(messages) if m.is_from("user") and m.text), "")
        return {"replies": [ChatMessage.from_assistant(self.replies.get(question, self.default_reply))]}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, replies=self.replies, default_reply=self.default_reply)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScriptedChatGenerator":
        return default_from_dict(cls, data)
