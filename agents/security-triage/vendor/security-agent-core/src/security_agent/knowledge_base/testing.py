"""Offline test doubles for the knowledge base.

They live in the package rather than in `tests/` for the same reason
`optimize.testing.ScriptedChatGenerator` does: the build side and the query side both need them,
and a double that is importable can be injected into a real Haystack pipeline.

Vectors are derived from a content hash and L2-normalized, so `FakeTextEmbedder` and
`FakeDocumentEmbedder` agree: embedding a document's exact content gives cosine similarity 1.0
with that document. That makes dense-retrieval assertions deterministic without downloading a
model.

`FakeSparse*Embedder` stands in for the fastembed BM25 leg, and `ScriptedKbAgentLlm` is the third double: it drives the agent pack's loop through a fixed
sequence of tool calls, so the whole `search_security_kb` tool — the pack's five tools, the
retrieval pipeline underneath them, and the untrusted framing on the way out — is testable with
no API key. `optimize.testing.ScriptedChatGenerator` cannot do this: it never emits a tool call.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import replace
from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, Document, SparseEmbedding, ToolCall
from haystack.tools import Tool, Toolset

DIM = 8


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    """Deterministic unit vector for `text`."""
    digest = hashlib.sha256((text or "").encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


@component
class FakeDocumentEmbedder:
    """Stand-in for `SentenceTransformersDocumentEmbedder`."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls = 0

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        self.calls += 1
        # `replace` rather than mutating in place, matching what Haystack asks of components.
        return {
            "documents": [
                replace(doc, embedding=fake_vector(doc.content or "", self.dim))
                for doc in documents
            ]
        }

    def to_dict(self) -> dict:
        return default_to_dict(self, dim=self.dim)

    @classmethod
    def from_dict(cls, data: dict) -> "FakeDocumentEmbedder":
        return default_from_dict(cls, data)


@component
class FakeTextEmbedder:
    """Stand-in for `SentenceTransformersTextEmbedder`, matching `FakeDocumentEmbedder`."""

    def __init__(self, dim: int = DIM, prefix: str = ""):
        self.dim = dim
        self.prefix = prefix

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict:
        # The prefix is deliberately NOT hashed: a query for a document's exact content must
        # score 1.0 against it, which is what the retrieval assertions rely on.
        return {"embedding": fake_vector(text, self.dim)}

    def to_dict(self) -> dict:
        return default_to_dict(self, dim=self.dim, prefix=self.prefix)

    @classmethod
    def from_dict(cls, data: dict) -> "FakeTextEmbedder":
        return default_from_dict(cls, data)


_WORD = re.compile(r"\w+")


@component
class FakeSimilarityRanker:
    """Stand-in for the cross-encoder: scores by query-token overlap, descending."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        score_threshold: Optional[float] = None,
    ) -> dict:
        tokens = set(_WORD.findall((query or "").lower()))
        scored = []
        for doc in documents:
            words = set(_WORD.findall((doc.content or "").lower()))
            overlap = len(tokens & words) / (len(tokens) or 1)
            scored.append(
                Document(id=doc.id, content=doc.content, meta=dict(doc.meta), score=overlap)
            )
        scored.sort(key=lambda d: (-(d.score or 0.0), d.id))
        return {"documents": scored[: (top_k or self.top_k)]}

    def to_dict(self) -> dict:
        return default_to_dict(self, top_k=self.top_k)

    @classmethod
    def from_dict(cls, data: dict) -> "FakeSimilarityRanker":
        return default_from_dict(cls, data)


@component
class ScriptedKbAgentLlm:
    """Stand-in for the knowledge-base sub-agent's LLM: scripted tool calls, then a text answer.

    Each entry in `script` is a `(tool_name, arguments)` pair, replayed one per assistant turn;
    once the script is exhausted the answer is returned as plain text, which is the agent's
    `exit_conditions=["text"]`. Counting assistant messages rather than keeping an instance
    counter keeps it correct when the same generator object drives more than one run — the pack
    hands the same instance to the agent loop and to its backup-answer hook.
    """

    def __init__(self, script: Optional[list[tuple[str, dict]]] = None, answer: str = "No answer."):
        self.script = script or []
        self.answer = answer
        self.calls: list[list[ChatMessage]] = []

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[Tool | Toolset] | Toolset] = None,
        **kwargs: Any,
    ) -> dict:
        self.calls.append(list(messages))
        turn = sum(1 for message in messages if message.is_from("assistant"))
        if turn < len(self.script):
            name, arguments = self.script[turn]
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        tool_calls=[ToolCall(tool_name=name, arguments=dict(arguments))]
                    )
                ]
            }
        return {"replies": [ChatMessage.from_assistant(self.answer)]}

    def to_dict(self) -> dict:
        return default_to_dict(self, script=[list(step) for step in self.script], answer=self.answer)

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptedKbAgentLlm":
        script = data.get("init_parameters", {}).get("script") or []
        data["init_parameters"]["script"] = [tuple(step) for step in script]
        return default_from_dict(cls, data)


def fake_sparse_vector(text: str) -> SparseEmbedding:
    """Deterministic sparse vector: one unit-weight dimension per distinct word.

    A stand-in for `Qdrant/bm25` that keeps the property the lexical leg exists for — a query
    sharing a rare token with a document scores against it, and one that shares nothing does not
    — without downloading fastembed's tokenizer or its IDF table. `hashlib` rather than `hash()`
    because the latter is salted per process, and a test corpus must index and query alike.
    """
    indices = sorted(
        int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % 100_000
        for word in set(_WORD.findall((text or "").lower()))
    )
    return SparseEmbedding(indices=indices, values=[1.0] * len(indices))


@component
class FakeSparseDocumentEmbedder:
    """Stand-in for `FastembedSparseDocumentEmbedder`."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        return {
            "documents": [
                replace(doc, sparse_embedding=fake_sparse_vector(doc.content or ""))
                for doc in documents
            ]
        }

    def to_dict(self) -> dict:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FakeSparseDocumentEmbedder":
        return default_from_dict(cls, data)


@component
class FakeSparseTextEmbedder:
    """Stand-in for `FastembedSparseTextEmbedder`, matching `FakeSparseDocumentEmbedder`."""

    @component.output_types(sparse_embedding=SparseEmbedding)
    def run(self, text: str) -> dict:
        return {"sparse_embedding": fake_sparse_vector(text)}

    def to_dict(self) -> dict:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FakeSparseTextEmbedder":
        return default_from_dict(cls, data)
