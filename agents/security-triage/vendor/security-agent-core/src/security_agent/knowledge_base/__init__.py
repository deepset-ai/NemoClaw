"""Security knowledge base: curated security datasets behind a Haystack RAG tool.

Build side (`secagent kb build`): vendored redamon curation clients fetch and chunk NVD and
ExploitDB (plus the local CWE reference); a Haystack indexing pipeline computes a dense and a
sparse (BM25) vector for each and writes them into a Qdrant collection
(`docker compose up -d qdrant`).

Query side: `search.SecurityKbSearch` — the agent pack's Advanced RAG Agent
(https://docs.haystack.deepset.ai/docs/advanced-rag-agent) over that store, with hybrid BM25 +
dense retrieval, reciprocal rank fusion and a cross-encoder rerank underneath its
`search_documents` tool. Wrapped as a `ComponentTool` named `search_security_kb` in
`seeds/secbench.yaml`.

See the repo-root NOTICE for the vendoring provenance and licence.

Submodules are imported lazily: `search` pulls sentence-transformers (and torch), which a test
that only needs `mapping` should not pay for.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "DEFAULT_PROFILES",
    "KbSettings",
    "SecurityKbSearch",
    "build_kb",
    "chunk_to_document",
    "documents_from_chunks",
    "make_store",
    "load_kb_settings",
]

_LAZY: dict[str, str] = {
    "DEFAULT_PROFILES": "settings",
    "KbSettings": "settings",
    "load_kb_settings": "settings",
    "chunk_to_document": "mapping",
    "documents_from_chunks": "mapping",
    "make_store": "store",
    "build_kb": "build",
    "SecurityKbSearch": "search",
}

if TYPE_CHECKING:  # pragma: no cover
    from security_agent.knowledge_base.build import build_kb
    from security_agent.knowledge_base.mapping import chunk_to_document, documents_from_chunks
    from security_agent.knowledge_base.search import SecurityKbSearch
    from security_agent.knowledge_base.settings import DEFAULT_PROFILES, KbSettings, load_kb_settings
    from security_agent.knowledge_base.store import make_store


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(f"{__name__}.{module_name}", fromlist=[name])
    return getattr(module, name)
