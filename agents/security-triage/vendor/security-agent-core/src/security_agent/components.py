"""Custom Haystack components for the triage agent.

`CweLookup` backs the `lookup_cwe` tool: exact lookup of a CWE id's definition.
`CweSearch` backs the `search_cwe` tool: BM25 search over the CWE reference to find
candidate classes from a free-text description of a weakness.

Both read the bundled CWE reference (`data/cwe_reference.json`) and are fully
serializable (they store only the reference file path + config), so the whole agent —
tools included — round-trips through its pipeline YAML, needs no network, and stays
deterministic for replay.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from security_agent import paths


def _resolve_reference(reference_path: str) -> Path:
    """Resolve `reference_path` against the project root when it is relative.

    The committed seed/pipeline YAMLs store a project-relative path (e.g.
    `data/cwe_reference.json`) so they stay portable across checkouts; an absolute path
    (e.g. a caller-supplied one) is used as-is.
    """
    p = Path(reference_path)
    return p if p.is_absolute() else paths.PROJECT_ROOT / p


def normalize_cwe_id(raw: str) -> Optional[str]:
    """Coerce loose model input ('787', 'cwe-787', 'CWE 787') to 'CWE-787'.

    Returns None if no digits are present.
    """
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return f"CWE-{digits}" if digits else None


@component
class CweLookup:
    def __init__(self, reference_path: str):
        self.reference_path = reference_path
        self._reference: Optional[dict] = None

    def warm_up(self) -> None:
        if self._reference is None:
            self._reference = json.loads(_resolve_reference(self.reference_path).read_text())

    @component.output_types(definition=str)
    def run(self, cwe_id: str) -> dict:
        self.warm_up()

        key = normalize_cwe_id(cwe_id)
        entry = self._reference.get(key) if key else None
        if not entry:
            known = ", ".join(sorted(self._reference)[:12])
            return {
                "definition": (
                    f"No definition found for '{cwe_id}'. "
                    f"Known ids include: {known}."
                )
            }
        return {
            "definition": f"{key}: {entry['name']} — {entry['description']}"
        }

    def to_dict(self) -> dict:
        return default_to_dict(self, reference_path=self.reference_path)

    @classmethod
    def from_dict(cls, data: dict) -> "CweLookup":
        return default_from_dict(cls, data)


@component
class CweSearch:
    """BM25 search over the CWE reference, backing the `search_cwe` tool.

    Given a free-text description of a weakness (e.g. "unbounded copy into a fixed
    stack buffer"), returns the top-k candidate CWE classes ranked by BM25 over each
    entry's name + description. Lets the agent go from an observed pattern to candidate
    CWE ids without having to recall exact numbers. The BM25 index is built in memory
    at warm-up from the reference file, so the component serializes to just its path
    and `top_k`, and runs offline/deterministically.
    """

    def __init__(self, reference_path: str, top_k: int = 5):
        self.reference_path = reference_path
        self.top_k = top_k
        self._store: Optional[InMemoryDocumentStore] = None

    def warm_up(self) -> None:
        if self._store is None:
            reference = json.loads(_resolve_reference(self.reference_path).read_text())
            store = InMemoryDocumentStore()
            store.write_documents([
                Document(
                    content=f"{entry['name']}. {entry['description']}",
                    meta={"cwe_id": cwe_id, "name": entry["name"],
                          "description": entry["description"]},
                )
                for cwe_id, entry in reference.items()
            ])
            self._store = store

    @component.output_types(matches=str)
    def run(self, query: str) -> dict:
        self.warm_up()
        assert self._store is not None
        hits = self._store.bm25_retrieval(query=query, top_k=self.top_k)
        if not hits:
            return {"matches": f"No CWE candidates found for query: {query!r}."}
        lines = [
            f"{d.meta['cwe_id']}: {d.meta['name']} — {d.meta['description']}"
            for d in hits
        ]
        return {"matches": "\n".join(lines)}

    def to_dict(self) -> dict:
        return default_to_dict(self, reference_path=self.reference_path, top_k=self.top_k)

    @classmethod
    def from_dict(cls, data: dict) -> "CweSearch":
        return default_from_dict(cls, data)
