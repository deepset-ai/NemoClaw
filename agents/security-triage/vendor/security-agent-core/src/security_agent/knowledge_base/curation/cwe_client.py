"""CWE reference client — a local, network-free knowledge-base source.

Not part of redamon: this project already ships `data/cwe_reference.json` (969 CWE entries),
which `security_agent.components.CweSearch` searches with BM25. Feeding it into the knowledge
base too buys two things:

1. A fully offline `dev` profile. Every other source needs the network, so without this there
   would be no way to build a real store in CI or in the opt-in integration test.
2. It makes `search_security_kb` a strict superset of the existing `search_cwe` tool, so the
   optimizer can meaningfully choose between them.

Follows the same BaseClient contract as the vendored clients so `build.py` treats it identically.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from security_agent import paths
from security_agent.knowledge_base.chunking import ChunkStrategy
from security_agent.knowledge_base.curation.base_client import BaseClient

logger = logging.getLogger(__name__)

# Project-relative, like the vendored clients' source_path values.
CWE_SOURCE_PATH = "data/cwe_reference.json"


class CweClient(BaseClient):
    """One chunk per CWE entry, read from the committed reference file."""

    SOURCE = "cwe"
    NODE_LABEL = "CweChunk"

    def __init__(self, cache_dir: Optional[str] = None, reference_path: Optional[str] = None):
        # `cache_dir` is unused (nothing is downloaded) but kept in the signature so every
        # client in the registry is constructed the same way.
        self.cache_dir = Path(cache_dir) if cache_dir else paths.KB_CACHE_DIR / "cwe"
        self.reference_path = Path(reference_path) if reference_path else paths.CWE_REFERENCE

    def fetch(self, **kwargs) -> list[dict]:
        """Read the CWE reference; returns one raw entry per CWE id."""
        try:
            reference = json.loads(Path(self.reference_path).read_text())
        except FileNotFoundError:
            logger.error("CWE reference not found at %s", self.reference_path)
            return []
        except json.JSONDecodeError as exc:
            logger.error("CWE reference at %s is not valid JSON: %s", self.reference_path, exc)
            return []

        entries = [
            {
                "cwe_id": cwe_id,
                "name": (entry.get("name") or "").strip(),
                "description": (entry.get("description") or "").strip(),
                "source_path": CWE_SOURCE_PATH,
            }
            for cwe_id, entry in reference.items()
            if isinstance(entry, dict)
        ]
        logger.info("Read %d CWE entries from %s", len(entries), self.reference_path)
        return entries

    def to_chunks(self, raw_data: list[dict]) -> list[dict]:
        """One chunk per CWE. Mirrors CweSearch's indexed text (`name. description`)."""
        chunks = []
        for entry in raw_data:
            cwe_id = entry["cwe_id"]
            name = entry.get("name", "")
            description = entry.get("description", "")
            body = ". ".join(part for part in (name, description) if part)
            chunks.append({
                "chunk_id": ChunkStrategy.generate_chunk_id(self.SOURCE, cwe_id),
                "content": f"{cwe_id}: {body}" if body else cwe_id,
                "title": f"{cwe_id}: {name}" if name else cwe_id,
                "source": self.SOURCE,
                "cwe_id": cwe_id,
                "source_path": entry.get("source_path", CWE_SOURCE_PATH),
            })
        logger.info("Created %d CWE chunks", len(chunks))
        return chunks
