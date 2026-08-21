"""Chunk dict -> Haystack `Document`. The whole contract surface between the vendored
curation clients and Haystack.

Three decisions worth knowing about:

1. `Document.id` is the chunk's own deterministic `chunk_id`, not a content hash Haystack
   generates. Builds stay idempotent, `DuplicatePolicy.OVERWRITE` does the right thing, and the
   incremental-build manifest keys are document ids.

2. The title is prepended to the content when the content does not already start with it. This
   is load-bearing, not cosmetic: an NVD chunk's content is *description + CVSS + Affected*, so
   the CVE id lives **only** in `title` — without the prepend, a BM25 search for
   "CVE-2024-21626" misses the record entirely. ExploitDB content already begins with its title,
   hence the `startswith` guard. Doing this in `content` rather than via the embedder's
   `meta_fields_to_embed` is deliberate: that option only affects the dense vector, leaving BM25
   blind to the title. One content string keeps both retrieval legs looking at identical text.

3. Content and metadata are scrubbed here, at index time (`sanitize.sanitize_document`), so what
   Qdrant holds is already safe to hand to a model. This used to happen when the store was
   loaded, which covered every read path because every read went through one hydration step.
   Qdrant has no such step: the agent pack reaches documents through `search_documents` (our
   pipeline), `fetch_documents_by_filter` (straight to the store) and three metadata tools that
   read payload values. Scrubbing on the way in is the only funnel all five pass through.
   The cost, accepted deliberately: changing the policy now needs `secagent kb rebuild`, and the
   stored text is no longer byte-faithful to upstream — the raw feed files in `data/kb_cache/`
   still are, which is what incident review actually needs.

Meta is an explicit per-source whitelist of flat JSON scalars (plus `list[str]` for tags).
Keys whose value is None are omitted rather than stored as None: a Haystack `>=` filter on a
missing key evaluates False without raising, so omitting is both safe and smaller.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from haystack.dataclasses import Document

from security_agent.knowledge_base.sanitize import sanitize_document

logger = logging.getLogger(__name__)

# Sources that carry a CVSS score. A severity floor only applies to these — see
# the sub-agent's system prompt, which teaches it the OR form.
CVSS_SOURCES: tuple[str, ...] = ("nvd",)

# Meta keys kept for every source.
_COMMON_KEYS: tuple[str, ...] = ("title", "source", "source_path")

# Per-source meta keys, on top of _COMMON_KEYS. Anything a client emits that is not listed here
# is dropped: the bulky fields (exploitdb's `codes`, NVD's full configuration tree) are already
# inside `content`, and meta bytes are resident memory at corpus scale.
_SOURCE_KEYS: dict[str, tuple[str, ...]] = {
    "cwe": ("cwe_id",),
    "nvd": ("cve_id", "cvss_score", "severity", "published_date", "affected_products"),
    "exploitdb": ("edb_id", "cve_id", "platform", "published_date"),
}

# Keys coerced to float: a client may hand CVSS over as a string.
_FLOAT_KEYS = frozenset({"cvss_score"})
# Keys normalized to a list of strings, capped so a single entry cannot dominate a rendered
# result (NVD's affected_products is already capped at 20 upstream).
_LIST_KEYS = frozenset({"tags", "affected_products"})
_LIST_MAX = 20


class ChunkMappingError(ValueError):
    """A chunk dict does not satisfy the BaseClient contract."""


def _compose_content(chunk: dict) -> str:
    title = (chunk.get("title") or "").strip()
    body = (chunk.get("content") or "").strip()
    if not body:
        return title
    if not title or body.startswith(title):
        return body
    return f"{title}\n\n{body}"


def _coerce(key: str, value: Any) -> Optional[Any]:
    """Coerce a chunk field to a filterable/renderable meta value, or None to drop it."""
    if value is None or value == "":
        return None
    if key in _FLOAT_KEYS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if key in _LIST_KEYS:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = [str(v) for v in value if v not in (None, "")]
        else:
            return None
        # Deduplicate, order-preserving: NVD's affected_products routinely repeats the same CPE
        # string for several version ranges, which would render as seven identical lines.
        items = list(dict.fromkeys(items))
        return items[:_LIST_MAX] or None
    if isinstance(value, (str, int, float, bool)):
        return value.strip() if isinstance(value, str) else value
    return str(value)


def chunk_to_document(chunk: dict) -> Document:
    """Convert one curation chunk dict into a Haystack `Document`.

    Raises `ChunkMappingError` when the chunk is missing `chunk_id`, `content` or `source` —
    the three fields the rest of the pipeline cannot work without.
    """
    chunk_id = chunk.get("chunk_id")
    source = chunk.get("source")
    if not chunk_id or not source:
        raise ChunkMappingError(
            f"chunk is missing chunk_id and/or source: {sorted(chunk)!r}"
        )
    content = _compose_content(chunk)
    if not content:
        raise ChunkMappingError(f"chunk {chunk_id} ({source}) has no content or title")

    meta: dict[str, Any] = {}
    for key in _COMMON_KEYS + _SOURCE_KEYS.get(source, ()):
        coerced = _coerce(key, chunk.get(key))
        if coerced is not None:
            meta[key] = coerced
    meta["source"] = source  # never dropped, even if a client left it blank elsewhere

    return sanitize_document(Document(id=str(chunk_id), content=content, meta=meta))


def documents_from_chunks(chunks: list[dict]) -> list[Document]:
    """Map a batch of chunks, dropping unmappable ones and collapsing duplicate ids.

    Duplicate `chunk_id`s are real (the public ExploitDB CSV repeats an `edb_id` across rows);
    last occurrence wins, matching the store's `DuplicatePolicy.OVERWRITE`. First-seen order is
    preserved so a build writes points in a stable order.
    """
    by_id: dict[str, Document] = {}
    skipped = 0
    for chunk in chunks:
        try:
            doc = chunk_to_document(chunk)
        except ChunkMappingError as exc:
            skipped += 1
            logger.debug("Skipping unmappable chunk: %s", exc)
            continue
        by_id[doc.id] = doc  # dict preserves first-insertion order, last value wins
    if skipped:
        logger.warning("Skipped %d unmappable chunk(s)", skipped)
    duplicates = len(chunks) - skipped - len(by_id)
    if duplicates:
        logger.info("Collapsed %d duplicate chunk_id(s)", duplicates)
    return list(by_id.values())
