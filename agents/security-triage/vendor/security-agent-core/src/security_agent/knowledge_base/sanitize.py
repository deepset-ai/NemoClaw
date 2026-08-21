"""Scrub untrusted knowledge-base text before it reaches the agent's context.

Ported from redamon's `agentic/tools.py` (`_sanitize_kb_content`, `_safe_surface`).

This matters more here than it does upstream: the SEC-bench agent holds `run_shell` and
`edit_file` on a live container, and knowledge-base text lands in the same context. Every chunk
is third-party content (NVD descriptions, ExploitDB titles, the CWE reference), so a poisoned
upstream entry is an injection vector.

Defence is layered; this module is one layer:

1. Immutable upstream commit pins (`knowledge_base.pins`) — upstream cannot mutate under us.
2. This module — neutralize role/boundary markers, cap length.
3. Privilege separation: the agent pack's five tools all run inside the knowledge-base
   sub-agent, which holds no container tools. Feed text now lands in *its* context first, and
   only a written summary crosses into the context that holds `run_shell`/`edit_file`.
4. Explicit untrusted-content framing at that boundary (`knowledge_base.render.render_answer`).
5. A system-prompt sentence in the sub-agent's prompt (`knowledge_base.prompts`) and in
   seeds/secbench.yaml, telling both agents that knowledge-base text is data.

Scrubbing happens at **index** time (`mapping.chunk_to_document` runs `sanitize_document` on
every chunk), so what Qdrant holds is already safe to hand to a model. Not at render time,
because the sub-agent reaches documents through five different pack tools — `search_documents`,
`fetch_documents_by_filter` and the three metadata tools — and a scrubber attached to one
renderer would leave the other four uncovered. It used to happen when the local artifact was
loaded, which covered all five for the same reason; Qdrant has no such hydration step, so the
funnel moved to the write path. The cost: changing the policy now needs `secagent kb rebuild`.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from haystack.dataclasses import Document

# Per-chunk content cap. Stops one poisoned (or merely enormous) entry dominating the context.
# Chunks are built under a 480-token budget (~1900 chars), so this bites only on an outlier.
KB_CONTENT_MAX_CHARS = 2000
# Per-metadata-value cap. Metadata reaches the LLM too — the pack's retrieval tool renders
# `json.dumps(doc.meta)` next to every result — so a title or a tag is as good an injection
# carrier as the body, just a shorter one.
KB_META_MAX_CHARS = 500
# Max items rendered for a list-valued field before a "+N more" suffix.
KB_LIST_MAX_ITEMS = 10

REPLACEMENT = "[role-marker stripped]"

# The framing markers the renderers emit — kept here so neither a chunk nor an answer the
# knowledge-base agent wrote about one can forge a closing boundary and make injected text look
# like it sits outside the untrusted region.
# `tests/test_kb_sanitize.py` derives the required patterns from the render module's literals, so
# the two cannot drift.
FRAME_BEGIN = "[BEGIN UNTRUSTED KNOWLEDGE BASE RESULTS]"
FRAME_END = "[END UNTRUSTED KNOWLEDGE BASE RESULTS]"
CHUNK_OPEN = "<kb_chunk>"
CHUNK_CLOSE = "</kb_chunk>"

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
    re.compile(r"<\s*/?\s*user\s*>", re.IGNORECASE),
    re.compile(r"<\s*/?\s*assistant\s*>", re.IGNORECASE),
    re.compile(r"<\s*/?\s*kb_chunk\s*>", re.IGNORECASE),
    re.compile(r"<\s*/?\s*kb_content\s*>", re.IGNORECASE),
    re.compile(r"\[\s*INST\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*/\s*INST\s*\]", re.IGNORECASE),
    re.compile(r"<\|\s*im_start\s*\|>", re.IGNORECASE),
    re.compile(r"<\|\s*im_end\s*\|>", re.IGNORECASE),
    # `\s+` rather than literal spaces, plus IGNORECASE, so case/whitespace variants cannot
    # dodge the strip.
    re.compile(r"\[\s*BEGIN\s+UNTRUSTED\s+KNOWLEDGE\s+BASE\s+RESULTS\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*END\s+UNTRUSTED\s+KNOWLEDGE\s+BASE\s+RESULTS\s*\]", re.IGNORECASE),
)


def sanitize(content: str, max_chars: int = KB_CONTENT_MAX_CHARS) -> str:
    """Neutralize role/boundary markers in untrusted text and cap its length."""
    if not content:
        return ""
    for pattern in _PROMPT_INJECTION_PATTERNS:
        content = pattern.sub(REPLACEMENT, content)
    if len(content) > max_chars:
        content = content[:max_chars] + "... [truncated]"
    return content


def _sanitize_meta_value(value: Any) -> Any:
    """Scrub a metadata value, preserving its type so filters still work.

    Numbers, booleans and dates stay as they are — `meta.cvss_score >= 7` has to keep comparing
    numerically after this runs, and there is no injection surface in a float anyway.
    """
    if isinstance(value, str):
        return sanitize(value, max_chars=KB_META_MAX_CHARS)
    if isinstance(value, list):
        return [_sanitize_meta_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_meta_value(item) for key, item in value.items()}
    return value


def sanitize_document(document: Document) -> Document:
    """Scrub a knowledge-base document's content and metadata.

    Applied by `mapping.chunk_to_document` to every chunk on its way into Qdrant, so every path
    the knowledge-base agent has into the corpus — relevance search, filter fetch, and the three
    metadata inspection tools — reads scrubbed text. The document `id` is carried through
    unchanged: it is the citation handle the agents use, and the key an incremental build
    upserts by.
    """
    return replace(
        document,
        content=sanitize(document.content or "") or None,
        meta={key: _sanitize_meta_value(value) for key, value in (document.meta or {}).items()},
    )


def safe_surface(value: Any, default: str = "") -> str:
    """Coerce any KB field to a display string, sanitizing every string that passes through.

    SECURITY INVARIANT (inherited from upstream): every KB value that reaches the LLM goes
    through this helper. Interpolating `doc.meta["x"]` straight into an f-string bypasses
    sanitization and reopens the injection surface — if you add a field to the renderer, wrap it.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        items = [v for v in value if v is not None and v != ""]
        if not items:
            return default
        rendered = ", ".join(sanitize(str(v)) for v in items[:KB_LIST_MAX_ITEMS])
        if len(items) > KB_LIST_MAX_ITEMS:
            rendered += f" (+{len(items) - KB_LIST_MAX_ITEMS} more)"
        return rendered
    return sanitize(str(value))
