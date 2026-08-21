"""Build the security knowledge base: fetch -> chunk -> embed (Haystack) -> Qdrant.

Ported from redamon's `curation/data_ingestion.py`, with its FAISS + Neo4j layer replaced by a
Haystack indexing pipeline writing into a Qdrant collection. What was worth keeping from
upstream: the per-source error isolation, the chunk-level content manifest that makes rebuilds
incremental, the file-level download caches inside each client, and the ingest lock.

The manifest survives the move to Qdrant even though upsert-by-id would make a full re-index
correct: correct but not cheap. Skipping unchanged chunks is what keeps a re-build from
re-embedding 55k documents through sentence-transformers.

Why the fetch/chunk stage is NOT a Haystack component: the curation clients interleave
downloading, tar/YAML safety bounds, upstream commit-pin verification and file-level hash
caching, and each one needs to fail independently without taking the build down. Wrapping them
as components would add a serialization surface for code that is never serialized (a KB build is
a CLI action, not a promoted pipeline artifact) and would turn a plain `continue` into awkward
pipeline error handling. The pipeline covers the part that genuinely benefits from it — batched
sentence-transformers embedding into a document store.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from security_agent.knowledge_base import profiles
from security_agent.knowledge_base.atomic_io import atomic_write_json, ingest_lock
from security_agent.knowledge_base.mapping import documents_from_chunks
from security_agent.knowledge_base.settings import KbSettings, load_kb_settings
from security_agent.knowledge_base.store import make_store, source_counts, write_meta

logger = logging.getLogger(__name__)

MANIFEST_FILE = ".manifest.json"
LAST_INGEST_FILE = ".last_ingest"


# ---------------------------------------------------------------------------
# Manifest + markers (ported from data_ingestion.py)
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """SHA256 of chunk content, first 16 hex chars."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_manifest(cache_dir: Path) -> dict[str, str]:
    """`{chunk_id: content_hash}` for everything previously embedded."""
    path = cache_dir / MANIFEST_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.warning("Chunk manifest at %s is unreadable; treating every chunk as new", path)
        return {}


def save_manifest(cache_dir: Path, manifest: dict[str, str]) -> None:
    atomic_write_json(cache_dir / MANIFEST_FILE, manifest)


def filter_unchanged(
    chunks: list[dict], manifest: dict[str, str]
) -> tuple[list[dict], dict[str, str]]:
    """Drop chunks whose content is unchanged, and collapse duplicate ids within the batch.

    Both halves matter. The manifest is what makes a rebuild incremental (only changed chunks
    are re-embedded). The within-batch dedup is not theoretical: upstream sources genuinely
    repeat a chunk_id — the public
    ExploitDB CSV mirror has duplicate edb_id rows (~1%). Last occurrence wins, matching the
    store's `DuplicatePolicy.OVERWRITE`; first-seen order is preserved.
    """
    last_index: dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        last_index[chunk["chunk_id"]] = idx
    if len(last_index) != len(chunks):
        deduped = [chunks[i] for i in sorted(last_index.values())]
        logger.info("Collapsed %d duplicate chunk_id(s) within batch", len(chunks) - len(deduped))
    else:
        deduped = chunks

    new_chunks: list[dict] = []
    updated = dict(manifest)
    for chunk in deduped:
        cid = chunk["chunk_id"]
        digest = content_hash(chunk["content"])
        if manifest.get(cid) == digest:
            continue
        new_chunks.append(chunk)
        updated[cid] = digest
    return new_chunks, updated


def clear_file_hashes(cache_dir: Path) -> None:
    """Remove every per-source `.file_hashes.json` so a rebuild re-parses all downloads."""
    if not cache_dir.exists():
        return
    for path in cache_dir.rglob(".file_hashes.json"):
        path.unlink()
        logger.debug("Cleared file hashes: %s", path)


def read_last_ingest(cache_dir: Path) -> Optional[dict]:
    path = cache_dir / LAST_INGEST_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_last_ingest(cache_dir: Path, profile: str) -> None:
    atomic_write_json(
        cache_dir / LAST_INGEST_FILE,
        {"timestamp": datetime.now(timezone.utc).isoformat(), "profile": profile},
    )


def kb_version(built_at: str, embedding_model: str, feed_pins: dict) -> str:
    """Short, stable identity for a built store.

    Recorded in `kb_meta.json` and worth logging into an optimize campaign's journal: if the KB
    changes mid-campaign, iteration scores stop being comparable and the optimizer would credit
    a KB refresh to a prompt change.
    """
    payload = json.dumps(
        {"built_at": built_at, "model": embedding_model, "pins": feed_pins}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# The indexing pipeline
# ---------------------------------------------------------------------------

def build_indexing_pipeline(
    settings: KbSettings,
    document_store: Any,
    embedder: Any = None,
    sparse_embedder: Any = None,
):
    """`embedder -> sparse_embedder -> writer` over the Qdrant collection.

    Two vectors per document, because Qdrant has no BM25: the dense one from
    sentence-transformers as before, and a sparse BM25 one that replaces the in-process
    `InMemoryBM25Retriever` index. They are computed in one pass so a rebuild embeds each chunk
    once for both legs.

    Both embedders are injectable so the unit tests can pass deterministic fakes and never
    download a model (the same seam `optimize/testing.py` uses for the chat generator).
    """
    from haystack import Pipeline
    from haystack.components.writers import DocumentWriter
    from haystack.document_stores.types import DuplicatePolicy

    if embedder is None:
        from haystack.utils import ComponentDevice
        from haystack_integrations.components.embedders.sentence_transformers import (
            SentenceTransformersDocumentEmbedder,
        )

        from security_agent.knowledge_base.pins import model_revision

        embedder = SentenceTransformersDocumentEmbedder(
            model=settings.embedding_model,
            revision=model_revision(settings.embedding_model),
            device=ComponentDevice.from_str(settings.device),
            prefix=settings.passage_prefix,
            # Normalized vectors make the store's dot_product similarity equal cosine, so the
            # dense leg is a plain matrix product.
            normalize_embeddings=True,
            batch_size=settings.st_batch_size,
            local_files_only=settings.offline,
            progress_bar=True,
        )

    if sparse_embedder is None:
        from haystack_integrations.components.embedders.fastembed import (
            FastembedSparseDocumentEmbedder,
        )

        sparse_embedder = FastembedSparseDocumentEmbedder(
            model=settings.sparse_model,
            batch_size=settings.st_batch_size,
            progress_bar=True,
        )

    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("sparse_embedder", sparse_embedder)
    # OVERWRITE is required, not cosmetic: DuplicatePolicy.NONE degrades to FAIL, and duplicate
    # ids survive across batches even after the within-batch dedup. It is also what makes an
    # incremental build a merge — Qdrant upserts the changed points by id and leaves the rest.
    pipeline.add_component(
        "writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)
    )
    pipeline.connect("embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")
    return pipeline


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_kb(
    settings: Optional[KbSettings] = None,
    *,
    profile: Optional[str] = None,
    source: Optional[str] = None,
    rebuild: bool = False,
    cache_dir: Optional[Path] = None,
    nvd_api_key: Optional[str] = None,
    embedder: Any = None,
    sparse_embedder: Any = None,
    document_store: Any = None,
) -> dict[str, Any]:
    """Fetch, chunk, embed and save. Returns a stats dict (also what the CLI prints).

    One source failing does not abort the build: its error lands in
    `stats["sources"][name]["error"]` and the remaining sources still index. That is the
    difference between "NVD was flaky, the build failed" and "NVD was flaky, you have 62k
    chunks and a loud summary line".
    """
    settings = settings or load_kb_settings()
    profile = profile or settings.profile
    cache_dir = Path(cache_dir) if cache_dir else settings.resolve(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sources = [source] if source else settings.sources(profile)
    started = time.monotonic()
    stats: dict[str, Any] = {
        "profile": profile if not source else f"{profile} (source={source})",
        "store": f"{settings.qdrant_url}/{settings.qdrant_index}",
        "rebuild": rebuild,
        "sources": {},
        "indexed": 0,
        "unchanged": 0,
        "duplicates": 0,
        "errors": 0,
    }

    # One build at a time: two concurrent builds would corrupt the shared download cache and
    # interleave writes to the same artifact.
    with ingest_lock(cache_dir):
        if rebuild:
            clear_file_hashes(cache_dir)
            (cache_dir / MANIFEST_FILE).unlink(missing_ok=True)
            # Deleting the NVD cursor matters: otherwise the next incremental fetch would ask
            # only for CVEs modified since the last run and "restore" a nearly empty NVD set.
            (cache_dir / LAST_INGEST_FILE).unlink(missing_ok=True)
            logger.info("Rebuild: cleared chunk manifest, file hashes and ingest marker")

        manifest = load_manifest(cache_dir)
        last_ingest = read_last_ingest(cache_dir)
        # `recreate=rebuild` IS the old "replace instead of merge": a rebuild drops the
        # collection, a normal build upserts into whatever is already there.
        store = document_store or make_store(settings, recreate=rebuild)
        pipeline = build_indexing_pipeline(
            settings, store, embedder=embedder, sparse_embedder=sparse_embedder
        )
        pipeline.warm_up()

        for name, client in profiles.resolve_clients(sources, cache_root=cache_dir):
            source_stats: dict[str, Any] = {}
            stats["sources"][name] = source_stats
            source_started = time.monotonic()
            try:
                raw = client.fetch(**_fetch_kwargs(name, settings, profile, nvd_api_key, last_ingest, rebuild))
                chunks = client.to_chunks(raw)
                source_stats["chunks"] = len(chunks)

                fresh, manifest = filter_unchanged(chunks, manifest)
                # Two distinct reasons a chunk does not reach the embedder, reported apart
                # because they mean opposite things: `duplicates` is upstream repeating a
                # chunk_id (expected for ExploitDB), `unchanged` is the manifest doing its
                # job on a re-build.
                duplicates = len(chunks) - len({c["chunk_id"] for c in chunks})
                source_stats["duplicates"] = duplicates
                source_stats["unchanged"] = len(chunks) - duplicates - len(fresh)
                stats["duplicates"] += duplicates
                stats["unchanged"] += source_stats["unchanged"]

                documents = documents_from_chunks(fresh)
                for start in range(0, len(documents), settings.batch_size):
                    batch = documents[start:start + settings.batch_size]
                    pipeline.run({"embedder": {"documents": batch}})
                source_stats["indexed"] = len(documents)
                stats["indexed"] += len(documents)
                # Release this source's working set before the next one starts downloading —
                # ExploitDB alone is ~45k chunks, and rebinding would otherwise hold them
                # through the next fetch.
                del raw, chunks, fresh, documents
            except Exception as exc:  # noqa: BLE001 - per-source isolation is the point
                logger.exception("Source %r failed", name)
                source_stats["error"] = f"{type(exc).__name__}: {exc}"
                stats["errors"] += 1
            source_stats["seconds"] = round(time.monotonic() - source_started, 1)

        if stats["indexed"]:
            from security_agent.knowledge_base import pins

            built_at = datetime.now(timezone.utc).isoformat()
            meta = {
                "kb_version": kb_version(built_at, settings.embedding_model, pins.FEED_PINS),
                "built_at": built_at,
                "profile": profile,
                "embedding_model": settings.embedding_model,
                "embedding_revision": pins.MODEL_PINS.get(settings.embedding_model),
                "dim": settings.embedding_dim,
                "similarity": "cosine",
                "normalized": True,
                "sparse_model": settings.sparse_model,
                "query_prefix": settings.query_prefix,
                "passage_prefix": settings.passage_prefix,
                "reranker_model": settings.reranker_model,
                "device": settings.device,
                "feed_pins": {k: v.get("ref") for k, v in pins.FEED_PINS.items()},
                # What THIS build contributed. The per-source inventory `kb stats` reports is
                # counted from the collection instead, because on an incremental build a source
                # can contribute nothing while still holding thousands of documents.
                "last_build_indexed": {
                    name: s.get("indexed", 0) for name, s in stats["sources"].items()
                },
            }
            meta["total"] = store.count_documents()
            meta["counts"] = source_counts(store)
            write_meta(store, meta, settings)
            stats["total"] = meta["total"]
            stats["kb_version"] = meta["kb_version"]
            save_manifest(cache_dir, manifest)
            write_last_ingest(cache_dir, profile)
        else:
            logger.warning(
                "Nothing indexed: every chunk was unchanged, or every source failed. "
                "The existing artifact (if any) is left untouched."
            )

    stats["seconds"] = round(time.monotonic() - started, 1)
    return stats


def _fetch_kwargs(
    name: str,
    settings: KbSettings,
    profile: str,
    nvd_api_key: Optional[str],
    last_ingest: Optional[dict],
    rebuild: bool,
) -> dict[str, Any]:
    """Per-source fetch arguments. Only NVD takes any."""
    if name != "nvd":
        return {}
    import os

    kwargs: dict[str, Any] = {
        # The vendored client treats "standard" as "apply the CVSS floor" and "full" as "every
        # severity". Our full profile deliberately keeps the floor — dropping it is what makes
        # the corpus explode past the memory budget.
        "profile": "standard",
        "nvd_api_key": nvd_api_key or os.getenv("NVD_API_KEY"),
        "nvd_days": settings.nvd_lookback_days,
        "nvd_min_cvss": settings.nvd_min_cvss,
    }
    if not rebuild and last_ingest and last_ingest.get("timestamp"):
        kwargs["since"] = last_ingest["timestamp"]
    return kwargs


def format_stats(stats: dict[str, Any]) -> str:
    """Human-readable build summary for the CLI."""
    lines = [
        f"profile={stats['profile']}  store={stats['store']}"
        + (f"  version={stats['kb_version']}" if stats.get("kb_version") else ""),
    ]
    for name, source_stats in stats["sources"].items():
        if "error" in source_stats:
            lines.append(f"  {name:<10} ERROR  {source_stats['error']}")
        else:
            lines.append(
                f"  {name:<10} indexed={source_stats.get('indexed', 0):<7} "
                f"unchanged={source_stats.get('unchanged', 0):<7} "
                f"duplicates={source_stats.get('duplicates', 0):<6} "
                f"{source_stats.get('seconds', 0)}s"
            )
    lines.append(
        f"  total in store: {stats.get('total', 'unchanged')}  "
        f"indexed={stats['indexed']}  unchanged={stats['unchanged']}  "
        f"duplicates={stats['duplicates']}  "
        f"errors={stats['errors']}  {stats.get('seconds', 0)}s"
    )
    return "\n".join(lines)
