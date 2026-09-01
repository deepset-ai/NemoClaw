"""The knowledge base's document store: a Qdrant collection, plus its build provenance.

Replaces the hand-rolled on-disk artifact (`documents.jsonl` + `embeddings.npy` + `kb_meta.json`,
merged by id under an atomic-write layer). Qdrant is the persistence layer now, which deletes all
of that: upsert-by-id *is* the incremental merge, the vectors never round-trip through Python at
query time, and metadata filters are evaluated by the engine rather than by a numpy mask we
maintain.

Retrieval is hybrid on two vectors per document rather than a dense vector plus an in-process
BM25 index:

* dense — `BAAI/bge-small-en-v1.5`, as before.
* sparse — fastembed's `Qdrant/bm25`, a real BM25 encoder rather than a learned sparse model.
  SPLADE-class models are semantic, which would duplicate what the dense leg already does; the
  lexical leg is here for exact identifiers (`CVE-2024-21626`) and the terse one-line titles
  ExploitDB rows are made of. The collection is created with `sparse_idf=True`, which is what
  makes Qdrant apply the IDF term BM25 needs — without it the sparse leg degrades to raw term
  frequency.

Deployment is `docker compose up -d qdrant` (see `docker-compose.yml`); tests use the same code
path against `location=":memory:"`, which supports sparse vectors, IDF and every metadata method
the agent pack requires. What is deliberately NOT used is `path=` (on-disk QdrantLocal):
`QdrantLocal` takes an exclusive non-blocking lock on the storage directory, so `secagent kb
build` and a running eval could not hold it at once, and the optimizer scores candidates in a
subprocess while the parent may have the store open.

Build provenance (which model built the vectors, when, from which feed pins) lives in a separate
one-point `<index>__meta` collection rather than a file next to the data. A file could drift from
the collection it describes, which is exactly the failure the model check exists to catch.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from security_agent.knowledge_base.settings import KbSettings, load_kb_settings

logger = logging.getLogger(__name__)

META_COLLECTION_SUFFIX = "__meta"
META_POINT_ID = 0
SETUP_HINT = (
    "Start it with `docker compose up -d qdrant` from security-agent-core/, then build the "
    "corpus with `secagent kb build --profile full` (or `--profile dev` for an offline smoke "
    "test)."
)


class KbStoreError(RuntimeError):
    """Qdrant is unreachable, the collection is missing, or it was built for a different model."""


def make_store(
    settings: Optional[KbSettings] = None,
    *,
    recreate: bool = False,
    embedding_dim: Optional[int] = None,
):
    """Open (or create) the knowledge-base collection.

    :param settings: Knowledge-base settings; defaults to `load_kb_settings()`.
    :param recreate: Drop and recreate the collection — `secagent kb rebuild`.
    :param embedding_dim: Override the configured dimension (tests use a tiny one).
    :raises KbStoreError: If Qdrant cannot be reached.
    """
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    settings = settings or load_kb_settings()
    try:
        store = QdrantDocumentStore(
            **_location_kwargs(settings.qdrant_url),
            index=settings.qdrant_index,
            embedding_dim=embedding_dim or settings.embedding_dim,
            # Vectors are normalized at index time, so cosine and dot product agree; cosine is
            # what Qdrant optimizes for and what the collection reports.
            similarity="cosine",
            use_sparse_embeddings=True,
            # BM25 is an IDF-weighted scheme. Without this the sparse leg silently becomes term
            # frequency only, which looks like it works and ranks badly.
            sparse_idf=True,
            recreate_index=recreate,
            return_embedding=False,
            timeout=settings.qdrant_timeout,
        )
        # Connect now rather than on first query: the store constructor is lazy, so an
        # unreachable Qdrant would otherwise surface several layers up as whatever the first
        # call happened to be doing — reported to the agent as a model-loading failure.
        store._initialize_client()
        return store
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable message, see search.py
        raise KbStoreError(
            f"Cannot reach Qdrant at {settings.qdrant_url} ({type(exc).__name__}: {exc}). "
            f"{SETUP_HINT}"
        ) from exc


def _location_kwargs(url: str) -> dict[str, Any]:
    """`:memory:` goes to `location=`, a real deployment to `url=`."""
    return {"location": url} if url == ":memory:" else {"url": url}


def _client(store: Any, settings: KbSettings):
    """The raw Qdrant client to hold the provenance collection with.

    An in-memory Qdrant lives inside its client, so the meta collection has to share the store's
    own client or it would land in a different, empty database. Against a server, either works;
    reusing the store's keeps it to one connection.
    """
    if store is not None:
        # `_client` is created lazily on first use.
        store._initialize_client()
        return store._client
    if settings.qdrant_url == ":memory:":
        return None
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url, timeout=int(settings.qdrant_timeout))


def _meta_collection(settings: KbSettings) -> str:
    return f"{settings.qdrant_index}{META_COLLECTION_SUFFIX}"


def write_meta(store: Any, meta: dict[str, Any], settings: Optional[KbSettings] = None) -> None:
    """Record what this build produced, alongside the collection it describes."""
    from qdrant_client import models

    settings = settings or load_kb_settings()
    client = _client(store, settings)
    collection = _meta_collection(settings)
    if not client.collection_exists(collection):
        # A one-point collection: Qdrant has no collection-level payload, and a 1-dim dummy
        # vector is the cheapest carrier for one.
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
        )
    client.upsert(
        collection_name=collection,
        points=[models.PointStruct(id=META_POINT_ID, vector=[0.0], payload={"kb_meta": json.dumps(meta)})],
    )


def read_meta(store: Any = None, settings: Optional[KbSettings] = None) -> dict[str, Any]:
    """Read the build provenance. Raises when the knowledge base has never been built."""
    settings = settings or load_kb_settings()
    collection = _meta_collection(settings)
    try:
        client = _client(store, settings)
        if client is None:
            raise KbStoreError(f"No knowledge base at {settings.qdrant_url}. {SETUP_HINT}")
        if not client.collection_exists(collection):
            raise KbStoreError(
                f"No knowledge base in Qdrant at {settings.qdrant_url} "
                f"(collection {collection!r} does not exist). {SETUP_HINT}"
            )
        points = client.retrieve(collection_name=collection, ids=[META_POINT_ID], with_payload=True)
    except KbStoreError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise KbStoreError(
            f"Cannot read the knowledge base provenance from {settings.qdrant_url} "
            f"({type(exc).__name__}: {exc}). {SETUP_HINT}"
        ) from exc
    if not points:
        raise KbStoreError(f"The knowledge base at {settings.qdrant_url} has no build record. {SETUP_HINT}")
    return json.loads(points[0].payload["kb_meta"])


def check_model(meta: dict[str, Any], embedding_model: Optional[str]) -> None:
    """Refuse a collection built with a different embedding model.

    Vectors from two models are not comparable, so a silent mismatch would degrade retrieval with
    no error anywhere — the one failure mode that looks exactly like "the corpus is unhelpful".
    """
    built = meta.get("embedding_model")
    if embedding_model and built and embedding_model != built:
        raise KbStoreError(
            f"The knowledge base was built with embedding model {built!r} (dim {meta.get('dim')}), "
            f"but this component is configured for {embedding_model!r}. Vectors from different "
            f"models are not comparable — either fix the configured model or run "
            f"`secagent kb rebuild`."
        )


def source_counts(store: Any) -> dict[str, int]:
    """Per-source document counts, for `secagent kb stats`.

    Read from the collection rather than from the build's own tally: on an incremental build a
    source can contribute nothing (its upstream files were unchanged) while still holding
    thousands of documents from an earlier build, and reporting the contribution as the inventory
    would read as "exploitdb: 0".

    The source list comes from the data too, not from the configured profile, so a document left
    behind by a profile that is no longer configured still shows up.
    """
    values, _ = store.get_metadata_field_unique_values("source", size=100)
    return {
        source: store.count_documents_by_filter(
            filters={"field": "meta.source", "operator": "==", "value": source}
        )
        for source in sorted(values)
    }
