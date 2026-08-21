"""Covers the build driver: batching, incrementality, per-source isolation, locking.

All offline — a `FakeClient` stands in for the curation clients and
`knowledge_base.testing.FakeDocumentEmbedder` for sentence-transformers, so no test here touches
the network or downloads a model.

The behaviours asserted are the ones that would otherwise fail silently:
* a second build must re-embed nothing and must still leave the whole store on disk;
* one source failing must not take the build down;
* `rebuild` must clear the manifest, or the "rebuild" would index nothing at all.
"""
import json

import pytest

from security_agent.knowledge_base import build, profiles
from security_agent.knowledge_base.store import read_meta
from security_agent.knowledge_base.atomic_io import IngestLockError, ingest_lock
from security_agent.knowledge_base.settings import KbSettings
from security_agent.knowledge_base.testing import (
    DIM,
    FakeDocumentEmbedder,
    FakeSparseDocumentEmbedder,
)


class FakeClient:
    """Emits a fixed set of chunks; `chunks` is mutable so a test can simulate upstream drift."""

    SOURCE = "fake"
    NODE_LABEL = "FakeChunk"
    chunks: list[dict] = []
    fail: bool = False

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir

    def fetch(self, **kwargs):
        if type(self).fail:
            raise RuntimeError("upstream is down")
        return list(type(self).chunks)

    def to_chunks(self, raw_data):
        return list(raw_data)


class OtherClient(FakeClient):
    SOURCE = "other"
    NODE_LABEL = "OtherChunk"
    chunks = [{"chunk_id": "o1", "content": "other content", "title": "O1", "source": "cwe"}]
    fail = False


def _chunk(i: int, content: str = None) -> dict:
    return {
        "chunk_id": f"c{i}",
        "content": content or f"chunk number {i}",
        "title": f"C{i}",
        "source": "cwe",  # a whitelisted source, so mapping keeps its meta
    }


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    FakeClient.chunks = [_chunk(i) for i in range(3)]
    FakeClient.fail = False
    OtherClient.fail = False
    monkeypatch.setitem(profiles.CLIENTS, "fake", "x:y")
    monkeypatch.setitem(profiles.CLIENTS, "other", "x:y")
    monkeypatch.setattr(
        profiles, "client_class",
        lambda source: {"fake": FakeClient, "other": OtherClient}[source],
    )
    yield


def _settings(tmp_path, **overrides) -> KbSettings:
    return KbSettings(
        # An ephemeral in-process Qdrant: same code path as the Docker deployment, including
        # sparse vectors and IDF, with nothing to clean up between tests.
        qdrant_url=":memory:",
        cache_dir=str(tmp_path / "cache"),
        profile="fake",
        embedding_model="test/model",
        embedding_dim=DIM,
        profiles={"fake": ["fake"], "both": ["fake", "other"]},
        **overrides,
    )


def _new_store(settings):
    from security_agent.knowledge_base.store import make_store

    return make_store(settings, recreate=True, embedding_dim=DIM)


def _build(tmp_path, *, profile="fake", rebuild=False, embedder=None, store=None, **overrides):
    """Build into `store` (create one if not given) and hand it back with the stats.

    The store is threaded through explicitly because an in-memory Qdrant belongs to its client:
    a second `make_store(":memory:")` is a second, empty database, so a test that builds and
    then reads must pass the same instance.
    """
    settings = _settings(tmp_path, **overrides)
    store = store if store is not None else _new_store(settings)
    stats = build.build_kb(
        settings,
        profile=profile,
        rebuild=rebuild,
        embedder=embedder or FakeDocumentEmbedder(dim=DIM),
        sparse_embedder=FakeSparseDocumentEmbedder(),
        document_store=store,
    )
    return stats, store


def _ids(store):
    return sorted(d.id for d in store.filter_documents())


def test_build_indexes_chunks_and_writes_the_collection(tmp_path):
    stats, store = _build(tmp_path)
    assert stats["indexed"] == 3
    assert stats["total"] == 3
    assert stats["errors"] == 0
    assert stats["sources"]["fake"]["chunks"] == 3

    assert _ids(store) == ["c0", "c1", "c2"]
    meta = read_meta(store, _settings(tmp_path))
    assert meta["embedding_model"] == "test/model"
    assert meta["kb_version"]
    assert meta["sparse_model"]
    # `counts` is the collection inventory, keyed by the documents' own meta.source (the
    # FakeClient emits chunks tagged `cwe`); `last_build_indexed` is this build's contribution.
    assert meta["counts"] == {"cwe": 3}
    assert meta["last_build_indexed"] == {"fake": 3}


def test_every_document_gets_both_vectors(tmp_path):
    """Qdrant has no BM25 index, so the lexical leg only exists if the sparse vector is written.
    A missing one is silent: dense retrieval still returns plausible results."""
    _, store = _build(tmp_path)
    store._initialize_client()
    points, _ = store._client.scroll(
        collection_name=store.index, limit=10, with_vectors=True, with_payload=False
    )
    assert points
    for point in points:
        assert "text-dense" in point.vector and "text-sparse" in point.vector


def test_batching_splits_the_work_but_indexes_everything(tmp_path):
    FakeClient.chunks = [_chunk(i) for i in range(7)]
    embedder = FakeDocumentEmbedder(dim=DIM)
    stats, store = _build(tmp_path, embedder=embedder, batch_size=2)
    assert embedder.calls == 4  # ceil(7 / 2)
    assert stats["indexed"] == 7
    assert len(_ids(store)) == 7


def test_second_build_re_embeds_nothing_and_keeps_the_whole_collection(tmp_path):
    """The incremental path. If an upsert were a replace, this would collapse to 0 documents."""
    _, store = _build(tmp_path)
    embedder = FakeDocumentEmbedder(dim=DIM)
    stats, store = _build(tmp_path, embedder=embedder, store=store)
    assert stats["indexed"] == 0
    assert stats["unchanged"] == 3
    assert embedder.calls == 0
    assert _ids(store) == ["c0", "c1", "c2"], "an incremental build must not truncate the store"


def test_only_changed_chunks_are_re_embedded(tmp_path):
    _, store = _build(tmp_path)
    FakeClient.chunks = [_chunk(0), _chunk(1, content="chunk number 1 (revised)"), _chunk(2)]
    embedder = FakeDocumentEmbedder(dim=DIM)
    stats, store = _build(tmp_path, embedder=embedder, store=store)
    assert stats["indexed"] == 1
    assert stats["unchanged"] == 2
    assert _ids(store) == ["c0", "c1", "c2"]
    revised = [d for d in store.filter_documents() if d.id == "c1"][0]
    assert "revised" in revised.content


def test_rebuild_clears_the_manifest_and_re_embeds_everything(tmp_path):
    _, store = _build(tmp_path)
    embedder = FakeDocumentEmbedder(dim=DIM)
    stats, _ = _build(tmp_path, rebuild=True, embedder=embedder, store=store)
    assert stats["indexed"] == 3
    assert stats["unchanged"] == 0
    assert embedder.calls == 1


def test_one_failing_source_does_not_abort_the_build(tmp_path):
    """Inherited from redamon's data_ingestion: a flaky feed costs its own chunks, not the build."""
    FakeClient.fail = True
    stats, store = _build(tmp_path, profile="both")
    assert stats["errors"] == 1
    assert "upstream is down" in stats["sources"]["fake"]["error"]
    assert stats["sources"]["other"]["indexed"] == 1
    assert stats["indexed"] == 1
    assert _ids(store) == ["o1"]


def test_a_build_where_everything_fails_leaves_the_collection_untouched(tmp_path):
    _, store = _build(tmp_path)
    FakeClient.fail = True
    stats, store = _build(tmp_path, store=store)
    assert stats["errors"] == 1
    assert stats["indexed"] == 0
    assert _ids(store) == ["c0", "c1", "c2"]


def test_duplicate_chunk_ids_within_a_batch_are_collapsed(tmp_path):
    FakeClient.chunks = [_chunk(0), _chunk(0, content="second version"), _chunk(1)]
    stats, store = _build(tmp_path)
    assert stats["indexed"] == 2
    # Reported apart from `unchanged`: a duplicate is upstream repeating an id (expected),
    # an unchanged chunk is the manifest working.
    assert stats["duplicates"] == 1
    assert stats["unchanged"] == 0
    assert _ids(store) == ["c0", "c1"]


def test_a_concurrent_build_is_refused(tmp_path):
    settings = _settings(tmp_path)
    cache_dir = settings.resolve(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with ingest_lock(cache_dir):
        with pytest.raises(IngestLockError):
            _build(tmp_path)


def test_filter_unchanged_reports_dedup_and_manifest_hits():
    chunks = [_chunk(0), _chunk(1), _chunk(1, content="newer")]
    fresh, manifest = build.filter_unchanged(chunks, {})
    assert [c["chunk_id"] for c in fresh] == ["c0", "c1"]
    assert manifest["c1"] == build.content_hash("newer")

    again, manifest2 = build.filter_unchanged(chunks, manifest)
    assert again == []
    assert manifest2 == manifest


def test_kb_version_changes_with_the_model_but_not_between_reads():
    pins = {"exploitdb": {"ref": "a" * 40}}
    one = build.kb_version("2026-01-01T00:00:00Z", "model-a", pins)
    assert one == build.kb_version("2026-01-01T00:00:00Z", "model-a", pins)
    assert one != build.kb_version("2026-01-01T00:00:00Z", "model-b", pins)
    assert one != build.kb_version("2026-01-02T00:00:00Z", "model-a", pins)


def test_nvd_fetch_kwargs_keep_the_cvss_floor_and_pass_the_cursor():
    settings = KbSettings(nvd_lookback_days=30, nvd_min_cvss=8.0)
    kwargs = build._fetch_kwargs(
        "nvd", settings, "full", "key", {"timestamp": "2026-01-01T00:00:00Z"}, rebuild=False
    )
    # Upstream's "full" profile drops the CVSS floor, which is what explodes the corpus; we
    # always ask for the filtered variant.
    assert kwargs["profile"] == "standard"
    assert kwargs["nvd_min_cvss"] == 8.0
    assert kwargs["nvd_days"] == 30
    assert kwargs["since"] == "2026-01-01T00:00:00Z"
    # A rebuild must not send `since`, or it would "restore" a nearly empty NVD set.
    assert "since" not in build._fetch_kwargs("nvd", settings, "full", None, {"timestamp": "x"}, rebuild=True)
    assert build._fetch_kwargs("exploitdb", settings, "full", None, None, rebuild=False) == {}


def test_format_stats_surfaces_errors(tmp_path):
    FakeClient.fail = True
    text = build.format_stats(_build(tmp_path, profile="both")[0])
    assert "ERROR" in text and "upstream is down" in text
