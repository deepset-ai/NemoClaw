"""Opt-in integration tests for the security knowledge base.

Skipped by default: unlike the rest of the suite these download HuggingFace models (both) and
hit the network (the second one). They run against a real Qdrant — `docker compose up -d qdrant`
— in a scratch collection they drop on the way in. Enable with

    SECAGENT_KB_INTEGRATION=1 pytest tests/test_kb_integration.py   # real models, offline feeds
    SECAGENT_KB_NETWORK=1     pytest tests/test_kb_integration.py   # + a real upstream fetch
    SECAGENT_KB_AGENT=1       pytest tests/test_kb_integration.py   # + the real LLM (costs tokens)

Guarded with `skipif` rather than a custom marker so `[tool.pytest.ini_options]` stays as it is
(only `testpaths`) and the default behaviour of the other modules is unchanged.

These are the only tests that prove the real embedder, the real cross-encoder, the hybrid wiring
and the store artifact work together — everything else substitutes a double for at least one of
them. Most stop at the retrieval stack, which `warm_up_retrieval()` / `retrieve()` exist to make
reachable without an LLM; the `SECAGENT_KB_AGENT` tier goes one step further and drives the agent
pack with a live model.
"""
import os

import pytest

INTEGRATION = pytest.mark.skipif(
    not os.getenv("SECAGENT_KB_INTEGRATION"),
    reason="set SECAGENT_KB_INTEGRATION=1 (downloads bge-small + ms-marco-MiniLM)",
)
NETWORK = pytest.mark.skipif(
    not os.getenv("SECAGENT_KB_NETWORK"),
    reason="set SECAGENT_KB_NETWORK=1 (fetches an upstream feed at its pinned commit)",
)
AGENT = pytest.mark.skipif(
    not (os.getenv("SECAGENT_KB_AGENT") and os.getenv("OPENAI_API_KEY")),
    reason="set SECAGENT_KB_AGENT=1 and OPENAI_API_KEY (real LLM calls, real tokens)",
)

# A scratch collection, recreated by each test that builds into it, so a run never touches the
# real `security_kb` corpus on the same Qdrant.
INDEX = "security_kb_integration_test"


@INTEGRATION
def test_dev_profile_build_then_query_with_the_real_models(tmp_path):
    """End-to-end on the offline `dev` profile (the committed CWE reference): build the store
    with the real embedder and the real BM25 sparse encoder, then retrieve through Qdrant's
    hybrid fusion and the cross-encoder."""
    from security_agent.knowledge_base import search
    from security_agent.knowledge_base.build import build_kb
    from security_agent.knowledge_base.search import SecurityKbSearch
    from security_agent.knowledge_base.settings import KbSettings, load_kb_settings
    from security_agent.knowledge_base.store import read_meta

    defaults = load_kb_settings()
    settings = KbSettings(
        qdrant_index=INDEX,
        cache_dir=str(tmp_path / "cache"),
        profile="dev",
        embedding_model=defaults.embedding_model,
        embedding_dim=defaults.embedding_dim,
        query_prefix=defaults.query_prefix,
        passage_prefix=defaults.passage_prefix,
        device="cpu",
    )
    stats = build_kb(settings, profile="dev", rebuild=True)
    assert stats["errors"] == 0
    assert stats["indexed"] > 900, "the committed CWE reference has ~969 entries"

    meta = read_meta(settings=settings)
    assert meta["dim"] == defaults.embedding_dim
    assert meta["embedding_model"] == defaults.embedding_model

    component = SecurityKbSearch(
        qdrant_index=INDEX,
        embedding_model=defaults.embedding_model,
        query_prefix=defaults.query_prefix,
        reranker_model=defaults.reranker_model,
        device="cpu",
        top_k=5,
    )
    component.warm_up_retrieval()
    assert component._error is None, component._error

    def titles(query, **kwargs):
        return " ".join(d.meta["title"] for d in component.retrieve(query, **kwargs))

    # Semantic recall: no shared vocabulary with the CWE titles, so this only works if the dense
    # leg and the reranker are both doing their job.
    found = titles("unbounded copy into a fixed stack buffer")
    assert any(cwe in found for cwe in ("CWE-120", "CWE-121", "CWE-787")), found

    # Exact-id recall goes through the sparse BM25 leg.
    assert "CWE-416" in titles("CWE-416")

    # A filter authored the way the knowledge-base agent authors them, against the real store.
    filtered = component.retrieve(
        "out of bounds write",
        filters={"field": "meta.cwe_id", "operator": "==", "value": "CWE-787"},
    )
    assert [d.meta["cwe_id"] for d in filtered] == ["CWE-787"]

    # A second component reuses the connection and the warmed models.
    twin = SecurityKbSearch(
        qdrant_index=INDEX,
        embedding_model=defaults.embedding_model,
        query_prefix=defaults.query_prefix,
        reranker_model=defaults.reranker_model,
        device="cpu",
        top_k=5,
    )
    twin.warm_up_retrieval()
    assert twin._pipeline is component._pipeline
    search.clear_agent_cache()


@INTEGRATION
def test_reranker_changes_the_ordering(tmp_path):
    """If the cross-encoder is not earning its latency, that is worth knowing explicitly."""
    from security_agent.knowledge_base import search
    from security_agent.knowledge_base.build import build_kb
    from security_agent.knowledge_base.search import SecurityKbSearch
    from security_agent.knowledge_base.settings import KbSettings, load_kb_settings
    from security_agent.knowledge_base.store import read_meta

    defaults = load_kb_settings()
    settings = KbSettings(
        qdrant_index=INDEX, cache_dir=str(tmp_path / "cache"), profile="dev",
        embedding_model=defaults.embedding_model, embedding_dim=defaults.embedding_dim,
        query_prefix=defaults.query_prefix, device="cpu",
    )
    build_kb(settings, profile="dev", rebuild=True)

    def top_titles(use_reranker):
        component = SecurityKbSearch(
            qdrant_index=INDEX, embedding_model=defaults.embedding_model,
            query_prefix=defaults.query_prefix, reranker_model=defaults.reranker_model,
            device="cpu", top_k=5, use_reranker=use_reranker,
        )
        component.warm_up_retrieval()
        assert component._error is None, component._error
        docs = component.retrieve("use after free of a linked list node", top_k=5)
        return [d.meta["title"] for d in docs]

    with_rerank, without = top_titles(True), top_titles(False)
    assert with_rerank and without
    assert with_rerank != without, "the cross-encoder made no difference to the top 5"
    search.clear_agent_cache()


@NETWORK
def test_exploitdb_fetches_at_its_pinned_commit_and_maps_cleanly(tmp_path):
    """Catches upstream schema drift at a bumped pin (and proves the pin is fetchable).

    ExploitDB is the only feed with a commit pin: NVD is an API and the CWE reference is
    committed to this repo.
    """
    from security_agent.knowledge_base.curation.exploitdb_client import ExploitDBClient
    from security_agent.knowledge_base.mapping import chunk_to_document

    client = ExploitDBClient(cache_dir=str(tmp_path / "exploitdb"))
    chunks = client.to_chunks(client.fetch())
    assert len(chunks) > 30_000, f"only {len(chunks)} ExploitDB chunks — upstream CSV may have moved"
    for chunk in chunks[:2000]:
        doc = chunk_to_document(chunk)  # must not raise for any chunk
        assert doc.meta["source"] == "exploitdb"
        assert doc.meta["edb_id"]


@INTEGRATION
@AGENT
def test_the_knowledge_base_agent_answers_from_the_real_store(tmp_path):
    """The one test that runs the agent pack for real: real models, real store, real LLM.

    Everything else stubs `make_llm`, which means it proves the wiring but not that a live model
    can drive these five tools to an answer. That is the claim worth paying tokens for — and it
    is the same code path `search_security_kb` takes inside a benchmark task.
    """
    from security_agent.knowledge_base import search
    from security_agent.knowledge_base.build import build_kb
    from security_agent.knowledge_base.sanitize import FRAME_BEGIN, FRAME_END
    from security_agent.knowledge_base.search import SecurityKbSearch
    from security_agent.knowledge_base.settings import KbSettings, load_kb_settings

    defaults = load_kb_settings()
    settings = KbSettings(
        qdrant_index=INDEX, cache_dir=str(tmp_path / "cache"), profile="dev",
        embedding_model=defaults.embedding_model, embedding_dim=defaults.embedding_dim,
        query_prefix=defaults.query_prefix, passage_prefix=defaults.passage_prefix, device="cpu",
    )
    build_kb(settings, profile="dev", rebuild=True)

    component = SecurityKbSearch(
        qdrant_index=INDEX,
        embedding_model=defaults.embedding_model,
        query_prefix=defaults.query_prefix,
        reranker_model=defaults.reranker_model,
        device="cpu",
        top_k=5,
    )
    component.warm_up()
    assert component.error is None, component.error

    out = component.run(
        query="Which CWE covers writing past the end of a fixed-size buffer, and what is the "
              "usual fix?"
    )["results"]
    assert out.startswith(FRAME_BEGIN) and out.rstrip().endswith(FRAME_END)
    assert any(cwe in out for cwe in ("CWE-787", "CWE-120", "CWE-121")), out
    # It cited something it actually retrieved, rather than answering from general knowledge.
    assert "Documents consulted:" in out and "[doc " in out

    # The corpus is CWE-only in the `dev` profile, so a question it cannot answer must be
    # refused rather than answered from the model's own knowledge.
    refused = component.run(query="What is the CVSS score of CVE-2024-21626?")["results"]
    assert "No matching information was found" in refused, refused

    search.clear_agent_cache()
