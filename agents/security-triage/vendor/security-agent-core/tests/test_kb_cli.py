"""Covers `secagent kb` argument parsing and dispatch (the build itself is faked out).

The one non-obvious thing asserted: `kb` takes no `--benchmark`. One knowledge base serves every
benchmark, and `main()` must keep working for a namespace that has no `benchmark` attribute.
"""
import pytest
from haystack.dataclasses import Document

from security_agent import cli


@pytest.fixture
def recorded_build(monkeypatch):
    calls = []

    def fake_build_kb(settings, **kwargs):
        calls.append({"settings": settings, **kwargs})
        return {
            "profile": kwargs.get("profile") or "?", "store": ":memory:/security_kb",
            "sources": {}, "indexed": 3, "unchanged": 0, "duplicates": 0, "errors": 0, "total": 3,
            "kb_version": "abc", "seconds": 0.1,
        }

    monkeypatch.setattr("security_agent.knowledge_base.build.build_kb", fake_build_kb)
    return calls


def test_build_dispatches_with_the_requested_profile(recorded_build, capsys):
    assert cli.main(["kb", "build", "--profile", "standard"]) == 0
    assert len(recorded_build) == 1
    assert recorded_build[0]["profile"] == "standard"
    assert recorded_build[0]["rebuild"] is False
    out = capsys.readouterr().out
    assert "indexed=3" in out
    # The build prints the seed snippet so seeds/<benchmark>.yaml cannot drift from the store.
    assert "embedding_model" in out and "query_prefix" in out


def test_rebuild_sets_the_rebuild_flag(recorded_build):
    assert cli.main(["kb", "rebuild"]) == 0
    assert recorded_build[0]["rebuild"] is True


def test_a_partial_build_exits_non_zero(monkeypatch):
    monkeypatch.setattr(
        "security_agent.knowledge_base.build.build_kb",
        lambda settings, **kw: {
            "profile": "full", "store": "x", "sources": {"nvd": {"error": "RuntimeError: 503"}},
            "indexed": 0, "unchanged": 0, "duplicates": 0, "errors": 1, "seconds": 1.0,
        },
    )
    assert cli.main(["kb", "build"]) == 1


def test_build_flags_reach_the_settings(recorded_build):
    cli.main([
        "kb", "build", "--profile", "standard", "--device", "mps", "--batch-size", "500",
        "--st-batch-size", "8", "--nvd-days", "30", "--nvd-min-cvss", "9.0", "--offline",
        "--source", "nvd", "--nvd-key", "secret",
    ])
    call = recorded_build[0]
    settings = call["settings"]
    assert settings.device == "mps"
    assert settings.batch_size == 500
    assert settings.st_batch_size == 8
    assert settings.nvd_lookback_days == 30
    assert settings.nvd_min_cvss == 9.0
    assert settings.offline is True
    assert call["source"] == "nvd"
    assert call["nvd_api_key"] == "secret"


def test_qdrant_flags_reach_the_settings(recorded_build):
    cli.main(["kb", "build", "--qdrant-url", ":memory:", "--qdrant-index", "other"])
    settings = recorded_build[0]["settings"]
    assert settings.qdrant_url == ":memory:"
    assert settings.qdrant_index == "other"


def test_kb_needs_no_benchmark_selector(recorded_build):
    # `--benchmark` belongs to the other commands; passing it here is an error, and omitting it
    # must work (main() reads it with getattr(..., None)).
    assert cli.main(["kb", "build"]) == 0
    with pytest.raises(SystemExit):
        cli.main(["kb", "build", "--benchmark", "secbench"])


def test_an_unknown_profile_is_rejected_by_argparse():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["kb", "build", "--profile", "gigantic"])
    assert excinfo.value.code == 2


def test_stats_on_an_unbuilt_store_exits_non_zero_with_the_setup_hint(capsys):
    assert cli.main(["kb", "stats", "--qdrant-url", ":memory:", "--qdrant-index", "nope"]) == 1
    assert "docker compose up -d qdrant" in capsys.readouterr().err


def test_stats_prints_the_recorded_provenance(monkeypatch, capsys):
    meta = {
        "kb_version": "v1", "built_at": "2026-07-31T00:00:00Z", "profile": "full",
        "embedding_model": "BAAI/bge-small-en-v1.5", "sparse_model": "Qdrant/bm25",
        "dim": 384, "device": "cpu", "total": 70123, "counts": {"nvd": 7500, "cwe": 969},
    }
    monkeypatch.setattr(
        "security_agent.knowledge_base.store.read_meta", lambda *a, **k: meta
    )
    assert cli.main(["kb", "stats"]) == 0
    out = capsys.readouterr().out
    assert "v1" in out and "70123" in out and "nvd" in out and "969" in out
    assert "Qdrant/bm25" in out


class _FakeSearch:
    """Records how the CLI drove `SecurityKbSearch`."""

    captured: dict = {}

    def __init__(self, **kwargs):
        _FakeSearch.captured = {"init": kwargs}

    def retrieve(self, query, filters=None, top_k=None):
        _FakeSearch.captured["retrieve"] = {"query": query, "filters": filters, "top_k": top_k}
        return [Document(id="d1", content="a heap overflow", meta={"source": "nvd", "title": "CVE-1"})]

    def run(self, **kwargs):
        _FakeSearch.captured["run"] = kwargs
        return {"results": "the agent's answer"}


def test_query_runs_retrieval_only_by_default(monkeypatch, capsys):
    """`kb query` is the pre-flight check for the retrieval stack: no LLM, no API key, no
    sub-agent. `--filters` takes the same Haystack filter the sub-agent writes, so the flag
    exercises the real path instead of a parallel `--source`/`--min-cvss` translation layer."""
    monkeypatch.setattr("security_agent.knowledge_base.search.SecurityKbSearch", _FakeSearch)
    assert cli.main([
        "kb", "query", "heap overflow", "--top-k", "7", "--no-rerank",
        "--filters", '{"field": "meta.source", "operator": "in", "value": ["nvd", "cwe"]}',
    ]) == 0
    captured = _FakeSearch.captured
    assert "run" not in captured, "the default path must not invoke the knowledge-base agent"
    assert captured["retrieve"]["query"] == "heap overflow"
    assert captured["retrieve"]["top_k"] == 7
    assert captured["retrieve"]["filters"] == {
        "field": "meta.source", "operator": "in", "value": ["nvd", "cwe"]
    }
    assert captured["init"]["use_reranker"] is False
    assert "source=nvd" in capsys.readouterr().out


def test_query_without_filters_passes_none(monkeypatch):
    monkeypatch.setattr("security_agent.knowledge_base.search.SecurityKbSearch", _FakeSearch)
    assert cli.main(["kb", "query", "heap overflow"]) == 0
    assert _FakeSearch.captured["retrieve"]["filters"] is None


def test_query_rejects_malformed_filter_json(monkeypatch, capsys):
    monkeypatch.setattr("security_agent.knowledge_base.search.SecurityKbSearch", _FakeSearch)
    assert cli.main(["kb", "query", "x", "--filters", "{not json"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_query_agent_flag_runs_the_real_tool(monkeypatch, capsys):
    monkeypatch.setattr("security_agent.knowledge_base.search.SecurityKbSearch", _FakeSearch)
    assert cli.main(["kb", "query", "how is CWE-787 fixed?", "--agent"]) == 0
    assert _FakeSearch.captured["run"] == {"query": "how is CWE-787 fixed?"}
    assert "the agent's answer" in capsys.readouterr().out
