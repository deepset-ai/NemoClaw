"""Locks the contract of the vendored redamon curation code.

These tests exist because the code is vendored, not depended on: a future re-vendor from a newer
upstream commit, or a "harmless" local tidy-up, must not silently change

* how chunk ids are derived — every stored Document.id and every chunk-manifest key would
  invalidate at once, so a rebuild would look like a full re-ingest and incremental builds would
  never hit;
* what the NVD/ExploitDB/CWE chunk dicts contain — `knowledge_base.mapping` is written against
  those exact keys;
* the network and archive safety bounds, which are the first layer of the prompt-injection
  defence.

Everything here is offline: the safe_get assertions fire before any socket is opened.
"""
import json

import pytest

from security_agent.knowledge_base import pins
from security_agent.knowledge_base.chunking import ChunkStrategy
from security_agent.knowledge_base.curation import safe_http
from security_agent.knowledge_base.curation.cwe_client import CweClient
from security_agent.knowledge_base.curation.exploitdb_client import ExploitDBClient
from security_agent.knowledge_base.curation.nvd_client import NVDClient

# Golden value: sha256("nvd:CVE-2024-1")[:16]. If this changes, every document id and manifest
# entry in every existing store silently stops matching.
GOLDEN_CHUNK_ID = "3771b3fd17355a69"


def test_chunk_id_derivation_is_stable():
    assert ChunkStrategy.generate_chunk_id("nvd", "CVE-2024-1") == GOLDEN_CHUNK_ID
    # Deterministic and source-scoped.
    assert ChunkStrategy.generate_chunk_id("nvd", "CVE-2024-1") == GOLDEN_CHUNK_ID
    assert ChunkStrategy.generate_chunk_id("exploitdb", "CVE-2024-1") != GOLDEN_CHUNK_ID
    assert len(GOLDEN_CHUNK_ID) == 16


def test_chunk_budget_comes_from_config_and_fits_the_embedder():
    # The chunker must stay under the 512-token cap of bge-small; the estimator is chars/4.
    assert ChunkStrategy.MAX_CHUNK_TOKENS <= 512
    assert ChunkStrategy.PREFERRED_CHUNK_TOKENS <= ChunkStrategy.MAX_CHUNK_TOKENS


def test_chunking_defaults_survive_a_broken_config(monkeypatch):
    """Read at class-definition time, so it must never raise — see chunking._load_chunking_defaults."""
    import security_agent.knowledge_base.chunking as chunking

    monkeypatch.setattr(
        "security_agent.knowledge_base.settings.load_kb_settings",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt config")),
    )
    assert chunking._load_chunking_defaults() == (480, 256)


def test_markdown_chunking_merges_small_sections_and_splits_large_ones():
    strategy = ChunkStrategy()
    text = "## One\nshort\n\n## Two\nalso short\n"
    chunks = strategy.chunk_markdown(text)
    assert len(chunks) == 1  # both sections are < 128 tokens, so they merge
    assert "One" in chunks[0]["title"]

    big = "## Big\n" + "\n\n".join(["word " * 200] * 8)
    parts = strategy.chunk_markdown(big)
    assert len(parts) > 1
    assert any("(part 2)" in p["title"] for p in parts)
    assert all(strategy.estimate_tokens(p["content"]) <= strategy.MAX_CHUNK_TOKENS for p in parts)


def test_structured_chunking_truncates_oversized_content():
    strategy = ChunkStrategy()
    entry = {"content": "x " * (strategy.MAX_CHUNK_TOKENS * 4)}
    out = strategy.chunk_structured([entry])
    assert out[0]["content"].endswith("...")
    assert len(out[0]["content"]) <= strategy.MAX_CHUNK_TOKENS * 4 + 3


# --- network + archive bounds (no socket is opened) ------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/payload.tar.gz",
        "http://127.0.0.1:8080/x",           # SSRF to a local service
        "https://raw.githubusercontent.com.evil.example.com/x",  # suffix trick
    ],
)
def test_safe_get_rejects_hosts_outside_the_allowlist(url):
    with pytest.raises(safe_http.UntrustedHostError):
        safe_http.safe_get(url)


def test_safe_get_rejects_non_http_schemes():
    with pytest.raises(safe_http.UntrustedHostError):
        safe_http.safe_get("file:///etc/passwd")


def test_allowlist_covers_every_host_our_feeds_use():
    hosts = set(safe_http.DEFAULT_ALLOWED_HOSTS)
    assert {"services.nvd.nist.gov", "codeload.github.com", "gitlab.com"} <= hosts
    # A cap exists on every download path, so a hostile feed cannot exhaust the disk.
    assert safe_http.MAX_DOWNLOAD_BYTES > 0
    assert safe_http.MAX_NVD_PAGE_BYTES <= safe_http.MAX_DOWNLOAD_BYTES


# --- chunk shapes the mapping is written against ---------------------------

def test_nvd_to_chunks_emits_the_documented_keys():
    chunks = NVDClient(cache_dir="/nonexistent").to_chunks([
        {
            "cve_id": "CVE-2024-1",
            "description": "A heap overflow in the row decoder.",
            "cvss_score": 9.8,
            "severity": "critical",
            "affected_products": ["libpng 1.6.0"],
            "published_date": "2024-01-01",
            "source_path": "data/kb_cache/nvd/nvd_cache.json",
        }
    ])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert {"chunk_id", "content", "title", "source", "cve_id", "cvss_score", "severity",
            "affected_products", "published_date", "source_path"} <= set(chunk)
    assert chunk["source"] == "nvd"
    assert chunk["title"] == "CVE-2024-1"
    assert chunk["cvss_score"] == 9.8
    assert "CVSS: 9.8 (critical)" in chunk["content"]
    # The CVE id is NOT in the content — this is why mapping prepends the title.
    assert "CVE-2024-1" not in chunk["content"]
    assert chunk["source_path"].startswith("data/kb_cache/")


def test_exploitdb_to_chunks_emits_the_documented_keys():
    chunks = ExploitDBClient(cache_dir="/nonexistent").to_chunks([
        {
            "id": "389",
            "description": "LibPNG Graphics Library - Remote Buffer Overflow",
            "platform": "linux",
            "type": "remote",
            "date_published": "2004-08-04",
            "codes": "CVE-2004-0597;OSVDB-8312",
            "source_path": "data/kb_cache/exploitdb/files_exploits.csv",
        }
    ])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert {"chunk_id", "content", "title", "source", "edb_id", "platform",
            "published_date"} <= set(chunk)
    assert chunk["source"] == "exploitdb"
    assert chunk["edb_id"] == "389"
    # The CVE is lifted out of the semicolon-delimited `codes` column into an indexed field.
    assert chunk["cve_id"] == "CVE-2004-0597"


def test_cwe_client_reads_the_local_reference(tmp_path):
    reference = tmp_path / "cwe.json"
    reference.write_text(json.dumps({
        "CWE-787": {"name": "Out-of-bounds Write", "description": "Writes past the end."},
        "CWE-120": {"name": "Buffer Copy", "description": "Classic overflow."},
    }))
    client = CweClient(cache_dir=str(tmp_path), reference_path=str(reference))
    chunks = client.to_chunks(client.fetch())
    assert len(chunks) == 2
    assert {c["cwe_id"] for c in chunks} == {"CWE-787", "CWE-120"}
    assert all(c["source"] == "cwe" and c["chunk_id"] for c in chunks)
    # Ids are stable across runs — the manifest depends on it.
    assert [c["chunk_id"] for c in chunks] == [
        c["chunk_id"] for c in client.to_chunks(client.fetch())
    ]


def test_cwe_client_survives_a_missing_reference(tmp_path):
    client = CweClient(cache_dir=str(tmp_path), reference_path=str(tmp_path / "nope.json"))
    assert client.fetch() == []


# --- pins ------------------------------------------------------------------

def test_every_network_feed_is_pinned_to_an_immutable_commit():
    for source in pins.FEED_PINS:
        ref = pins.get_feed_ref(source)
        assert len(ref) == 40 and all(c in "0123456789abcdef" for c in ref), source
    # NVD is an API and `cwe` is committed to this repo, so `exploitdb` is the only pinned feed.
    assert set(pins.FEED_PINS) == {"exploitdb"}


def test_unpinned_feed_fails_loudly():
    with pytest.raises(KeyError):
        pins.get_feed_ref("nope")


def test_verify_sha256_no_ops_without_a_recorded_hash_and_raises_on_mismatch(monkeypatch):
    pins.verify_sha256("exploitdb", b"anything")  # no recorded sha256 -> no-op
    monkeypatch.setitem(pins.FEED_PINS, "exploitdb", {"ref": "0" * 40, "sha256": "deadbeef"})
    with pytest.raises(pins.PinMismatchError):
        pins.verify_sha256("exploitdb", b"anything")


def test_retrieval_models_are_listed_in_model_pins():
    """The revision is what keeps stored vectors and query vectors from drifting apart."""
    from security_agent.knowledge_base.settings import KbSettings

    defaults = KbSettings()
    assert defaults.embedding_model in pins.MODEL_PINS
    assert defaults.reranker_model in pins.MODEL_PINS
