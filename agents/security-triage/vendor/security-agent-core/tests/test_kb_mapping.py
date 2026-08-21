"""Pins the chunk-dict -> Haystack Document contract.

This mapping is the only surface between the vendored curation clients and Haystack, and two of
its rules are silent failures if broken:

* the title prepend — without it, BM25 cannot find an NVD record by its CVE id, because the id
  lives only in the chunk's `title`;
* `cvss_score` as a float with the key ABSENT when unknown — a stored None or a string would
  make the `min_cvss` filter either raise or silently drop every result.

Both are asserted against the real filter evaluator, not just the dict shape.
"""
import pytest
from haystack.utils.filters import document_matches_filter

from security_agent.knowledge_base.mapping import (
    ChunkMappingError,
    chunk_to_document,
    documents_from_chunks,
)

# One representative chunk per source, shaped exactly as each client's to_chunks() emits it.
NVD_CHUNK = {
    "chunk_id": "a1b2c3d4e5f60718",
    "content": "runc allows a container escape via an internal file descriptor leak.",
    "title": "CVE-2024-21626",
    "source": "nvd",
    "cve_id": "CVE-2024-21626",
    "cvss_score": 8.6,
    "severity": "high",
    "affected_products": ["runc 1.0.0-rc93", "docker 24.0.0"],
    "published_date": "2024-01-31",
    "source_path": "data/kb_cache/nvd/nvd_cache.json",
}
EXPLOITDB_CHUNK = {
    "chunk_id": "3333333333333333",
    "content": "Apache 2.4.49 - Path Traversal",
    "title": "EDB-50383: Apache 2.4.49 - Path Traversal",
    "source": "exploitdb",
    "edb_id": "50383",
    "cve_id": "CVE-2021-41773",
    "platform": "multiple",
    "published_date": "2021-10-05",
    "codes": ["CVE-2021-41773", "OSVDB-1234"],
    "source_path": "data/kb_cache/exploitdb/files_exploits.csv",
}
CWE_CHUNK = {
    "chunk_id": "4444444444444444",
    "content": "CWE-787: Out-of-bounds Write. The product writes data past the end of a buffer.",
    "title": "CWE-787: Out-of-bounds Write",
    "source": "cwe",
    "cwe_id": "CWE-787",
    "source_path": "data/cwe_reference.json",
}
ALL_CHUNKS = [NVD_CHUNK, EXPLOITDB_CHUNK, CWE_CHUNK]


@pytest.mark.parametrize("chunk", ALL_CHUNKS, ids=lambda c: c["source"])
def test_document_id_is_the_chunk_id(chunk):
    # Builds are idempotent and the manifest keys are document ids only because of this.
    assert chunk_to_document(chunk).id == chunk["chunk_id"]


def test_title_is_prepended_when_the_content_does_not_carry_it():
    # The CVE id lives only in `title`; without the prepend, BM25 cannot find this record by id.
    doc = chunk_to_document(NVD_CHUNK)
    assert doc.content.startswith("CVE-2024-21626\n\n")
    assert "CVE-2024-21626" in doc.content


def test_title_is_not_duplicated_when_the_content_already_starts_with_it():
    """ExploitDB's content is its title; prepending again would double every row."""
    chunk = {**EXPLOITDB_CHUNK, "content": EXPLOITDB_CHUNK["title"] + "\nDetails."}
    doc = chunk_to_document(chunk)
    assert doc.content == chunk["content"]
    assert doc.content.count(EXPLOITDB_CHUNK["title"]) == 1


@pytest.mark.parametrize("chunk", ALL_CHUNKS, ids=lambda c: c["source"])
def test_meta_has_no_none_values_and_only_simple_types(chunk):
    meta = chunk_to_document(chunk).meta
    for key, value in meta.items():
        assert value is not None, f"{key} stored as None"
        assert isinstance(value, (str, int, float, bool, list)), f"{key} is {type(value)}"
        if isinstance(value, list):
            assert all(isinstance(v, str) for v in value), f"{key} has non-str items"


@pytest.mark.parametrize("chunk", ALL_CHUNKS, ids=lambda c: c["source"])
def test_unwhitelisted_chunk_fields_are_dropped(chunk):
    meta = chunk_to_document(chunk).meta
    for dropped in ("codes", "content", "chunk_id"):
        assert dropped not in meta


def test_cvss_is_numeric_and_absent_when_unknown():
    assert chunk_to_document(NVD_CHUNK).meta["cvss_score"] == pytest.approx(8.6)
    assert isinstance(chunk_to_document(NVD_CHUNK).meta["cvss_score"], float)
    # A client that sends cvss_score=None: the key must be missing, not None, or a `>=` filter
    # would have to reason about nulls.
    scoreless = chunk_to_document({**NVD_CHUNK, "cvss_score": None, "cve_id": None})
    assert "cvss_score" not in scoreless.meta
    assert "cve_id" not in scoreless.meta


def test_list_meta_is_deduplicated_and_capped():
    """NVD repeats the same CPE string for each affected version range."""
    doc = chunk_to_document({
        **NVD_CHUNK,
        "affected_products": ["cpe:openssl"] * 7 + ["cpe:runc"] + [""] + [None],
    })
    assert doc.meta["affected_products"] == ["cpe:openssl", "cpe:runc"]

    long_doc = chunk_to_document({**NVD_CHUNK, "affected_products": [f"p{i}" for i in range(50)]})
    assert len(long_doc.meta["affected_products"]) == 20


def test_cvss_from_a_string_is_coerced():
    doc = chunk_to_document({**NVD_CHUNK, "cvss_score": "9.8"})
    assert doc.meta["cvss_score"] == pytest.approx(9.8)


def test_min_cvss_filter_selects_the_expected_documents():
    """The real filter evaluator, on the real mapping — the semantics the tool depends on."""
    docs = [chunk_to_document(c) for c in ALL_CHUNKS]
    docs.append(chunk_to_document({**NVD_CHUNK, "chunk_id": "low", "cvss_score": 3.1}))
    # The OR form the sub-agent's system prompt teaches: a floor on the scored source, plus an
    # arm that admits the sources which never carry a score.
    filters = {
        "operator": "OR",
        "conditions": [
            {"field": "meta.cvss_score", "operator": ">=", "value": 7.0},
            {"field": "meta.source", "operator": "not in", "value": ["nvd"]},
        ],
    }
    kept = [d.id for d in docs if document_matches_filter(filters, d)]
    # The high-scoring NVD record passes and the low-scoring one does not; CWE and ExploitDB
    # never carry a score and must be unaffected by the floor rather than silently dropped.
    assert "low" not in kept
    assert {NVD_CHUNK["chunk_id"], EXPLOITDB_CHUNK["chunk_id"], CWE_CHUNK["chunk_id"]} <= set(kept)


def test_source_filter_is_an_in_clause():
    docs = [chunk_to_document(c) for c in ALL_CHUNKS]
    filters = {"field": "meta.source", "operator": "in", "value": ["cwe", "nvd"]}
    kept = {d.meta["source"] for d in docs if document_matches_filter(filters, d)}
    assert kept == {"cwe", "nvd"}


def test_missing_required_fields_raise():
    with pytest.raises(ChunkMappingError):
        chunk_to_document({"content": "x", "source": "nvd"})
    with pytest.raises(ChunkMappingError):
        chunk_to_document({"chunk_id": "x", "content": "y"})
    with pytest.raises(ChunkMappingError):
        chunk_to_document({"chunk_id": "x", "source": "nvd", "content": "", "title": ""})


def test_duplicate_chunk_ids_collapse_keeping_the_last_and_preserving_order():
    """The public ExploitDB CSV repeats ids; the store's OVERWRITE policy means last wins."""
    chunks = [
        {**CWE_CHUNK, "chunk_id": "dup", "content": "first"},
        NVD_CHUNK,
        {**CWE_CHUNK, "chunk_id": "dup", "content": "second"},
    ]
    docs = documents_from_chunks(chunks)
    assert [d.id for d in docs] == ["dup", NVD_CHUNK["chunk_id"]]
    assert docs[0].content.endswith("second")


def test_unmappable_chunks_are_skipped_not_fatal():
    docs = documents_from_chunks([{"nonsense": True}, CWE_CHUNK])
    assert [d.id for d in docs] == [CWE_CHUNK["chunk_id"]]
