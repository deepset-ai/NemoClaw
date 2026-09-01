# Vendored from redamon (MIT, Copyright (c) 2026 Samuele Giampieri, Ritesh Gohil).
# Upstream: knowledge_base/curation/pins.py @ 380554275fdfcf177737253132bd094178314078
# See the repo-root NOTICE and knowledge_base/curation/LICENSE-redamon.
# Local changes: moved up out of curation/; MODEL_PINS replaced with the two models this project uses.

"""
Feed & model integrity pins (STRIDE T15).

Every KB curation feed is pinned to an **immutable upstream commit** so a
poisoned or silently-mutated upstream cannot land in the RAG corpus and survive
restarts as a persistent prompt-injection surface.

Two integrity anchors, in order of strength:

1. **Commit-SHA in the URL (primary, always on).** Each client builds its
   download URL from the pinned commit here instead of a mutable
   ``refs/heads/<branch>`` ref. A git commit is immutable, so this alone
   guarantees the client fetches exactly the reviewed tree — silent upstream
   mutation is impossible.
2. **sha256 of the downloaded artifact (optional, defence-in-depth).** Enforced
   only when a non-``None`` hash is recorded for the feed. It is intentionally
   left ``None`` for the GitHub *source archive* feeds (``/archive/<sha>.tar.gz``)
   because those tarballs are NOT guaranteed byte-reproducible over time
   (GitHub has changed its archive gzip/git plumbing before, e.g. Sept 2023),
   so a hard sha256 on them would risk a spurious ingest abort — a
   non-breaking violation. Feeds whose bytes ARE stable (a raw file blob at a
   fixed commit) may carry a sha256 for full enforcement.

**Bumping a pin** is a deliberate, reviewed change: update the SHA here and
rebuild the agent + kb-refresh images (curation code is image-baked). A moved
or removed pinned commit fails the *fetch* loudly (per-feed abort) rather than
silently ingesting whatever ``main`` now points at.

Model revisions are pinned the same way so the embedder/reranker load a fixed
HuggingFace commit rather than "latest", keeping embeddings deterministic.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PinMismatchError(RuntimeError):
    """Raised when a downloaded artifact's sha256 does not match its pin.

    A curation client raises this from ``fetch()``; ``data_ingestion`` catches
    it and aborts the single offending feed, leaving the prior corpus intact
    (fail-closed per-feed, not whole-KB).
    """


# ---------------------------------------------------------------------------
# Feed pins — commit SHA (immutable) + optional sha256 of the fetched artifact.
# SHAs captured 2026-07-10 (each repo's then-current default-branch HEAD), so
# pinning is byte-content-identical to what an unpinned fetch returned that day.
# ---------------------------------------------------------------------------
FEED_PINS: dict[str, dict[str, Optional[str]]] = {
    # GitLab raw file blob at a fixed commit — byte-stable, so a sha256 could
    # be recorded here for full enforcement without spurious-abort risk. Left
    # None by default (commit-pin is the anchor); set to enable the second check.
    "exploitdb": {"ref": "6a57361db10b709c4b4f339c09ee250acc2d8aac", "sha256": None},
}
# NVD is fetched from its REST API rather than a repo, so it has no commit to pin; `cwe` is a
# file committed to this repo. Only `exploitdb` needs a pin today. The sha256 machinery below
# stays because a re-added tarball feed needs it — see `profiles.CLIENTS`.

# ---------------------------------------------------------------------------
# Model pins — HuggingFace commit revisions. Passed as revision=<sha> so the
# embedder/reranker load a fixed commit instead of the moving branch head.
# ---------------------------------------------------------------------------
#
# Pinning matters more here than it does upstream: vectors in the store artifact and vectors
# computed at query time must come from the same weights. An upstream model-card commit that
# touches config.json or the pooling config silently changes embeddings, which would degrade
# retrieval with no error anywhere.
#
# Captured 2026-07-31 with
#   python -c "from huggingface_hub import HfApi; print(HfApi().model_info('<id>').sha)"
# Bumping one is a deliberate change: the embedding pin invalidates every stored vector, so it
# must be paired with `secagent kb rebuild`.
MODEL_PINS: dict[str, Optional[str]] = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    # The sparse leg. `Qdrant/bm25` is not a neural model — fastembed ships it as a tokenizer
    # plus an IDF table, and exposes no revision parameter to pin it with, so the version comes
    # from the pinned `fastembed-haystack` dependency instead. Listed here so it is visible
    # alongside the two that ARE pinned rather than looking overlooked.
    "Qdrant/bm25": None,
}


def get_feed_ref(source: str) -> str:
    """Return the pinned commit/tag ref for a feed.

    Raises KeyError if the feed is unpinned — a loud failure is correct here:
    a new feed must be pinned before it ships, never fetched from a branch head.
    """
    return FEED_PINS[source]["ref"]  # type: ignore[return-value]


def get_feed_sha256(source: str) -> Optional[str]:
    """Return the recorded sha256 for a feed's artifact, or None if unset."""
    return FEED_PINS.get(source, {}).get("sha256")


def verify_sha256(source: str, data: bytes) -> None:
    """Verify ``data`` against the feed's recorded sha256, if one is set.

    No-op when the feed has no recorded hash (commit-pin URL is then the sole
    anchor). Raises :class:`PinMismatchError` on mismatch so the caller aborts
    the feed before extraction/parsing.
    """
    expected = get_feed_sha256(source)
    if not expected:
        return
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise PinMismatchError(
            f"{source}: downloaded artifact sha256 {actual} does not match "
            f"pinned {expected} — aborting feed ingest (possible upstream "
            f"tampering or an un-bumped pin)."
        )
    logger.info(f"{source}: artifact sha256 verified against pin")


def model_revision(model_name: str) -> Optional[str]:
    """Return the pinned HuggingFace revision for a model, or None if unpinned.

    Unpinned models fall back to the library default (branch head) with a
    warning rather than failing — an operator may configure a custom model the
    manifest does not know about, and blocking that would be breaking.
    """
    rev = MODEL_PINS.get(model_name)
    if rev is None and model_name in MODEL_PINS:
        logger.warning(
            f"Model {model_name!r} has no captured revision in MODEL_PINS; loading its "
            f"default revision (branch head). Capture the sha for reproducibility."
        )
    elif rev is None:
        logger.warning(
            f"Model {model_name!r} is not pinned in MODEL_PINS; loading its "
            f"default revision (branch head). Add a pin for reproducibility."
        )
    return rev
