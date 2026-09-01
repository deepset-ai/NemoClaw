"""Knowledge-base settings: the `knowledge_base:` block of config.yaml.

Deliberately separate from `security_agent.optimize.settings`. That module's whole purpose is
per-benchmark precedence (benchmark block > cross-cutting section > default) keyed on the active
benchmark; the knowledge base is benchmark-independent, so threading it through that machinery
would advertise a per-benchmark override that does not exist.

Precedence per key: caller argument > environment variable > config.yaml > dataclass default.

`load_kb_settings` must never raise: `chunking.ChunkStrategy` reads it at class-definition time,
so an exception here would break every import of a curation client. A missing or corrupt
config.yaml degrades to the dataclass defaults.

These values configure the BUILD side. The query-side component
(`knowledge_base.search.SecurityKbSearch`) never reads this file — it carries its own init
parameters so seeds/secbench.yaml stays the single source of truth and stays optimizer-mutable.
`secagent kb build` prints the matching seed snippet so the two cannot drift.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Profile -> source names. Ordered cheapest-first so a profile is a prefix of the next one.
# `dev` is fully offline (the CWE reference is committed), which is what CI and the integration
# test build against.
DEFAULT_PROFILES: dict[str, list[str]] = {
    "dev": ["cwe"],
    "standard": ["cwe", "nvd"],
    "full": ["cwe", "nvd", "exploitdb"],
}


@dataclass(frozen=True)
class KbSettings:
    """Resolved knowledge-base configuration."""

    cache_dir: str = "data/kb_cache"
    profile: str = "full"

    # The Qdrant deployment holding the corpus. `docker compose up -d qdrant` serves the default;
    # tests pass ":memory:", which runs the same code against an ephemeral in-process Qdrant.
    qdrant_url: str = "http://localhost:6333"
    qdrant_index: str = "security_kb"
    qdrant_timeout: float = 60.0

    # Retrieval models. `embedding_model` is coupled to the collection: changing it invalidates
    # every stored vector, which `store.check_model` refuses to ignore.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # bge models are asymmetric: the instruction goes on the QUERY side only.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""
    # The lexical leg, as a sparse vector. `Qdrant/bm25` is BM25 proper rather than a learned
    # sparse model — SPLADE-class models are semantic and would duplicate the dense leg.
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # "cpu" (never None): letting sentence-transformers auto-select picks MPS on macOS, so the
    # same text would embed differently depending on which machine built vs queried the store.
    device: str = "cpu"
    # Documents per indexing-pipeline run. Bounds peak memory during a 70k-chunk build.
    batch_size: int = 2000
    # sentence-transformers encode batch size.
    st_batch_size: int = 64
    offline: bool = False

    # Chunking budget. Must stay under the embedder's 512-token cap; the chunker's estimator is
    # chars/4, so 480 leaves slop.
    max_chunk_tokens: int = 480
    preferred_chunk_tokens: int = 256

    # NVD sizing knobs. 90 days at CVSS>=7 is ~7.5k chunks; 730 days would alone exceed 60k.
    nvd_lookback_days: int = 90
    nvd_min_cvss: float = 7.0

    profiles: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_PROFILES))

    def sources(self, profile: Optional[str] = None) -> list[str]:
        """Source names for `profile` (defaults to the configured one)."""
        name = profile or self.profile
        try:
            return list(self.profiles[name])
        except KeyError:
            raise ValueError(
                f"Unknown knowledge-base profile {name!r}. Known profiles: "
                f"{', '.join(sorted(self.profiles))}."
            ) from None

    def resolve(self, path_str: str) -> Path:
        """Resolve a project-relative path against the project root."""
        from security_agent import paths

        p = Path(path_str)
        return p if p.is_absolute() else paths.PROJECT_ROOT / p


def _env_str(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() or None if value else None


def _env_int(name: str) -> Optional[int]:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not an integer", name, raw)
        return None


def _env_float(name: str) -> Optional[float]:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not a number", name, raw)
        return None


def _env_bool(name: str) -> Optional[bool]:
    raw = _env_str(name)
    if raw is None:
        return None
    return raw.lower() in {"1", "true", "yes", "on"}


# config.yaml key -> env var. Only keys worth overriding per-invocation are listed.
_ENV_OVERRIDES: dict[str, Any] = {
    "qdrant_url": ("KB_QDRANT_URL", _env_str),
    "qdrant_index": ("KB_QDRANT_INDEX", _env_str),
    "profile": ("KB_PROFILE", _env_str),
    "embedding_model": ("KB_EMBEDDING_MODEL", _env_str),
    "reranker_model": ("KB_RERANKER_MODEL", _env_str),
    "device": ("KB_DEVICE", _env_str),
    "batch_size": ("KB_BATCH_SIZE", _env_int),
    "st_batch_size": ("KB_ST_BATCH_SIZE", _env_int),
    "offline": ("KB_OFFLINE", _env_bool),
    "nvd_lookback_days": ("NVD_LOOKBACK_DAYS", _env_int),
    "nvd_min_cvss": ("NVD_MIN_CVSS", _env_float),
}

_FIELDS = {f for f in KbSettings.__dataclass_fields__}


def load_kb_settings(project_root: Optional[Path] = None, **overrides: Any) -> KbSettings:
    """Load knowledge-base settings; never raises.

    `overrides` (typically CLI flags) win over everything; `None` values are ignored so callers
    can pass argparse results straight through.
    """
    if project_root is None:
        from security_agent import paths

        project_root = paths.PROJECT_ROOT

    values: dict[str, Any] = {}

    # 1. config.yaml
    try:
        import yaml

        raw = yaml.safe_load((Path(project_root) / "config.yaml").read_text()) or {}
        block = raw.get("knowledge_base") or {}
        if not isinstance(block, dict):
            raise TypeError(f"knowledge_base must be a mapping, got {type(block).__name__}")
        values.update({k: v for k, v in block.items() if k in _FIELDS and v is not None})
    except FileNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - defensive; see module docstring
        logger.warning("Could not read the knowledge_base config block (%s); using defaults", exc)

    # 2. environment
    for key, (env_name, parse) in _ENV_OVERRIDES.items():
        parsed = parse(env_name)
        if parsed is not None:
            values[key] = parsed

    # 3. caller arguments
    values.update({k: v for k, v in overrides.items() if k in _FIELDS and v is not None})

    try:
        settings = KbSettings(**values)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Invalid knowledge-base settings (%s); using defaults", exc)
        return KbSettings()

    # A profile map from YAML may be partial; fill in the built-ins it omits so `dev` always works.
    if settings.profiles is not DEFAULT_PROFILES:
        merged = {**DEFAULT_PROFILES, **settings.profiles}
        if merged != settings.profiles:
            settings = replace(settings, profiles=merged)
    return settings
