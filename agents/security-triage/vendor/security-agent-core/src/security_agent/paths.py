"""Filesystem locations for the security agent."""
from __future__ import annotations

from pathlib import Path

# The package lives at <project>/src/security_agent; data, benchmark data, and run outputs live at
# the project root (PROJECT_ROOT), resolved relative to this file so they are found whether run from
# a checkout or an install.
PKG_DIR = Path(__file__).resolve().parent          # <project>/src/security_agent
PROJECT_ROOT = PKG_DIR.parents[1]                   # <project> (security-agent-core)

DATA_DIR = PROJECT_ROOT / "data"
CWE_REFERENCE = DATA_DIR / "cwe_reference.json"
# Security knowledge base: the per-source feed downloads. Gitignored and reproducible from
# `secagent kb build` at the pinned feed commits (knowledge_base.pins.FEED_PINS). The corpus
# itself lives in Qdrant, not on this filesystem — see docker-compose.yml.
KB_CACHE_DIR = DATA_DIR / "kb_cache"
BENCHMARK_DATA = PROJECT_ROOT / "benchmark_data"     # pre-sampled per-benchmark task files
TRACE_DIR = PROJECT_ROOT / "traces"
TRACE_FILE = TRACE_DIR / "traces.jsonl"              # eval traces


def optimize_trace_file(benchmark: str) -> Path:
    """OTLP trace file for an optimize campaign, per benchmark (kept apart from eval traces so
    meta-agent + target-agent spans from a run don't tangle with eval spans)."""
    return TRACE_DIR / f"optimize-{benchmark}.jsonl"
