"""Shared schema, pattern loading, file crawling, and function extraction for the
code-triage skill.

Ported from T3MP3ST's src/recon/code-ingest.ts + src/recon/ts-parse.ts, collapsed
to a single Python-`re` backend (no ripgrep) plus tree-sitter for function
resolution. Every external capability is fail-open: tree-sitter missing or a parse
error degrades to raw hit lines, never a crash.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Crawl configuration — mirrors code-ingest.ts DEFAULT_EXCLUDES + the multilang
# include-extension set. maxFileBytes (1 MB) is a security control, not just a
# perf guard: it bounds the memory a single hostile file can force us to read.
# ---------------------------------------------------------------------------

INCLUDE_EXTS: tuple[str, ...] = (
    ".py", ".js", ".ts", ".tsx", ".go", ".java", ".c", ".cpp",
)

DEFAULT_EXCLUDES: tuple[str, ...] = (
    "node_modules", ".git", "dist", "build", "venv", ".venv",
    "__pycache__", "site-packages",
    # test dirs
    "test", "tests", "__tests__", "testing",
)

MAX_FILE_BYTES = 1_000_000

# Base priority score per exposure class (code-ingest.ts EXPOSURE_BASE). The
# `security_control` tier is kept deliberately (review finding #6): auth /
# validation functions are exactly where bypass bugs live.
EXPOSURE_BASE: dict[str, int] = {
    "exposed_externally": 100,
    "attack_surface": 80,
    "exposed_internally": 50,
    "security_control": 40,
    "neutral": 10,
}

# Primary CWE weakness class per sink label — the class a reader should assume when
# that sink appears. A triage hint, not a proof. Used for the `cwe` field in findings
# and for the SARIF taxonomy tags. Bandit findings prefer Bandit's own CWE instead.
SINK_CWE: dict[str, str] = {
    # outbound request -> SSRF
    "requests.*": "CWE-918", "urllib": "CWE-918", "urlopen": "CWE-918",
    "httpx": "CWE-918", "socket": "CWE-918", "http.Get/Post/NewRequest": "CWE-918",
    "http(s).request": "CWE-918", "client.Do/Get/Post": "CWE-918",
    "fetch()": "CWE-918", "axios": "CWE-918",
    # process/command execution -> OS command injection
    "subprocess": "CWE-78", "os.system": "CWE-78", "system()": "CWE-78",
    "popen()": "CWE-78", "exec-family()": "CWE-78", "exec.Command": "CWE-78",
    "Runtime.getRuntime": "CWE-78", "ProcessBuilder": "CWE-78",
    # dynamic code evaluation -> eval injection
    "eval()": "CWE-95", "exec()": "CWE-95",
    # unsafe deserialization
    "pickle.loads": "CWE-502", "yaml.load": "CWE-502",
    # SQL
    ".execute()": "CWE-89", ".raw()": "CWE-89",
    # file access -> path traversal
    "open()": "CWE-22",
}

# When several sinks match one function, report the most severe class.
_CWE_SEVERITY = {"CWE-78": 5, "CWE-95": 5, "CWE-502": 5, "CWE-89": 4, "CWE-918": 3, "CWE-22": 2}


def cwe_for_sink_labels(labels: list[str]) -> str | None:
    """The most severe CWE class among the matched sink labels, or None."""
    cwes = [SINK_CWE[label] for label in labels if label in SINK_CWE]
    if not cwes:
        return None
    return max(cwes, key=lambda c: _CWE_SEVERITY.get(c, 0))


# Extension -> tree_sitter_language_pack language name.
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
}

# tree-sitter node types that name a callable/class definition, per language.
# A generic descendant walk (not a .scm query) is used so the extractor survives
# grammar-version drift in tree_sitter_language_pack — the price of the source's
# precise queries not porting cleanly across grammar builds.
_DEF_NODE_TYPES: dict[str, tuple[str, ...]] = {
    "python": ("function_definition", "class_definition"),
    "javascript": ("function_declaration", "method_definition", "class_declaration"),
    "typescript": ("function_declaration", "method_definition", "class_declaration"),
    "tsx": ("function_declaration", "method_definition", "class_declaration"),
    "go": ("function_declaration", "method_declaration"),
    "java": ("method_declaration", "constructor_declaration", "class_declaration"),
    "c": ("function_definition",),
    "cpp": ("function_definition",),
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sinks.json"


# ---------------------------------------------------------------------------
# Data model — the JSON contract, stable regardless of which backend actually ran.
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    """One stage-1 sink match: a line number and the label of the sink pattern."""

    line: int
    label: str

    def to_dict(self) -> dict:
        return {"line": self.line, "label": self.label}


@dataclass
class FileHits:
    path: str
    hits: list[Hit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "hits": [h.to_dict() for h in self.hits]}


@dataclass
class Function:
    """A resolved function/method: enough to score it and to point a reader at it."""

    file: str
    name: str
    params: list[str]
    body: str
    line_start: int
    line_end: int
    decorators: str = ""  # leading decorator/annotation text, for entry-point matching


@dataclass
class Candidate:
    file: str
    function: str | None
    line_start: int
    line_end: int
    sink_matches: list[str]
    is_entry_point: bool
    exposure: str
    hop_distance: int
    ssrf_idor_flag: bool
    priority_score: int
    finding: dict | None = None

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "function": self.function,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "sink_matches": self.sink_matches,
            "is_entry_point": self.is_entry_point,
            "exposure": self.exposure,
            "hop_distance": self.hop_distance,
            "ssrf_idor_flag": self.ssrf_idor_flag,
            "finding": self.finding,
            "priority_score": self.priority_score,
        }


# ---------------------------------------------------------------------------
# Pattern loading — compile everything from config/sinks.json once.
# ---------------------------------------------------------------------------

@dataclass
class Patterns:
    sink_evidence: list[tuple[str, re.Pattern]]  # (label, compiled)
    dangerous_sink: re.Pattern                    # OR-union of sink_evidence
    outbound_request: re.Pattern
    risky_params: re.Pattern
    security_controls: re.Pattern
    entry_point_decorators: list[re.Pattern]
    entry_point_names: list[re.Pattern]


def load_patterns(config_path: Path = CONFIG_PATH) -> Patterns:
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    ci = set(cfg.get("case_insensitive", []))

    def _flags(key: str) -> int:
        return re.IGNORECASE if key in ci else 0

    sink_evidence = [
        (item["label"], re.compile(item["pattern"], _flags("sink_evidence")))
        for item in cfg["sink_evidence"]
    ]
    # DANGEROUS_SINK_RE is exactly the OR-union of the sink_evidence alternatives;
    # deriving it here keeps a single source of truth for the sink list.
    dangerous_sink = re.compile(
        "|".join(item["pattern"] for item in cfg["sink_evidence"]),
        _flags("sink_evidence"),
    )

    return Patterns(
        sink_evidence=sink_evidence,
        dangerous_sink=dangerous_sink,
        outbound_request=re.compile(cfg["outbound_request"], _flags("outbound_request")),
        risky_params=re.compile(cfg["risky_params"], _flags("risky_params")),
        security_controls=re.compile(cfg["security_controls"], _flags("security_controls")),
        entry_point_decorators=[
            re.compile(p, _flags("entry_point_decorators"))
            for p in cfg["entry_point_decorators"]
        ],
        entry_point_names=[
            re.compile(p, _flags("entry_point_names"))
            for p in cfg["entry_point_names"]
        ],
    )


# ---------------------------------------------------------------------------
# Stage 1 crawl — os.walk with in-place pruning, deterministic order, size- and dir-gated.
# ---------------------------------------------------------------------------

def _is_excluded(rel_parts: Iterable[str], exclude_globs: Iterable[str]) -> bool:
    parts = set(rel_parts)
    for ex in exclude_globs:
        if not ex:
            continue
        if ex in parts:
            return True
        if "*" in ex and ex.replace("*", "") in "/".join(rel_parts):
            return True
    return False


def crawl(
    repo_root: Path,
    include_exts: Iterable[str] = INCLUDE_EXTS,
    exclude_globs: Iterable[str] = DEFAULT_EXCLUDES,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> Iterator[Path]:
    """Yield source files under repo_root, sorted, skipping excluded dirs, wrong
    extensions, and files larger than max_file_bytes."""
    exts = tuple(include_exts)
    exclude_globs = list(exclude_globs)
    repo_root = repo_root.resolve()

    for dirpath, dirnames, filenames in os.walk(repo_root):
        rel_parts = Path(dirpath).relative_to(repo_root).parts
        # Prune in place: os.walk never descends into what we drop here, so an
        # excluded node_modules/.venv costs one check instead of 100k stat() calls.
        dirnames[:] = sorted(
            d for d in dirnames
            if not _is_excluded((*rel_parts, d), exclude_globs)
        )
        for name in sorted(filenames):
            if not name.endswith(exts):
                continue
            path = Path(dirpath) / name
            try:
                if not path.is_file() or path.stat().st_size > max_file_bytes:
                    continue
            except OSError:
                continue
            yield path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Function resolution — tree-sitter, fail-open per file.
# ---------------------------------------------------------------------------

def _first_identifier(node) -> object | None:
    """Depth-first search for the first identifier-ish node under `node` — used to
    dig a function name out of C/C++ declarator chains where there is no `name`
    field."""
    stack = list(node.children)
    while stack:
        n = stack.pop(0)
        if n.type in ("identifier", "field_identifier", "type_identifier",
                       "property_identifier", "destructor_name", "operator_name"):
            return n
        stack[:0] = list(n.children)
    return None


def _node_name(node) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return name_node.text.decode("utf-8", "replace")
    # C/C++: name lives inside the declarator subtree.
    decl = node.child_by_field_name("declarator")
    target = decl if decl is not None else node
    ident = _first_identifier(target)
    if ident is not None:
        return ident.text.decode("utf-8", "replace")
    return None


def _node_params(node) -> list[str]:
    params_node = node.child_by_field_name("parameters")
    if params_node is None:
        decl = node.child_by_field_name("declarator")
        if decl is not None:
            params_node = decl.child_by_field_name("parameters")
    if params_node is None:
        return []
    raw = params_node.text.decode("utf-8", "replace")
    return split_params(raw)


def split_params(raw: str) -> list[str]:
    """Split a parameter-list source string into individual parameter texts.

    A heuristic, not a parse: params are matched against RISKY_PARAM_RE as
    substrings downstream, so returning each comma-separated piece (rather than
    the isolated identifier) is safe and language-agnostic."""
    raw = raw.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in raw:
        if ch in "([{<":
            depth += 1
            current += ch
        elif ch in ")]}>":
            depth = max(0, depth - 1)
            current += ch
        elif ch == "," and depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _leading_decorators(content: str, start_line: int) -> str:
    """Collect contiguous decorator / annotation lines immediately above a
    definition (Python `@deco`, Java `@Annotation`), so entry-point decorator
    patterns can match. start_line is 1-indexed."""
    lines = content.splitlines()
    collected: list[str] = []
    i = start_line - 2  # line directly above the def (0-indexed)
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("@"):
            collected.append(stripped)
            i -= 1
        elif stripped == "":
            i -= 1
        else:
            break
    return "\n".join(reversed(collected))


def parse_functions(path: Path, content: str) -> tuple[list[Function], str | None]:
    """Extract functions/methods from one file via tree-sitter.

    Returns (functions, warning). On any failure — tree-sitter absent, unsupported
    language, grammar load or parse error — returns ([], warning) so the caller can
    fall back to raw hit lines rather than dropping the file (ts-parse.ts fail-open
    philosophy)."""
    lang = EXT_TO_LANG.get(path.suffix)
    if lang is None:
        return [], f"unsupported language for {path} (ext {path.suffix})"

    try:
        from tree_sitter_language_pack import configure, get_parser
        from tree_sitter_language_pack.options import PackConfig

        # Pin the same cache_dir a consuming image's build-time prefetch step
        # uses (venv/tree-sitter-cache, next to this skill's own venv), so a
        # grammar downloaded ahead of time is found here later under a
        # different user/HOME -- the package's default cache location is
        # HOME-relative. A missing/uncached grammar still fails open (caught
        # below), just via a real network fetch instead of a hit.
        configure(PackConfig(cache_dir=str(Path(__file__).resolve().parent.parent / "venv" / "tree-sitter-cache")))
    except Exception as exc:  # ImportError or a broken install
        return [], f"tree-sitter unavailable ({exc.__class__.__name__}); function resolution skipped"

    source = content.encode("utf-8")
    try:
        parser = get_parser(lang)
        tree = parser.parse(source)
    except Exception as exc:
        return [], f"parse failed for {path} ({exc.__class__.__name__}); kept raw hits"

    def_types = _DEF_NODE_TYPES.get(lang, ())
    functions: list[Function] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in def_types:
            name = _node_name(node)
            if name:
                line_start = node.start_point[0] + 1
                line_end = node.end_point[0] + 1
                functions.append(
                    Function(
                        file=str(path),
                        name=name,
                        params=_node_params(node),
                        body=source[node.start_byte:node.end_byte].decode("utf-8", "replace"),
                        line_start=line_start,
                        line_end=line_end,
                        decorators=_leading_decorators(content, line_start),
                    )
                )
        stack.extend(node.children)

    return functions, None
