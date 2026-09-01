#!/usr/bin/env python3
"""Stage 1 — cheap whole-repo sweep.

Walk the repo once with pathlib + Python `re`, flag every file that contains a
dangerous-sink pattern, and record which patterns matched on which lines. Emits
structured JSON — never file contents — so a downstream agent can decide where to
spend a slower, deeper look (stage 2, deepen.py).

Single backend by design: one Python-`re` pass. No ripgrep — in the NemoClaw
sandbox an external `rg` is blocked by the binary allowlist, and a static musl
build can't run the source's look-behind guards anyway (build-plan findings #1/#2).

Usage:
    sweep.py <repo_root> [--json] [--max-file-bytes N]

Output (stdout):
    {"backend_used": "python-re", "root": "...", "files": [
        {"path": "src/upload.py", "hits": [{"line": 12, "label": "subprocess"}]}
    ]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import FileHits, Hit, MAX_FILE_BYTES, crawl, load_patterns, read_text


def sweep(repo_root: Path, max_file_bytes: int = MAX_FILE_BYTES) -> list[FileHits]:
    patterns = load_patterns()
    results: list[FileHits] = []

    for path in crawl(repo_root, max_file_bytes=max_file_bytes):
        content = read_text(path)
        if content is None:
            continue

        hits: list[Hit] = []
        for lineno, line in enumerate(content.splitlines(), start=1):
            for label, regex in patterns.sink_evidence:
                if regex.search(line):
                    hits.append(Hit(line=lineno, label=label))

        if hits:
            try:
                rel = str(path.relative_to(repo_root.resolve()))
            except ValueError:
                rel = str(path)
            results.append(FileHits(path=rel, hits=hits))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-1 whole-repo sink sweep.")
    parser.add_argument("repo_root", help="Path to the repository to sweep.")
    parser.add_argument(
        "--max-file-bytes", type=int, default=MAX_FILE_BYTES,
        help=f"Skip files larger than this (default {MAX_FILE_BYTES}).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="(Accepted for symmetry; output is always JSON.)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    if not repo_root.is_dir():
        print(json.dumps({"error": f"not a directory: {repo_root}"}), file=sys.stderr)
        return 2

    files = sweep(repo_root, max_file_bytes=args.max_file_bytes)
    output = {
        "backend_used": "python-re",
        "root": str(repo_root.resolve()),
        "files": [f.to_dict() for f in files],
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
