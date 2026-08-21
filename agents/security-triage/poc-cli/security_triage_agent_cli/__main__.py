# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``security-triage-agent`` CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="security-triage-agent",
        description="Security triage agent runtime for NemoClaw (POC).",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    gw = sub.add_parser("gateway", help="Start the health-probe HTTP gateway.")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=8661)

    rn = sub.add_parser("run", help="Run a one-shot triage over a repo and print the report.")
    # No URL/clone flag: the target repo arrives as the sandbox's working
    # directory (policy-additions.yaml `include_workdir: true`), so this
    # defaults to cwd.
    rn.add_argument("--repo", default=".", help="Path to the repo to triage (default: cwd).")

    args = parser.parse_args(argv)

    if args.command == "gateway":
        from .gateway import serve

        serve(args.host, args.port)
        return 0

    if args.command == "run":
        from .agent_runtime import run_once

        report, meta = run_once(args.repo)
        print(f"[security-triage-agent run-meta] {json.dumps(meta, default=str)}", file=sys.stderr, flush=True)
        print(report, flush=True)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
