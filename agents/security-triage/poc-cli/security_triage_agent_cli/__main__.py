# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``security-triage-agent`` CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="security-triage-agent",
        description="Security triage agent runtime for NemoClaw: scan a repository with a Haystack Agent.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    gw = sub.add_parser("gateway", help="Start the health-probe HTTP gateway.")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=8661)

    rn = sub.add_parser("run", help="Scan one repository and write a JSON report (findings + trajectory).")
    rn.add_argument("--repo", default=".", help="Path to the repository to scan (default: cwd).")
    rn.add_argument("--out", default=None, help="Reports directory (default: /sandbox/.security-triage/reports).")
    rn.add_argument("--config", default=None, help="Agent YAML (default: uploaded agent.yaml, else the packaged seed).")
    rn.add_argument("--model", default=None, help="Served model name (default: $NEMOCLAW_MODEL, else the YAML's).")
    rn.add_argument("--max-steps", type=int, default=None, help="Override the YAML's max_agent_steps.")
    rn.add_argument("--name", default=None, help="Report name (default: the repo directory name).")
    rn.add_argument("--print-report", action="store_true", help="Also print the parsed report JSON to stdout.")

    args = parser.parse_args(argv)

    if args.command == "gateway":
        from .gateway import serve

        serve(args.host, args.port)
        return 0

    if args.command == "run":
        from .agent_runtime import DEFAULT_REPORTS_DIR, run_once

        # One scan is a single blocking Agent.run() with no intermediate output; on a real repo it
        # can go minutes between lines. Heartbeat so a watcher can tell "working" from "hung".
        stop = threading.Event()

        def _heartbeat() -> None:
            start = time.monotonic()
            while not stop.wait(30):
                print(f"[security-triage-agent] still scanning... {int(time.monotonic() - start)}s elapsed",
                      file=sys.stderr, flush=True)

        threading.Thread(target=_heartbeat, daemon=True).start()
        try:
            record = run_once(
                args.repo, config_path=args.config, model=args.model, max_steps=args.max_steps, name=args.name
            )
        finally:
            stop.set()

        from security_agent.repo_scan import summary_line

        out_dir = Path(args.out) if args.out else DEFAULT_REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{record['repo']}.json"
        target.write_text(json.dumps(record, indent=2, default=str))
        print(summary_line(record), flush=True)
        print(f"report: {target}", flush=True)
        if args.print_report:
            print(json.dumps(record.get("report"), indent=2), flush=True)
        return 0 if not record.get("error") else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
