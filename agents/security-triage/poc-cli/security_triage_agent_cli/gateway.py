# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal long-running HTTP gateway exposing a health surface.

Mirrors deep-research's gateway contract:

    GET /health  -> 200 {"status": "ok", "platform": "security-triage-agent"}
    (any other)  -> 404

The gateway is a harmless liveness process only -- it keeps the sandbox's
foreground process alive. An actual triage run is a separate, on-demand
``security-triage-agent run`` invocation, never inline here.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__


class _HealthHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") in ("/health", "") or self.path == "/health":
            self._send(
                200,
                {"status": "ok", "platform": "security-triage-agent", "version": __version__},
            )
        else:
            self._send(404, {"status": "not_found", "path": self.path})

    def log_message(self, fmt: str, *args) -> None:  # keep stdout clean-ish
        import sys

        sys.stderr.write("[security-triage-agent gateway] " + (fmt % args) + "\n")


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), _HealthHandler)
    print(
        f"[security-triage-agent] gateway listening on http://{host}:{port} (GET /health)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
