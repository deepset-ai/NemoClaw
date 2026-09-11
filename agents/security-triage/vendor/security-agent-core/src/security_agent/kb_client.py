"""`search_security_kb` as a thin HTTP client, for agents that run where the KB stack cannot.

`knowledge_base.search.SecurityKbSearch` needs Qdrant, fastembed and a torch cross-encoder in the
same process. Inside a NemoClaw/OpenShell sandbox none of that is present (the image is ~400 MB,
the reranker alone is ~2.5 GB of torch), and the sandbox has no route to Qdrant anyway. So the
knowledge base stays a host service — `scripts/kb_server.py`, Qdrant behind it — and the sandbox
agent calls it over the one bridge endpoint the policy grants (`host.openshell.internal`).

This component is that client. Same tool name, same one-string-in / one-string-out contract as the
in-process tool, so an agent YAML swaps one `component.type` and nothing else. Failures come back as
text (the agent should read "the KB is unreachable" and move on), never as an exception that kills a
scan; `raise_on_tool_invocation_failure: false` in the seeds relies on that.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from haystack import component, default_from_dict, default_to_dict

DEFAULT_KB_URL_ENV = "SECURITY_KB_URL"
DEFAULT_TIMEOUT = 60.0
DEFAULT_TOP_K = 5


@component
class RemoteKbSearch:
    """Ask the host-side security knowledge base (`scripts/kb_server.py`) for a cited briefing.

    :param url: Base URL of the KB server. Defaults to the `SECURITY_KB_URL` environment variable,
        read at run time so the YAML carries no deployment address.
    :param top_k: Passages to return.
    :param timeout: Seconds to wait for the server.
    """

    def __init__(self, url: str | None = None, top_k: int = DEFAULT_TOP_K, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.url = url
        self.top_k = top_k
        self.timeout = timeout

    def _base_url(self) -> str | None:
        return (self.url or os.environ.get(DEFAULT_KB_URL_ENV) or "").rstrip("/") or None

    @component.output_types(results=str)
    def run(self, query: str) -> dict:
        base = self._base_url()
        if base is None:
            return {"results": f"(error: no knowledge base configured; set {DEFAULT_KB_URL_ENV})"}
        body = json.dumps({"query": query, "top_k": self.top_k}).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/search", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            return {"results": f"(error: knowledge base returned HTTP {exc.code})"}
        except Exception as exc:  # unreachable host must read as a tool result, not a dead scan
            return {"results": f"(error: knowledge base unreachable: {type(exc).__name__})"}
        return {"results": str(payload.get("results", "")) or "(no results)"}

    def to_dict(self) -> dict:
        return default_to_dict(self, url=self.url, top_k=self.top_k, timeout=self.timeout)

    @classmethod
    def from_dict(cls, data: dict) -> "RemoteKbSearch":
        return default_from_dict(cls, data)
