"""Source registry: name -> curation client class, and profile -> client instances.

Ported from redamon's `curation/data_ingestion.py::_resolve_clients`, minus the Neo4j/FAISS
wiring. Client classes are imported lazily: the NVD and ExploitDB clients pull httpx and read
settings, and nothing here should be paid for by an import of `security_agent.knowledge_base`
in a test that only needs the mapping.

Upstream also ships GTFOBins, LOLBAS, OWASP WSTG and Nuclei clients. They are deliberately not
carried here: SEC-bench is memory-safety bugs in C/C++ found by sanitizers, and privilege
escalation via setuid binaries, Windows LOLBins, web-app testing methodology and network
detection templates do not help fix a heap overflow. They were also the volume-skewed half of
the corpus. Adding one back is a ~200-line client against `BaseClient` plus an entry here — do
that when an eval asks for it, not before.
"""
from __future__ import annotations

from typing import Any, Optional

# Source name -> "module:ClassName" inside security_agent.knowledge_base.curation.
CLIENTS: dict[str, str] = {
    "cwe": "cwe_client:CweClient",
    "nvd": "nvd_client:NVDClient",
    "exploitdb": "exploitdb_client:ExploitDBClient",
}


def client_class(source: str) -> type:
    """Import and return the client class for `source`."""
    try:
        spec = CLIENTS[source]
    except KeyError:
        raise ValueError(
            f"Unknown knowledge-base source {source!r}. Known sources: "
            f"{', '.join(sorted(CLIENTS))}."
        ) from None
    module_name, class_name = spec.split(":")
    module = __import__(
        f"security_agent.knowledge_base.curation.{module_name}", fromlist=[class_name]
    )
    return getattr(module, class_name)


def resolve_clients(
    sources: list[str], cache_root: Optional[Any] = None
) -> list[tuple[str, Any]]:
    """Instantiate the clients for `sources`, in the given order.

    `cache_root` (a Path) overrides each client's cache directory — the tests point it at
    `tmp_path` so a build never touches the real download cache.
    """
    from pathlib import Path

    resolved: list[tuple[str, Any]] = []
    for name in sources:
        cls = client_class(name)
        if cache_root is not None:
            resolved.append((name, cls(cache_dir=str(Path(cache_root) / name))))
        else:
            resolved.append((name, cls()))
    return resolved
