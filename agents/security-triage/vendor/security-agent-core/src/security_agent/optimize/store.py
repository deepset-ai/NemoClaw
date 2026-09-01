"""Content-addressed config store, champion pointer, and campaign journal.

Everything lives under the runs directory:

- ``configs/<sha256>.yaml``     every config ever produced
- ``champion.json``             pointer to the current best config + its score
- ``journal.jsonl``             one record per optimization event (append-only)
- ``components/``               versioned copies of generated component sources
- ``extensions.json``           registered component extensions (tool catalog additions)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from security_agent.optimize.component_authoring import ComponentExtension


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.configs_dir = root / "configs"
        self.components_dir = root / "components"
        self.champion_file = root / "champion.json"
        self.journal_file = root / "journal.jsonl"
        self.extensions_file = root / "extensions.json"
        for d in (self.root, self.configs_dir, self.components_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- configs ---

    def save_config(self, config: dict) -> str:
        digest = config_hash(config)
        path = self.configs_dir / f"{digest}.yaml"
        if not path.exists():
            path.write_text(yaml.safe_dump(config, sort_keys=False))
        return digest

    def load_config(self, digest: str) -> dict:
        path = self.configs_dir / f"{digest}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No stored config with hash '{digest}'.")
        return yaml.safe_load(path.read_text())

    # --- champion ---

    def champion(self) -> dict[str, Any] | None:
        if not self.champion_file.exists():
            return None
        return json.loads(self.champion_file.read_text())

    def set_champion(self, digest: str, score: float, metrics: dict[str, Any] | None = None) -> None:
        record = {"hash": digest, "score": score, "metrics": metrics or {}}
        self.champion_file.write_text(json.dumps(record, indent=2))
        # Keep a readable copy of the champion config next to the pointer.
        (self.root / "champion.yaml").write_text(
            yaml.safe_dump(self.load_config(digest), sort_keys=False)
        )

    def load_champion_config(self) -> dict:
        champion = self.champion()
        if champion is None:
            raise FileNotFoundError("No champion set. Run `security_agent.optimize init` first.")
        return self.load_config(champion["hash"])

    # --- journal ---

    def append_journal(self, record: dict[str, Any]) -> None:
        with self.journal_file.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_journal(self) -> list[dict[str, Any]]:
        if not self.journal_file.exists():
            return []
        return [json.loads(line) for line in self.journal_file.read_text().splitlines() if line.strip()]

    def next_iteration(self) -> int:
        return sum(1 for r in self.read_journal() if r.get("type") == "iteration")

    # --- generated components ---

    def save_component_source(self, module_name: str, source: str) -> Path:
        path = self.components_dir / f"{module_name}.py"
        path.write_text(source)
        return path

    def extensions(self) -> list[ComponentExtension]:
        if not self.extensions_file.exists():
            return []
        raw = json.loads(self.extensions_file.read_text())
        return [ComponentExtension(**entry) for entry in raw]

    def add_extension(self, extension: ComponentExtension) -> None:
        entries = [dataclasses.asdict(e) for e in self.extensions()]
        entries.append(dataclasses.asdict(extension))
        self.extensions_file.write_text(json.dumps(entries, indent=2))
