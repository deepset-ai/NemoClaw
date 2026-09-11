# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security-triage runtime: one repository scan by a YAML-configured Haystack Agent.

The agent itself lives in the vendored security-agent-core (`security_agent.repo_scan`):
`OpenAIChatGenerator` on NemoClaw's managed inference route (`OPENAI_BASE_URL` =
`https://inference.local/v1`, placeholder key, model from `NEMOCLAW_MODEL`), the
`run_bash` / `run_python` / `read_file` / `write_file` tools bound to *this* sandbox through
`security_agent.local_sandbox.LocalSandbox` (the OpenShell sandbox is the isolation boundary;
nothing nests a container), `search_security_kb` as an HTTP client to the host-side knowledge
base (`SECURITY_KB_URL`), and the bundled `code-triage` skill through `SkillToolset` +
`run_skill_script`.

Which YAML runs, in order of precedence: `--config`, then an operator-uploaded
`/sandbox/.security-triage/agent.yaml` (the "harness config" a self-improvement loop produces
and a human approves), then `$SECURITY_SCAN_CONFIG` (the image's packaged seed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_SKILLS_DIR = Path(os.environ.get("SECURITY_TRIAGE_SKILLS_DIR", "/opt/security-agent-core/skills"))
STATE_DIR = Path(os.environ.get("SECURITY_TRIAGE_STATE_DIR", "/sandbox/.security-triage"))
OPERATOR_CONFIG = STATE_DIR / "agent.yaml"
DEFAULT_REPORTS_DIR = STATE_DIR / "reports"


def resolve_config_path(explicit: Optional[str]) -> Optional[Path]:
    """`--config` > uploaded operator YAML > the packaged seed (None lets repo_scan pick it)."""
    if explicit:
        return Path(explicit)
    if OPERATOR_CONFIG.is_file():
        return OPERATOR_CONFIG
    return None


def run_once(
    repo_path: str,
    *,
    config_path: Optional[str] = None,
    model: Optional[str] = None,
    max_steps: Optional[int] = None,
    skills_dir: Optional[Path] = None,
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Scan `repo_path` and return the JSON-able record (report, raw answer, trajectory, timing)."""
    from security_agent import repo_scan

    chosen = resolve_config_path(config_path)
    config = repo_scan.apply_overrides(repo_scan.load_config(chosen), model=model, max_steps=max_steps)
    record = repo_scan.scan(config, Path(repo_path), skills_dir=skills_dir or DEFAULT_SKILLS_DIR, name=name)
    record["config"] = str(chosen or os.environ.get(repo_scan.CONFIG_ENV) or repo_scan.DEFAULT_CONFIG)
    return record
