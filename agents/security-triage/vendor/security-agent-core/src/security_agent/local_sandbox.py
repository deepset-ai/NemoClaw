"""Run the code-execution tools in the *current* process's environment, not in a Docker container.

`container_tools` (`run_bash`, `run_python`, `read_file`, `write_file`) act on whatever
`use_container(...)` binds. `docker_eval.ContainerHandle` binds a per-task Docker container; this
module binds the process's own filesystem and shell. It exists for one deployment shape: the agent
already runs inside an isolation boundary that it cannot nest containers in — a NemoClaw/OpenShell
sandbox. There the sandbox itself is the "local sandbox" of the architecture (deny-by-default egress,
Landlock filesystem policy, non-root user), so a second container would add nothing except a Docker
socket the policy rightly does not grant.

Consequently this backend provides **no isolation of its own**. Do not use it on a developer
machine to run model-written commands; use `sandbox.start_sandbox()` there. `LocalSandbox`
inherits read/write/diff from `ShellContainer`, so the tools behave identically on both backends —
same `timeout -k 5 N bash -lc` wrapper, same head+tail output cap — and a YAML written for one runs
unchanged on the other.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from security_agent.docker_eval import ShellContainer, bound_output


class LocalSandbox(ShellContainer):
    """A `ShellContainer` whose `exec` is a subprocess in this process's environment.

    :param work_dir: Working directory for every command; created if missing.
    :param default_timeout: Seconds a command may run when the tool passes no timeout.
    :param env: Extra environment variables for the commands (merged over `os.environ`).
    """

    def __init__(self, work_dir: str, default_timeout: int = 120, env: Optional[dict[str, str]] = None) -> None:
        self.work_dir = str(Path(work_dir).resolve())
        self.default_timeout = default_timeout
        self._env = {**os.environ, **(env or {})}
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

    def exec(
        self,
        command: str,
        timeout: Optional[int] = None,
        max_output_chars: int | None = None,
    ) -> tuple[int, str]:
        wrapped, secs = self._wrap(command, timeout)
        try:
            proc = subprocess.run(
                wrapped,
                shell=True,
                cwd=self.work_dir,
                env=self._env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # `timeout -k 5` inside the wrapper is the real budget; this outer limit only
                # guards against the wrapper itself never returning.
                timeout=secs + 15,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or b"").decode("utf-8", "replace")
            return 124, bound_output(partial + f"\n(killed: exceeded {secs}s)", max_output_chars)
        return proc.returncode, bound_output(proc.stdout.decode("utf-8", "replace"), max_output_chars)
