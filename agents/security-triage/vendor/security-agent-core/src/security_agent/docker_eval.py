"""A thin Docker-backed verifier for SEC-bench (the `patch` task).

SEC-bench (https://github.com/SEC-bench/SEC-bench, MIT) is a BYO-agent, one-image-per-CVE
benchmark: the agent produces a unified git diff, and success is decided by *running* it
inside the instance's prebuilt Docker image (`hwiwonlee/secb.eval.x86_64.<instance_id>:patch`)
with sanitizers. Their evaluator is not pip-installable, so we reimplement just the slice we
need with the `docker` SDK, faithfully mirroring:

- the command sequence of `secb/evaluator/templates/eval_patch_script.j2`
  (`secb patch` → `secb build` → `timeout 10 secb repro`), and
- the sanitizer-report detection of `secb/evaluator/utils.py`.

Two roles:
- `ContainerHandle` + `start_container` give the SEC-bench agent a live container to edit
  source in (the tools in `security_agent.container_tools` drive it); `git_diff()` extracts
  the produced patch.
- `verify_patch` runs the produced diff against a *fresh* `:patch` container and returns a
  pass/fail verdict — this is what `SecBench.score` aggregates into a resolve rate.

Nothing here imports `docker` at module load, so importing this module (e.g. for the pure
`extract_sanitizer_report` helper in a unit test) never requires a running daemon.
"""

from __future__ import annotations

import base64
import contextlib
import re
import shlex
import tempfile
from pathlib import Path
from typing import Iterator, Optional

# Docker Hub prefix for the prebuilt per-instance eval images (from SEC-bench's
# build_eval_instances.py: SECB_IMAGE_PREFIX). The full ref is
# f"{PREFIX}.{instance_id}:{kind}" where kind is "patch" or "poc".
SECB_IMAGE_PREFIX = "hwiwonlee/secb.eval.x86_64"

# `secb repro` is wrapped in `timeout 10`; these are the shell exit codes a timeout yields.
_TIMEOUT_EXIT_CODES = (124, 137)

# Container resource limits, matching SEC-bench's config.example.toml [docker].run_kwargs.
_RUN_KWARGS = {"mem_limit": "8g", "network_mode": "host", "tty": False}

# How long to wait for the whole verify script (build can be slow); mirrors their 600s.
_VERIFY_WAIT_SECONDS = 600


def image_ref(instance_id: str, kind: str = "patch") -> str:
    """The Docker Hub reference for an instance's eval image (`:patch` or `:poc`)."""
    return f"{SECB_IMAGE_PREFIX}.{instance_id}:{kind}"


# --------------------------------------------------------------------------- #
# Sanitizer-report detection (ported from SEC-bench secb/evaluator/utils.py)
# --------------------------------------------------------------------------- #
# ASan/MSan/UBSan/LSan print a block bracketed by `==<pid>==ERROR/WARNING: <X>Sanitizer:`
# and `==<pid>==ABORTING`. We also fall back to literal markers some runs emit.
_SANITIZER_START = re.compile(r"==\d+==\s*(?:ERROR|WARNING):\s*\w+Sanitizer:")
_SANITIZER_END = re.compile(r"==\d+==ABORTING")
_SANITIZER_MARKERS = (
    "ERROR: AddressSanitizer:",
    "WARNING: MemorySanitizer:",
    "ERROR: LeakSanitizer:",
    "SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior",
    "UndefinedBehaviorSanitizer:DEADLYSIGNAL",
)


def extract_sanitizer_report(logs: str) -> Optional[str]:
    """Return the sanitizer crash report found in container logs, or None.

    A non-None return means a sanitizer fired (the vulnerability still triggers).
    """
    if not logs:
        return None
    start = _SANITIZER_START.search(logs)
    if start:
        end = _SANITIZER_END.search(logs, start.end())
        return logs[start.start() : (end.end() if end else len(logs))]
    for marker in _SANITIZER_MARKERS:
        idx = logs.find(marker)
        if idx != -1:
            return logs[idx : idx + 2000]
    return None


# --------------------------------------------------------------------------- #
# Docker client + a live container the agent edits source in
# --------------------------------------------------------------------------- #
def _client():
    import docker  # lazy: importing this module must not require docker/the daemon

    return docker.from_env()


def ensure_image(instance_id: str, kind: str = "patch") -> str:
    """Return the eval image ref, pulling it from Docker Hub if not present locally."""
    from docker.errors import ImageNotFound

    ref = image_ref(instance_id, kind)
    client = _client()
    try:
        client.images.get(ref)
    except ImageNotFound:
        client.images.pull(ref)
    return ref


class ContainerHandle:
    """A running container the SEC-bench agent operates in: run shell, read/write files,
    and finally extract the staged diff. Each shell call is wrapped in `timeout` so a
    hung command cannot stall the whole task."""

    def __init__(self, container, work_dir: str, default_timeout: int = 120) -> None:
        self._c = container
        self.work_dir = work_dir
        self.default_timeout = default_timeout

    def exec(self, command: str, timeout: Optional[int] = None) -> tuple[int, str]:
        """Run a bash command in the container at `work_dir`; return (exit_code, output)."""
        secs = int(timeout if timeout is not None else self.default_timeout)
        wrapped = f"timeout {secs} bash -lc {shlex.quote(command)}"
        res = self._c.exec_run(["bash", "-lc", wrapped], workdir=self.work_dir, demux=False)
        out = res.output.decode("utf-8", "replace") if res.output else ""
        return res.exit_code, out

    def read_file(self, path: str) -> str:
        code, out = self.exec(f"cat {shlex.quote(path)}")
        return out if code == 0 else f"(error: could not read {path}: exit {code})\n{out}"

    def write_file(self, path: str, content: str) -> str:
        # base64 round-trip keeps arbitrary content (quotes, newlines) intact.
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        code, out = self.exec(
            f"mkdir -p \"$(dirname {shlex.quote(path)})\" && "
            f"printf %s {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}"
        )
        return "ok" if code == 0 else f"(error: could not write {path}: exit {code})\n{out}"

    def git_diff(self) -> str:
        """Return an apply-able diff of the agent's edits to *tracked* files.

        Plain `git diff` (working tree vs. index, tracked files only) — NOT `git add -A` first.
        Staging everything sweeps in build artifacts the agent created by running `secb build`
        (e.g. `build/` outputs and binaries when the repo doesn't gitignore them), which pollute
        the patch and make the verifier's `git apply` fail on binary blobs. SEC-bench gold patches
        are source-only modifications to tracked files, so this matches what the verifier expects.
        (A fix that adds a brand-new file isn't captured — rare for a CVE patch.)
        """
        _, out = self.exec("git diff")
        return out


@contextlib.contextmanager
def start_container(image: str, work_dir: str) -> Iterator[ContainerHandle]:
    """Start `image` as a long-lived container and yield a handle; always removed on exit."""
    client = _client()
    container = client.containers.run(
        image,
        command=["sleep", "infinity"],
        working_dir=work_dir,
        detach=True,
        **_RUN_KWARGS,
    )
    try:
        yield ContainerHandle(container, work_dir)
    finally:
        with contextlib.suppress(Exception):
            container.remove(force=True)


# --------------------------------------------------------------------------- #
# Patch verification (mirrors eval_patch_script.j2)
# --------------------------------------------------------------------------- #
# The prediction is bind-mounted here (read-only); the script copies it to /testcase,
# where the in-image `secb patch` expects it.
_PRED_MOUNT = "/tmp/secb_pred"

# `set +e` so we can observe each step's exit code; the sentinels below let us tell how
# far we got and what `secb repro` returned, independent of the wrapper's own exit code.
_PATCH_SCRIPT = f"""set +e
cp {_PRED_MOUNT}/model_patch.diff /testcase/model_patch.diff
secb patch
if [ $? -ne 0 ]; then echo "SECB_FAIL=patch"; exit 0; fi
secb build
if [ $? -ne 0 ]; then echo "SECB_FAIL=build"; exit 0; fi
echo "SECB_REACHED_REPRO=1"
timeout 10 secb repro
echo "SECB_REPRO_EXIT=$?"
"""


def _interpret_patch(mode: str, logs: str, gold: dict) -> tuple[bool, str]:
    """Decide patch success from the verify logs, mirroring interpret_patch_results.

    strict:  reached repro, repro exited 0, no timeout, no sanitizer report.
    medium:  repro exit code equals the dataset `exit_code`, no timeout, no sanitizer.
    generous: any non-timeout repro exit with no sanitizer report.
    """
    if "SECB_FAIL=patch" in logs:
        return False, "patch did not apply"
    if "SECB_FAIL=build" in logs:
        return False, "compilation failed after patch"
    if "SECB_REACHED_REPRO=1" not in logs:
        return False, "did not reach repro step"

    report = extract_sanitizer_report(logs)
    if report is not None:
        return False, "sanitizer still triggers after patch"

    m = re.search(r"SECB_REPRO_EXIT=(-?\d+)", logs)
    repro_exit = int(m.group(1)) if m else None
    if repro_exit is None:
        return False, "no repro exit code captured"
    if repro_exit in _TIMEOUT_EXIT_CODES:
        return False, "repro timed out"

    if mode == "strict":
        ok = repro_exit == 0
    elif mode == "medium":
        ok = repro_exit == int(gold.get("exit_code", 0))
    else:  # generous
        ok = True
    return (ok, "resolved" if ok else f"repro exit {repro_exit} not accepted in {mode} mode")


def verify_patch(
    instance_id: str, diff: str, gold: dict, *, mode: str = "strict"
) -> tuple[bool, str, str]:
    """Run `diff` against a fresh `:patch` container and return (success, reason, logs)."""
    if not diff or not diff.strip():
        return False, "no patch produced", ""

    image = ensure_image(instance_id, "patch")
    work_dir = gold.get("work_dir") or "/src"
    client = _client()

    with tempfile.TemporaryDirectory(prefix="secbench-pred-") as tmp:
        (Path(tmp) / "model_patch.diff").write_text(diff)
        container = client.containers.run(
            image,
            command=["bash", "-lc", _PATCH_SCRIPT],
            working_dir=work_dir,
            volumes={tmp: {"bind": _PRED_MOUNT, "mode": "ro"}},
            detach=True,
            **_RUN_KWARGS,
        )
        try:
            try:
                container.wait(timeout=_VERIFY_WAIT_SECONDS)
            except Exception:
                with contextlib.suppress(Exception):
                    container.kill()
            logs = container.logs().decode("utf-8", "replace")
        finally:
            with contextlib.suppress(Exception):
                container.remove(force=True)

    success, reason = _interpret_patch(mode, logs, gold)
    return success, reason, logs
