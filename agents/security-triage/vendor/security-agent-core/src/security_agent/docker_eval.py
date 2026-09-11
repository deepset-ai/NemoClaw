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
import codecs
import contextlib
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Iterator, Optional

# Docker Hub prefix for the prebuilt per-instance eval images (from SEC-bench's
# build_eval_instances.py: SECB_IMAGE_PREFIX). The full ref is
# f"{PREFIX}.{instance_id}:{kind}" where kind is "patch" or "poc".
SECB_IMAGE_PREFIX = "hwiwonlee/secb.eval.x86_64"

# Images whose containers this module may start and therefore may reap: SEC-bench's per-instance
# eval images and the generic tool sandbox (security_agent.sandbox.SANDBOX_IMAGE, spelled out here
# to keep this module import-light).
OWNED_IMAGE_PREFIXES = (SECB_IMAGE_PREFIX, "secagent-sandbox")

# `secb repro` is wrapped in `timeout 10`; these are the shell exit codes a timeout yields.
_TIMEOUT_EXIT_CODES = (124, 137)

# Every container we start is stamped with the PID that started it, so a later run can tell its own
# leftovers (and those of a dead run) apart from containers a *concurrent* run is actively using.
OWNER_PID_LABEL = "security-agent.owner-pid"

# Container resource limits. SEC-bench's config.example.toml uses 8g, but a run at
# max_concurrent_tasks=4 reserves 4x this for agent containers *plus* the verification containers
# score() starts in its own pass. At 8g that exceeded the host's 31G and the optimizer was
# OOM-killed 11.5h into a run; 6g keeps the worst case inside physical memory.
_RUN_KWARGS = {
    "mem_limit": "6g",
    "network_mode": "host",
    "tty": False,
}

# How long to wait for the whole verify script (build can be slow); mirrors their 600s.
_VERIFY_WAIT_SECONDS = 600


class _StreamingTextCapture:
    """Decode a byte stream while retaining only its head and tail.

    Docker's non-streaming ``exec_run`` materializes the complete command output before a caller
    can truncate it. A noisy reproducer can therefore consume all host memory even when the tool
    eventually returns only 20k characters. This collector applies that limit as bytes arrive.
    """

    def __init__(self, max_chars: int | None) -> None:
        self.max_chars = max_chars
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._unbounded: list[str] | None = [] if not max_chars or max_chars < 0 else None
        self._head_limit = max_chars // 2 if max_chars else 0
        self._tail_limit = max_chars - self._head_limit if max_chars else 0
        self._head = ""
        self._tail = ""
        self._total_chars = 0

    def feed(self, chunk: bytes) -> None:
        self._add(self._decoder.decode(chunk))

    def finish(self) -> str:
        self._add(self._decoder.decode(b"", final=True))
        if self._unbounded is not None:
            return "".join(self._unbounded)
        if self._total_chars <= self.max_chars:
            return self._head + self._tail
        dropped = self._total_chars - len(self._head) - len(self._tail)
        return (
            f"{self._head}\n...(truncated {dropped} chars from the middle; "
            f"re-run narrowed to see them)...\n{self._tail}"
        )

    def _add(self, text: str) -> None:
        if not text:
            return
        self._total_chars += len(text)
        if self._unbounded is not None:
            self._unbounded.append(text)
            return

        if len(self._head) < self._head_limit:
            take = min(self._head_limit - len(self._head), len(text))
            self._head += text[:take]
            text = text[take:]
        if not text or self._tail_limit == 0:
            return
        if len(text) >= self._tail_limit:
            self._tail = text[-self._tail_limit :]
        else:
            self._tail = (self._tail + text)[-self._tail_limit :]


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


def _owner_alive(pid_label: str | None) -> bool:
    """Is the process that started a container still running?"""
    if not pid_label:
        return False  # unlabelled: predates owner stamping, so treat it as abandoned
    try:
        os.kill(int(pid_label), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except ValueError:
        return False  # unparseable label
    return True


def remove_orphaned_containers() -> list[str]:
    """Remove SEC-bench containers left behind by runs that are no longer alive.

    Containers are only cleaned up when a run exits normally, so every crash or kill strands its
    containers holding their full `mem_limit` reservation. Enough of those and the next run is
    itself OOM-killed — which is exactly how one ACE run died 11.5 hours in. Call this before
    starting any optimization.

    Containers whose owning process is still running are left alone, so this is safe to call while
    another optimizer is mid-run.

    :returns: The names of the containers that were removed.
    """
    removed: list[str] = []
    try:
        containers = _client().containers.list(all=True)
    except Exception as exc:  # noqa: BLE001 - no daemon is a caller problem, not a cleanup failure
        print(f"Could not list containers for cleanup: {type(exc).__name__}: {exc}")
        return removed

    for container in containers:
        image = ((container.attrs or {}).get("Config") or {}).get("Image", "")
        if not image.startswith(OWNED_IMAGE_PREFIXES):
            continue
        if _owner_alive((container.labels or {}).get(OWNER_PID_LABEL)):
            continue
        try:
            container.remove(force=True)
        except Exception as exc:  # noqa: BLE001 - a container that vanished mid-sweep is fine
            print(f"Could not remove {container.name}: {type(exc).__name__}: {exc}")
            continue
        removed.append(container.name)
    return removed


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


def bound_output(text: str, max_output_chars: int | None) -> str:
    """Apply the streaming capture's head/tail bound to text that is already complete.

    For backends whose provider hands back the whole output at once, this cannot save the memory
    that `ContainerHandle.exec` saves — it exists so every backend truncates identically, with the
    same "dropped from the middle" marker, rather than each inventing its own.
    """
    capture = _StreamingTextCapture(max_output_chars)
    capture.feed(text.encode("utf-8"))
    return capture.finish()


class ShellContainer:
    """Read, write and diff, expressed purely in terms of `exec`.

    Subclasses supply `exec` for one container backend: `ContainerHandle` below (the `docker`
    SDK) and `gym_sandbox.SandboxContainer` (`nemo_gym.sandbox`). Everything else here is just
    shell commands, so it is written once and cannot drift between the two.
    """

    work_dir: str
    default_timeout: int = 120

    def _wrap(self, command: str, timeout: Optional[int]) -> tuple[str, int]:
        """The command as it is actually run, plus its budget in seconds.

        Every call goes through `timeout N bash -lc ...`: the `timeout` so a hung command cannot
        stall the whole task, and the *login* shell because the images put `secb` on PATH from
        `/etc/profile.d`. Both backends use this, so neither can quietly lose either property.
        """
        secs = int(timeout if timeout is not None else self.default_timeout)
        # `-k 5`: a command that ignores TERM (or a grandchild that outlives its shell) is KILLed
        # five seconds later, so a wedged process cannot hold the slot until the container dies.
        return f"timeout -k 5 {secs} bash -lc {shlex.quote(command)}", secs

    def exec(
        self,
        command: str,
        timeout: Optional[int] = None,
        max_output_chars: int | None = None,
    ) -> tuple[int, str]:
        """Run a bash command in the container at `work_dir`; return (exit_code, output).

        `max_output_chars` bounds the captured output to a head and a tail; None keeps all of it.
        It is part of the backend contract rather than a caller-side courtesy because only the
        backend sees the output arrive, and that is the only point where a huge log can be
        discarded instead of buffered.
        """
        raise NotImplementedError

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


class ContainerHandle(ShellContainer):
    """A running container the SEC-bench agent operates in, backed by the `docker` SDK."""

    def __init__(self, container, work_dir: str, default_timeout: int = 120) -> None:
        self._c = container
        self.work_dir = work_dir
        self.default_timeout = default_timeout

    def exec(
        self,
        command: str,
        timeout: Optional[int] = None,
        max_output_chars: int | None = None,
    ) -> tuple[int, str]:
        """Stream the output so a noisy command cannot materialize in full before truncation.

        `exec_run` returns only once it has buffered everything, which is how a runaway reproducer
        exhausts host memory even when the tool ultimately returns 20k characters. Driving
        `exec_create`/`exec_start` directly lets `_StreamingTextCapture` apply the bound as bytes
        arrive. The command still goes through `_wrap`, so the login shell and the in-container
        `timeout` stay identical to the sandbox backend by construction.
        """
        wrapped, _secs = self._wrap(command, timeout)
        api = self._c.client.api
        created = api.exec_create(
            self._c.id,
            ["bash", "-lc", wrapped],
            stdout=True,
            stderr=True,
            stdin=False,
            tty=False,
            workdir=self.work_dir,
        )
        exec_id = created["Id"]
        capture = _StreamingTextCapture(max_output_chars)
        for chunk in api.exec_start(exec_id, stream=True, demux=False):
            capture.feed(chunk.encode() if isinstance(chunk, str) else chunk)
        return api.exec_inspect(exec_id)["ExitCode"], capture.finish()


def run_kwargs_with_labels(
    run_kwargs: Optional[dict], labels: Optional[dict[str, str]] = None
) -> dict:
    """`docker run` kwargs with the owner-pid label merged in (plus any caller labels).

    Every container we start is stamped with the starting PID so `remove_orphaned_containers`
    can reap the leftovers of a dead run. The stamp goes in here, not in the kwargs dicts, so a
    caller passing its own `labels` cannot accidentally drop it — and so the dict never has to
    carry a `labels` key that would collide with an explicit `labels=` argument.
    """
    kwargs = dict(_RUN_KWARGS if run_kwargs is None else run_kwargs)
    merged = {**(kwargs.pop("labels", None) or {}), OWNER_PID_LABEL: str(os.getpid()), **(labels or {})}
    kwargs["labels"] = merged
    return kwargs


@contextlib.contextmanager
def start_container(
    image: str,
    work_dir: str,
    labels: Optional[dict[str, str]] = None,
    run_kwargs: Optional[dict] = None,
) -> Iterator[ContainerHandle]:
    """Start `image` as a long-lived container and yield a handle; always removed on exit.

    `labels` are attached to the container. They matter when the *owning process* can die without
    running this contextmanager's cleanup — a long-lived server, say — because they are the only
    way an outside reaper can tell our containers apart and identify who owned them.

    `run_kwargs` are the `docker run` options; the default is SEC-bench's `_RUN_KWARGS`. The
    generic sandbox (`security_agent.sandbox`) passes its own, much tighter set.
    """
    client = _client()
    container = client.containers.run(
        image,
        command=["sleep", "infinity"],
        working_dir=work_dir,
        detach=True,
        **run_kwargs_with_labels(run_kwargs, labels),
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
            **run_kwargs_with_labels(None),
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
