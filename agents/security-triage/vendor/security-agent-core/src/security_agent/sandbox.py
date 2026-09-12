"""A per-task, offline, non-root Docker sandbox for the agent's code-execution tools.

`container_tools` gives an agent `run_bash`, `run_python`, `read_file`, `write_file` … as
serializable components that act on whatever container is bound via `use_container`. SEC-bench
binds the CVE's own image. This module supplies the container for every other benchmark: a
generic C/C++ toolbox image (`docker/sandbox/Dockerfile`), started fresh per task with no network,
a read-only rootfs, a non-root user, tmpfs scratch space and resource limits.

Two things live here so no benchmark has to re-invent them:

- `start_sandbox()` — the container, as a `docker_eval.ContainerHandle`; same `ShellContainer`
  contract as SEC-bench's containers, so the tool components work unchanged.
- `run_agent_in_containers()` — the per-task lifecycle (start container → build agent → bind →
  run → collect), extracted from `SecBench.run` so SEC-bench and PrimeVul share one implementation.

`docker` is imported lazily (as in `docker_eval`), so importing this module never needs a daemon.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Optional, TypeVar

from security_agent import docker_eval, paths
from security_agent.trace_messages import to_openai_messages

if TYPE_CHECKING:  # `benchmarks` imports this module; keep the runtime dependency one-way
    from security_agent.benchmarks.base import Task, TaskResult

SANDBOX_IMAGE = "secagent-sandbox:latest"
SANDBOX_WORK_DIR = "/work"
SANDBOX_DOCKERFILE_DIR = paths.PROJECT_ROOT / "docker" / "sandbox"
SANDBOX_UID = 1000

# Tools whose component lives here act on the bound container, so a config that carries one of
# them needs a sandbox. Everything else (KB lookups, pure-Python components) runs without one.
SANDBOX_TOOL_MODULES = ("security_agent.container_tools.",)

# `docker run` options for one sandbox. Deliberately NOT docker_eval._RUN_KWARGS, which is tuned for
# SEC-bench's images (host networking so `secb` can reach its services, 6g for a build).
#   network none      no egress even if the process-level offline env vars are missing
#   read_only rootfs  only the two tmpfs mounts are writable, and they die with the container
#   user 1000         never root inside; cap_drop ALL + no-new-privileges closes the usual escalations
#   limits            2 GiB, 256 pids, 2 CPUs per task — eight concurrent tasks stay under 16 GiB
SANDBOX_RUN_KWARGS: dict[str, Any] = {
    "network_mode": "none",
    "read_only": True,
    "tmpfs": {
        SANDBOX_WORK_DIR: f"rw,exec,nosuid,size=512m,uid={SANDBOX_UID},gid={SANDBOX_UID}",
        "/tmp": f"rw,exec,nosuid,size=256m,uid={SANDBOX_UID},gid={SANDBOX_UID}",
    },
    "user": f"{SANDBOX_UID}:{SANDBOX_UID}",
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges"],
    "mem_limit": "2g",
    "memswap_limit": "2g",
    "pids_limit": 256,
    "nano_cpus": 2_000_000_000,
    "tty": False,
}

_T = TypeVar("_T")
_R = TypeVar("_R")


# --------------------------------------------------------------------------- #
# Image and container
# --------------------------------------------------------------------------- #
def ensure_sandbox_image() -> str:
    """Return the sandbox image ref; raise if it has not been built.

    Never builds on demand: a benchmark run should not silently spend minutes (and network) on
    `docker build`, and the airgapped runs have no network to build with anyway.
    """
    from docker.errors import ImageNotFound

    try:
        docker_eval._client().images.get(SANDBOX_IMAGE)
    except ImageNotFound as exc:
        raise RuntimeError(
            f"Sandbox image {SANDBOX_IMAGE} not found; build it once with `secagent sandbox build`."
        ) from exc
    return SANDBOX_IMAGE


def build_sandbox_image(no_cache: bool = False) -> str:
    """`docker build` the sandbox image from `docker/sandbox/`; returns the image id."""
    image, _logs = docker_eval._client().images.build(
        path=str(SANDBOX_DOCKERFILE_DIR), tag=SANDBOX_IMAGE, nocache=no_cache, rm=True
    )
    return image.id


def start_sandbox(labels: Optional[dict[str, str]] = None) -> ContextManager[docker_eval.ContainerHandle]:
    """Start one sandbox container and yield its handle; removed on exit (see `start_container`)."""
    return docker_eval.start_container(
        SANDBOX_IMAGE, SANDBOX_WORK_DIR, labels, run_kwargs=SANDBOX_RUN_KWARGS
    )


def needs_sandbox(config: dict) -> bool:
    """Does this serialized agent config carry a tool that acts on a bound container?"""
    tools = (config.get("init_parameters") or {}).get("tools") or []
    for tool in tools:
        component = ((tool.get("data") or {}).get("component") or {}) if isinstance(tool, dict) else {}
        component_type = str(component.get("type") or "")
        if component_type.startswith(SANDBOX_TOOL_MODULES):
            return True
    return False


# --------------------------------------------------------------------------- #
# The per-task agent-in-a-container lifecycle (shared by SEC-bench and PrimeVul)
# --------------------------------------------------------------------------- #
def serialize_messages(messages: object) -> list[dict[str, Any]]:
    """Snapshot Haystack messages in the same readable format used by trace reconstruction."""
    if not isinstance(messages, list):
        return []
    raw = [message.to_dict() if hasattr(message, "to_dict") else message for message in messages]
    return to_openai_messages(raw)


class TrajectoryCollector:
    """Keep the latest complete message state even when ``Agent.run`` raises."""

    allowed_hook_points = ("before_llm", "after_tool", "after_run")

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def run(self, state) -> None:
        self.messages = serialize_messages(state.data.get("messages", []))

    def attach(self, agent: object) -> None:
        hooks = getattr(agent, "hooks", None)
        if not isinstance(hooks, dict):
            return
        for point in self.allowed_hook_points:
            hooks.setdefault(point, []).append(self)


def bounded_thread_map(
    function: Callable[[_T], _R], items: list[_T], max_concurrent_tasks: int
) -> list[_R]:
    """Map in input order while limiting the number of concurrently running task workers."""
    if max_concurrent_tasks < 1:
        raise ValueError("max_concurrent_tasks must be greater than or equal to 1.")
    if len(items) < 2 or max_concurrent_tasks == 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(
        max_workers=min(max_concurrent_tasks, len(items)), thread_name_prefix="sandbox-task"
    ) as executor:
        return list(executor.map(function, items))


def run_agent_in_containers(
    config: dict,
    tasks: list[Task],
    settings,
    *,
    start_container: Callable[["Task"], ContextManager[Any]],
    prepare: Callable[[Any, "Task"], str],
    answer: Callable[[Any, Optional[dict]], str],
    final_answer_prompt: Optional[str] = None,
) -> list["TaskResult"]:
    """Run the serialized agent `config` once per task, each inside its own container.

    Per task: `start_container(task)` yields a handle; `prepare(handle, task)` stages whatever the
    task needs in the container and returns the user message; the agent runs in-process with the
    handle bound (`use_container`), so the `container_tools` components act on it; `answer(handle,
    out)` turns the agent output (`None` if it raised) into `TaskResult.answer`. Nothing raises:
    Docker failures and agent exceptions become `TaskResult.error`, and a partial trajectory is
    kept via the hook collector. Tasks run `settings.max_concurrent_tasks`-wide; the ContextVar
    binding is per thread, and it is entered inside the worker, so concurrent tasks never see each
    other's container.

    The agent is built in-process rather than in the shared subprocess runner because a live
    container handle cannot cross a serialized-config boundary; the container is the isolation
    boundary for model-written code.

    `final_answer_prompt`, when given, is the budget-exhaustion fallback for answer-shaped
    benchmarks: if the agent stops because `max_agent_steps` ran out while it was still calling
    tools, its last message has no text and the task would score as a parse error for a
    *harness* reason. One extra, tool-free generator call with this prompt appended asks for the
    final answer from what the agent has gathered. It fires only on exhaustion, so an agent that
    answers within budget is untouched.
    """
    from haystack.dataclasses import ChatMessage

    from security_agent.benchmarks.base import TaskResult
    from security_agent.container_tools import use_container
    from security_agent.optimize import validate

    def run_task(task: "Task") -> TaskResult:
        started = time.monotonic()
        result_answer, steps, error, out = "", None, None, None
        trajectory: list[dict[str, Any]] = []
        try:
            with start_container(task) as handle:
                agent = validate.load_agent(config)
                collector = TrajectoryCollector()
                collector.attach(agent)
                agent.warm_up()
                message = prepare(handle, task)
                with use_container(handle):
                    try:
                        out = agent.run(messages=[ChatMessage.from_user(message)])
                        steps = out.get("step_count")
                        if final_answer_prompt and not last_message_text(handle, out):
                            out = force_final_answer(agent, out, final_answer_prompt)
                        trajectory = serialize_messages(out.get("messages"))
                    except Exception as e:  # noqa: BLE001 - a task failure must not abort the run
                        error = f"{type(e).__name__}: {e}"
                        trajectory = collector.messages
                result_answer = answer(handle, out)
        except Exception as e:  # noqa: BLE001 - Docker/image failures become a failed task
            error = f"{type(e).__name__}: {e}"
        return TaskResult(
            task_id=task.id,
            answer=result_answer or "",
            error=error,
            steps=steps,
            seconds=round(time.monotonic() - started, 2),
            trajectory=trajectory,
        )

    return bounded_thread_map(run_task, tasks, int(getattr(settings, "max_concurrent_tasks", 1) or 1))


def last_message_text(_handle: Any, out: Optional[dict]) -> str:
    """`answer` callback for classification benchmarks: the agent's final text."""
    if not out:
        return ""
    last = out.get("last_message")
    return (getattr(last, "text", None) if last is not None else "") or ""


# The generic nudge. Deliberately says nothing about any benchmark's output format: the agent's own
# system prompt already states the contract, and this only reminds it that the budget is gone.
FINAL_ANSWER_PROMPT = (
    "Your step budget is exhausted and no further tool calls will be executed. Give your final "
    "answer now, in exactly the format your instructions require, using what you have gathered."
)


def force_final_answer(agent: Any, out: dict, prompt: str) -> dict:
    """One tool-free generator call that turns an exhausted agent run into an answered one.

    Appends the nudge as a user message to the run's messages, calls the agent's chat generator
    directly with no tools (so the model cannot spend the turn on another call), and returns an
    `out` dict with `messages`/`last_message` extended. A generator failure leaves `out` as it was
    — the task then scores as it would have without the fallback.
    """
    from haystack.dataclasses import ChatMessage

    messages = list(out.get("messages") or [])
    nudge = ChatMessage.from_user(prompt)
    try:
        replies = agent.chat_generator.run(messages=[*messages, nudge], tools=None).get("replies") or []
    except Exception:  # noqa: BLE001 - the fallback must never turn a scored task into a crash
        return out
    if not replies:
        return out
    reply = replies[0]
    if not (reply.text or "").strip():
        # A reasoning model may spend the whole reply thinking and leave the text empty; the verdict is
        # often complete inside the reasoning. Salvage it there before giving up on the task.
        salvaged = _json_from_reasoning(reply)
        if salvaged is None:
            try:
                retry = agent.chat_generator.run(
                    messages=[*messages, nudge, ChatMessage.from_user("Reply with the JSON object only, no reasoning.")], tools=None
                ).get("replies") or []
            except Exception:  # noqa: BLE001
                retry = []
            if retry and (retry[0].text or "").strip():
                reply = retry[0]
            elif retry:
                salvaged = _json_from_reasoning(retry[0])
        if salvaged is not None:
            reply = ChatMessage.from_assistant(salvaged)
    return {**out, "messages": [*messages, nudge, reply], "last_message": reply, "forced_final_answer": True}


def _json_from_reasoning(reply: Any) -> Optional[str]:
    """The last JSON object inside a reply's reasoning text, when the visible text is empty."""
    from security_agent.verdict import extract_json_object

    reasoning = getattr(reply, "reasoning", None)
    text = getattr(reasoning, "reasoning_text", None) if reasoning is not None else None
    return extract_json_object(text or "") if text else None


# --------------------------------------------------------------------------- #
# Self-test (`secagent sandbox check`)
# --------------------------------------------------------------------------- #
def self_test() -> list[tuple[str, bool, str]]:
    """Start one sandbox and probe the properties the design relies on. Returns (name, ok, detail)."""
    from security_agent.container_tools import DockerShell, RunPython, use_container

    checks: list[tuple[str, bool, str]] = []
    ensure_sandbox_image()
    with start_sandbox() as handle, use_container(handle):
        shell = DockerShell(timeout=30, max_output_chars=2000)

        def probe(name: str, command: str, ok: Callable[[int, str], bool]) -> None:
            out = shell.run(command)["output"]
            code = int(out.split(")", 1)[0].removeprefix("(exit ")) if out.startswith("(exit ") else -1
            body = out.split("\n", 1)[1] if "\n" in out else ""
            checks.append((name, ok(code, body), body.strip()[:120]))

        probe("non-root user", "id -u", lambda c, b: c == 0 and b.strip() == str(SANDBOX_UID))
        probe("gcc present", "gcc --version | head -1", lambda c, b: c == 0 and "gcc" in b)
        probe("clang present", "clang --version | head -1", lambda c, b: c == 0 and "clang" in b)
        probe("cppcheck present", "cppcheck --version", lambda c, b: c == 0 and "Cppcheck" in b)
        probe("flawfinder present", "flawfinder --version", lambda c, b: c == 0)
        probe("python3 works", "python3 -c 'print(6*7)'", lambda c, b: c == 0 and b.strip() == "42")
        probe("no network (DNS fails)", "getent hosts example.com", lambda c, b: c != 0)
        probe("no network (connect fails)",
              "python3 -c 'import socket;socket.create_connection((\"1.1.1.1\",80),timeout=3)'",
              lambda c, b: c != 0)
        probe("rootfs read-only", "touch /etc/secagent_probe", lambda c, b: c != 0)
        probe("/work writable", f"touch {SANDBOX_WORK_DIR}/probe && rm {SANDBOX_WORK_DIR}/probe",
              lambda c, b: c == 0)
        probe("compile + ASan works",
              "printf 'int main(){int a[2];return a[3];}' > /work/p.c && "
              "gcc -fsanitize=address -g /work/p.c -o /work/p && /work/p; echo rc=$?",
              lambda c, b: "AddressSanitizer" in b)
        out = DockerShell(timeout=1, max_output_chars=200).run("sleep 5")["output"]
        checks.append(("timeout enforced", out.startswith("(exit 124)"), out[:60]))
        out = RunPython(timeout=30).run("```python\nimport json\nprint(json.dumps({'ok': 1}))\n```")["output"]
        checks.append(("run_python (fenced)", '{"ok": 1}' in out, out[:80]))
    return checks
