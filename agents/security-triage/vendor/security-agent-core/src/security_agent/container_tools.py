"""The SEC-bench agent's tools: serializable Haystack components that act on a Docker container.

This module is CODE only — the three tools (`run_shell`, `read_file`, `edit_file`) are ordinary
serializable components that hold NO live Docker handle, so the agent config round-trips through
YAML and the meta-agent can mutate it like any other config. The live container is supplied
out-of-band per task via a `ContextVar` that `SecBench.run` sets (`use_container`) and the tools
read (`active_container`). This mirrors the meta-agent's own `optimize/meta/session.py` pattern.

The agent's *definition* — system prompt, tool descriptions, step budget, generation kwargs — is
NOT here. It lives in the committed seed YAML (`seeds/secbench.yaml`), the single source of truth;
the tool catalog reconstructs these tools from that seed (`optimize/tool_catalog.get_catalog`).
"""

from __future__ import annotations

import re
import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional, Protocol, Union

from haystack import component, default_from_dict, default_to_dict

# Per-shell-command timeout (seconds), used when a DockerShell is constructed without an explicit
# one. Generous because a clean build inside a single run_shell call (e.g. `secb build`) can take
# minutes; too small and the agent can never verify its fix. The seed YAML stores the tool's actual
# timeout, so this default only applies to a bare DockerShell().
DEFAULT_SHELL_TIMEOUT = 600

# Cap on run_shell output fed back to the model (~5k tokens). A `secb build` or a wide grep can emit
# tens of thousands of lines; uncapped, one call floods the context. We keep the head AND tail (the
# start carries setup/context, the end carries the decisive error/result), dropping the middle.
DEFAULT_SHELL_OUTPUT_CHARS = 20_000

# A crashing input run under gdb rarely needs the long build timeout; keep the debugger snappier so
# a hung target can't eat the whole step budget. The seed YAML stores the tool's actual timeout.
DEFAULT_DEBUG_TIMEOUT = 120

# The navigator's first call may install cscope and index the whole tree (slow under emulation);
# later calls are cheap (cached index). Generous so the one-time build can't time out mid-run.
DEFAULT_NAV_TIMEOUT = 300


# --------------------------------------------------------------------------- #
# The active-container context the tools operate through
# --------------------------------------------------------------------------- #
class _Container(Protocol):
    """The subset of docker_eval.ContainerHandle the tools rely on (duck-typed so this
    module never has to import docker/docker_eval)."""

    def exec(self, command: str, timeout: Optional[int] = None) -> tuple[int, str]: ...
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> str: ...


_ACTIVE: ContextVar[Optional[_Container]] = ContextVar("secbench_active_container", default=None)


@contextmanager
def use_container(handle: _Container) -> Iterator[None]:
    """Bind `handle` as the container the SEC-bench tools act on, for the duration of the block."""
    token = _ACTIVE.set(handle)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_container() -> _Container:
    handle = _ACTIVE.get()
    if handle is None:
        raise RuntimeError(
            "No active container. SEC-bench tools can only run inside SecBench.run "
            "(wrap the agent call in use_container(...))."
        )
    return handle


# --------------------------------------------------------------------------- #
# Tools (serializable components; the live container comes from the ContextVar)
# --------------------------------------------------------------------------- #
def _truncate_middle(text: str, max_chars: int) -> str:
    """Keep the head and tail of `text`, dropping the middle with a notice, when it exceeds
    `max_chars`. Preserves both ends because command output carries context at the start and the
    decisive error/result at the end. `max_chars <= 0` disables truncation."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    head, tail = text[:half], text[-half:]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n...(truncated {dropped} chars from the middle; re-run narrowed to see them)...\n{tail}"


@component
class DockerShell:
    """Run a bash command in the task's container. Backs the `run_shell` tool."""

    def __init__(
        self, timeout: int = DEFAULT_SHELL_TIMEOUT, max_output_chars: int = DEFAULT_SHELL_OUTPUT_CHARS
    ) -> None:
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    @component.output_types(output=str)
    def run(self, command: str) -> dict:
        exit_code, out = active_container().exec(command, timeout=self.timeout)
        out = _truncate_middle(out, self.max_output_chars)
        return {"output": f"(exit {exit_code})\n{out}".strip()}

    def to_dict(self) -> dict:
        return default_to_dict(self, timeout=self.timeout, max_output_chars=self.max_output_chars)

    @classmethod
    def from_dict(cls, data: dict) -> "DockerShell":
        return default_from_dict(cls, data)


_RANGE_RE = re.compile(r"^\s*(\d+)\s*[,\-:]\s*(\d+)\s*$")
_LEADING_INT_RE = re.compile(r"^\s*(-?\d+)")


def _to_int(value: Union[int, str, None]) -> Optional[int]:
    """Best-effort int from a model-supplied value; None if it can't be read as one."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = _LEADING_INT_RE.match(str(value))
    return int(match.group(1)) if match else None


def _parse_line_spec(
    offset: Union[int, str, None], limit: Union[int, str, None]
) -> tuple[Optional[int], Optional[int]]:
    """Normalize loose offset/limit inputs into (offset, limit). A `start,end`/`start-end` range in
    `offset` sets both (limit = the span) unless `limit` was given explicitly."""
    lim = _to_int(limit)
    if isinstance(offset, str):
        match = _RANGE_RE.match(offset)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return start, (lim if lim is not None else max(1, end - start + 1))
    return _to_int(offset), lim


@component
class DockerReadFile:
    """Read a file (optionally a line range) in the task's container. Backs the `read_file` tool.

    Returns the file's exact bytes (no line-number prefixes), so a snippet can be copied verbatim
    into `edit_file`'s `old_str`. `offset`/`limit` let the agent read a slice of a large file
    rather than dumping the whole thing into its context."""

    @component.output_types(content=str)
    def run(
        self,
        path: str,
        offset: Union[int, str, None] = None,
        limit: Union[int, str, None] = None,
    ) -> dict:
        offset, limit = _parse_line_spec(offset, limit)
        content = active_container().read_file(path)
        if content.startswith("(error:") or (offset is None and limit is None):
            return {"content": content}
        start = max(1, offset) if offset else 1
        end = (start - 1 + limit) if limit is not None else None
        lines = content.splitlines(keepends=True)
        return {"content": "".join(lines[start - 1 : end])}

    def to_dict(self) -> dict:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DockerReadFile":
        return default_from_dict(cls, data)


@component
class DockerEditFile:
    """Replace an exact, unique snippet in a file in the task's container. Backs the `edit_file`
    tool — targeted str_replace edits (Anthropic text-editor semantics) rather than full-file
    overwrites, so patches stay minimal and reviewable. Match failures are returned as an
    actionable message (never raised), like the other tools."""

    @component.output_types(result=str)
    def run(self, path: str, old_str: str, new_str: str) -> dict:
        container = active_container()
        content = container.read_file(path)
        if content.startswith("(error:"):  # read_file's failure sentinel
            return {"result": content}
        count = content.count(old_str)
        if count == 0:
            return {"result": (
                f"(error: old_str not found in {path}. read_file the region and copy the exact "
                "text, including indentation and whitespace.)"
            )}
        if count > 1:
            return {"result": (
                f"(error: old_str appears {count} times in {path}; include more surrounding "
                "context so it matches exactly once.)"
            )}
        write_result = container.write_file(path, content.replace(old_str, new_str, 1))
        if write_result != "ok":
            return {"result": write_result}
        return {"result": f"ok: replaced 1 occurrence in {path}"}

    def to_dict(self) -> dict:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DockerEditFile":
        return default_from_dict(cls, data)


# --------------------------------------------------------------------------- #
# The debugger tool (run the crashing input under gdb, return a crash report)
# --------------------------------------------------------------------------- #
# Sentinel echoed by the shell when gdb is absent, so we return an actionable message rather than a
# confusing "command not found" (mirrors the other tools' non-raising failure style).
_NO_GDB_SENTINEL = "__SECAGENT_GDB_MISSING__"

# The fixed gdb batch script: run the target, then dump the crash context. These are `-ex` commands
# executed in order; after `run` stops at the fault, gdb is positioned at the innermost frame, so
# `info locals`/`info args` describe the crash site. The `echo` separators (gdb interprets the `\n`
# escapes) keep the sections easy for the agent to parse. `bt` gives the call chain; `frame` prints
# the faulting source line. Any caller-supplied commands run afterwards, still at the crash frame.
_DEFAULT_GDB_COMMANDS = (
    "set pagination off",
    "set confirm off",
    "set backtrace limit 40",
    "run",
    "echo \\n===== BACKTRACE =====\\n",
    "backtrace",
    "echo \\n===== CRASH FRAME =====\\n",
    "frame",
    "echo \\n===== LOCALS =====\\n",
    "info locals",
    "echo \\n===== ARGS =====\\n",
    "info args",
)

# gdb/loader chatter that carries no signal for root-causing; dropped so the report stays readable.
# "Dwarf Error" appears when the image's gdb is older than the DWARF version clang emitted (e.g. gdb
# 9.2 vs DWARF5's DW_FORM_strx1) — harmless symbol-reading noise, not a crash detail.
_GDB_NOISE_PREFIXES = (
    "Reading symbols from",
    "Using host libthread_db library",
    "[Thread debugging using libthread_db enabled]",
    "warning: File",
    "Dwarf Error:",
)

_SIGNAL_RE = re.compile(r"Program (?:received signal|terminated with signal) (\w+)")
_EXITED_RE = re.compile(r"\[Inferior .*exited (?:normally|with code)")
# gdb can't ptrace the inferior — most often because the target runs under CPU emulation (an
# x86_64 image on an arm64 host), where register/memory access over ptrace is unsupported.
_PTRACE_FAIL_MARKERS = ("Couldn't get registers", "ptrace: Operation not permitted", "ptrace: Input/output error")

# A leading `VAR=value` shell-style assignment. Peeled off the command and turned into gdb
# `set environment` so the agent can, e.g., make a sanitizer abort (UBSAN_OPTIONS=abort_on_error=1)
# — needed for gdb to catch a signal — or set LD_LIBRARY_PATH, without it being mistaken for argv[0].
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _split_env(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split leading `VAR=value` tokens into gdb `set environment` commands and the remaining argv."""
    env_cmds: list[str] = []
    i = 0
    while i < len(argv) and _ENV_ASSIGN_RE.match(argv[i]):
        env_cmds.append(f"set environment {argv[i]}")
        i += 1
    return env_cmds, argv[i:]


def _crash_header(report: str) -> str:
    """A one-line summary at the top of the report so the crash signal is never lost to truncation."""
    if any(marker in report for marker in _PTRACE_FAIL_MARKERS):
        return (
            "[gdb] could not inspect the process (ptrace failed), so no backtrace was captured. "
            "This usually means the target is running under CPU emulation (e.g. an x86_64 image on "
            "an arm64 host); run on a native x86_64 host to debug it."
        )
    signal = _SIGNAL_RE.search(report)
    if signal:
        return f"[gdb] crashed with {signal.group(1)}"
    if _EXITED_RE.search(report):
        return (
            "[gdb] the program ran to completion without crashing — the command may not reproduce "
            "the bug, or the crash is already fixed. Confirm the crashing invocation with secb repro."
        )
    return ""


def _format_gdb_report(raw: str, max_output_chars: int) -> str:
    """Denoise gdb output and cap its length (keeping the tail, where the crash context lives)."""
    body = "\n".join(
        line for line in raw.splitlines() if not line.startswith(_GDB_NOISE_PREFIXES)
    ).strip()
    header = _crash_header(body)
    if len(body) > max_output_chars:
        body = f"...(truncated to the last {max_output_chars} chars)...\n{body[-max_output_chars:]}"
    return f"{header}\n{body}" if header else body


@component
class DockerDebugger:
    """Run a crashing command under gdb in the task's container and return a crash report. Backs the
    `debug_crash` tool.

    The sanitizer report from `secb repro` shows *where* the program stops; this tool adds *why* — a
    backtrace with `file:line` frames plus the local variables and arguments at the crash (the index,
    size, or pointer that is actually wrong). Runs `gdb --batch` with a fixed capture script, then any
    caller-supplied `gdb_commands` (executed at the crash frame). Failures are returned as an
    actionable message (missing gdb, or a run that never crashed) rather than raised, like the other
    tools."""

    def __init__(self, timeout: int = DEFAULT_DEBUG_TIMEOUT, max_output_chars: int = 8000) -> None:
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    @component.output_types(report=str)
    def run(self, command: str, gdb_commands: Optional[list[str]] = None) -> dict:
        env_cmds, argv = _split_env(shlex.split(command))
        if not argv:
            return {"report": (
                "(error: empty command. Pass the exact crashing program invocation, e.g. "
                "'./build/tool /testcase/poc'.)"
            )}
        # env `set environment` commands must precede `run` (which is inside _DEFAULT_GDB_COMMANDS).
        ex_commands = [*env_cmds, *_DEFAULT_GDB_COMMANDS, *(gdb_commands or [])]
        tokens = ["gdb", "-q", "-batch"]
        for cmd in ex_commands:
            tokens += ["-ex", cmd]
        tokens += ["--args", *argv]
        gdb_cmd = " ".join(shlex.quote(tok) for tok in tokens)
        # Probe for gdb first so a missing binary yields our sentinel, not a raw "command not found".
        shell = f"if command -v gdb >/dev/null 2>&1; then {gdb_cmd}; else echo {_NO_GDB_SENTINEL}; fi"
        _, out = active_container().exec(shell, timeout=self.timeout)
        if _NO_GDB_SENTINEL in out:
            return {"report": (
                "(error: gdb is not installed in this container. Read the sanitizer backtrace from "
                "`secb repro` via run_shell instead.)"
            )}
        return {"report": _format_gdb_report(out, self.max_output_chars)}

    def to_dict(self) -> dict:
        return default_to_dict(self, timeout=self.timeout, max_output_chars=self.max_output_chars)

    @classmethod
    def from_dict(cls, data: dict) -> "DockerDebugger":
        return default_from_dict(cls, data)


# --------------------------------------------------------------------------- #
# Semantic code navigation (cscope-backed find-definition / callers / references)
# --------------------------------------------------------------------------- #
# A single C/C++ identifier — the only thing we search for (keeps cscope/grep queries sane and makes
# shell injection a non-issue on top of the shlex quoting).
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

# mode -> cscope search type. cscope emits one uniform line per hit ("file scope line text") for all
# of these, so a single parser handles every mode: -1 global definition, -3 functions calling this
# (callers), -0 every symbol occurrence (references).
_CSCOPE_FLAG = {"definition": "-1", "callers": "-3", "references": "-0"}

# Built once per container under /tmp and reused across calls; rebuilt only when a source file is
# newer than the index (so it stays correct as the agent edits). Runs at the repo root (the tools'
# working dir), so cscope's paths are repo-relative — exactly what read_file/edit_file expect.
_NAV_SCRIPT = r"""
NAV=/tmp/secagent_nav; mkdir -p "$NAV"
have() {{ command -v cscope >/dev/null 2>&1; }}
if ! have && [ ! -f "$NAV/.install_tried" ]; then
  touch "$NAV/.install_tried"; apt-get install -y -q cscope >/dev/null 2>&1 || true
fi
if have; then
  if [ ! -s "$NAV/nav.files" ]; then
    (git ls-files '*.c' '*.h' '*.cc' '*.cpp' '*.cxx' '*.hpp' '*.hh' '*.hxx' '*.c++' 2>/dev/null || true) > "$NAV/nav.files"
    [ -s "$NAV/nav.files" ] || find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.hpp' \) > "$NAV/nav.files" 2>/dev/null
  fi
  if [ ! -f "$NAV/cscope.out" ] || [ -n "$(find $(cat "$NAV/nav.files") -newer "$NAV/cscope.out" 2>/dev/null | head -1)" ]; then
    cscope -bqk -i "$NAV/nav.files" -f "$NAV/cscope.out" >/dev/null 2>&1 || true
  fi
  if [ -f "$NAV/cscope.out" ]; then
    echo CSCOPE_OK
    cscope -dL {flag} {sym} -f "$NAV/cscope.out" 2>/dev/null
    exit 0
  fi
fi
echo GREP_FALLBACK
FILES=$(git ls-files '*.c' '*.h' '*.cc' '*.cpp' '*.cxx' '*.hpp' '*.hh' 2>/dev/null)
grep -nE "\b{sym}\b" $FILES 2>/dev/null | head -200
"""


def _format_nav(out: str, mode: str, symbol: str, max_results: int) -> str:
    """Turn the navigator's raw output into a compact `file:line [scope] text` list with a header."""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return f"(no output from the code navigator for '{symbol}')"
    lexical = lines[0].strip() == "GREP_FALLBACK"
    backend = "grep (lexical fallback; cscope unavailable)" if lexical else "cscope"
    hits = lines[1:] if lines[0].strip() in ("CSCOPE_OK", "GREP_FALLBACK") else lines
    if not hits:
        return (
            f"[{backend}] no {mode} found for '{symbol}'. Try mode='references' to see every use, "
            "check the spelling, or note it may be a macro or inlined."
        )
    truncated = len(hits) > max_results
    formatted = []
    for hit in hits[:max_results]:
        if lexical:
            parts = hit.split(":", 2)  # grep -n over many files: file:line:text
            formatted.append(f"{parts[0]}:{parts[1]}  {parts[2].strip()}" if len(parts) == 3 else hit)
        else:
            parts = hit.split(None, 3)  # cscope: file scope line text
            if len(parts) == 4:
                formatted.append(f"{parts[0]}:{parts[2]}  [{parts[1]}]  {parts[3]}")
            else:
                formatted.append(hit)
    header = f"[{backend}] {mode} of '{symbol}' — {len(hits)} hit(s)"
    if truncated:
        header += f" (showing first {max_results})"
    return header + "\n" + "\n".join(formatted)


@component
class CodeNavigator:
    """Find where a C/C++ symbol is defined or used across the repo. Backs the `find_symbol` tool.

    More precise than raw grep for localizing a vulnerability: jump from a crash-backtrace frame to
    the function's source, or enumerate every caller that might pass the bad input. Backed by a
    cscope index built once over the repo's source (refreshed when files change); if cscope can't be
    installed it falls back to a lexical grep. Non-raising, like the other tools."""

    def __init__(self, timeout: int = DEFAULT_NAV_TIMEOUT, max_results: int = 40) -> None:
        self.timeout = timeout
        self.max_results = max_results

    @component.output_types(result=str)
    def run(self, symbol: str, mode: str = "definition") -> dict:
        symbol = (symbol or "").strip()
        if not _IDENT_RE.match(symbol):
            return {"result": (
                f"(error: '{symbol}' is not a single C/C++ identifier. Pass one function, struct, "
                "typedef, macro, or global name, e.g. 'opj_j2k_decode'.)"
            )}
        flag = _CSCOPE_FLAG.get(mode)
        if flag is None:
            return {"result": (
                f"(error: unknown mode '{mode}'. Use 'definition', 'callers', or 'references'.)"
            )}
        script = _NAV_SCRIPT.format(flag=flag, sym=shlex.quote(symbol))
        _, out = active_container().exec(script, timeout=self.timeout)
        return {"result": _format_nav(out, mode, symbol, self.max_results)}

    def to_dict(self) -> dict:
        return default_to_dict(self, timeout=self.timeout, max_results=self.max_results)

    @classmethod
    def from_dict(cls, data: dict) -> "CodeNavigator":
        return default_from_dict(cls, data)
