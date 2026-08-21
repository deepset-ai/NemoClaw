"""Offline unit tests for the SEC-bench container tools (no Docker daemon needed).

The live container is supplied via a ContextVar; here we bind a fake handle so the tool
components run without Docker, and we check they serialize (the optimizer round-trips them).
"""

import pytest

from security_agent.container_tools import (
    _NO_GDB_SENTINEL,
    CodeNavigator,
    DockerDebugger,
    DockerEditFile,
    DockerReadFile,
    DockerShell,
    active_container,
    use_container,
)


class FakeHandle:
    def __init__(self, files=None):
        self.files = dict(files or {})
        self.writes = []

    def exec(self, command, timeout=None):
        return (0, f"ran: {command}")

    def read_file(self, path):
        return self.files.get(path, f"contents of {path}")

    def write_file(self, path, content):
        self.files[path] = content
        self.writes.append((path, content))
        return "ok"


class GdbHandle:
    """Captures the shell command debug_crash runs and returns a canned gdb transcript."""

    def __init__(self, output=""):
        self.output = output
        self.last_command = None
        self.last_timeout = None

    def exec(self, command, timeout=None):
        self.last_command = command
        self.last_timeout = timeout
        return (0, self.output)


def test_shell_tool_uses_active_container():
    with use_container(FakeHandle()):
        out = DockerShell().run("grep -rn foo .")
    assert out["output"] == "(exit 0)\nran: grep -rn foo ."


def test_read_tool_uses_active_container():
    with use_container(FakeHandle()):
        assert DockerReadFile().run("src/x.c")["content"] == "contents of src/x.c"


def test_read_tool_line_range_is_verbatim():
    handle = FakeHandle({"f.c": "l1\nl2\nl3\nl4\nl5\n"})
    with use_container(handle):
        # offset is 1-based; limit is a line count; output has no line-number prefixes so it can
        # be pasted straight into edit_file's old_str.
        assert DockerReadFile().run("f.c", offset=2, limit=2)["content"] == "l2\nl3\n"
        assert DockerReadFile().run("f.c", offset=4)["content"] == "l4\nl5\n"
        assert DockerReadFile().run("f.c")["content"] == "l1\nl2\nl3\nl4\nl5\n"


def test_parse_line_spec_coerces_loose_inputs():
    from security_agent.container_tools import _parse_line_spec

    assert _parse_line_spec(None, None) == (None, None)
    assert _parse_line_spec(5, 3) == (5, 3)
    assert _parse_line_spec("5", "3") == (5, 3)  # numeric strings
    assert _parse_line_spec("255,285", None) == (255, 31)  # range packed into offset -> (start, span)
    assert _parse_line_spec("255-285", None) == (255, 31)
    assert _parse_line_spec("255:285", None) == (255, 31)
    assert _parse_line_spec("255,285", "10") == (255, 10)  # explicit limit wins over the range span
    assert _parse_line_spec("garbage", None) == (None, None)


def test_read_tool_accepts_string_and_range_offset():
    handle = FakeHandle({"f.c": "l1\nl2\nl3\nl4\nl5\n"})
    with use_container(handle):
        assert DockerReadFile().run("f.c", offset="2", limit="2")["content"] == "l2\nl3\n"
        # A range packed into offset (the shape the model emitted) reads that line span.
        assert DockerReadFile().run("f.c", offset="2,3")["content"] == "l2\nl3\n"
        assert DockerReadFile().run("f.c", offset="4-5")["content"] == "l4\nl5\n"


def test_read_tool_via_component_tool_accepts_string_range_offset():
    # Reproduces the exact failure from the campaign log: the model calls read_file with
    # offset='255,285', which used to raise a pydantic 'int_parsing' error at the tool boundary.
    from haystack.tools.component_tool import ComponentTool

    tool = ComponentTool(component=DockerReadFile(), name="read_file", description="read a file")
    handle = FakeHandle({"f.c": "l1\nl2\nl3\nl4\nl5\n"})
    with use_container(handle):
        result = tool.invoke(path="f.c", offset="2,3")  # no ValidationError
    content = result["content"] if isinstance(result, dict) else str(result)
    assert content == "l2\nl3\n"


def test_read_tool_range_surfaces_read_error():
    class Broken(FakeHandle):
        def read_file(self, path):
            return "(error: could not read f.c: exit 1)\n"

    with use_container(Broken()):
        assert DockerReadFile().run("f.c", offset=2, limit=1)["content"].startswith("(error:")


def test_edit_tool_replaces_unique_snippet():
    handle = FakeHandle({"src/x.c": "int a = 1;\nint b = 2;\n"})
    with use_container(handle):
        out = DockerEditFile().run("src/x.c", "int b = 2;", "int b = 3;")
    assert out["result"] == "ok: replaced 1 occurrence in src/x.c"
    assert handle.files["src/x.c"] == "int a = 1;\nint b = 3;\n"


def test_edit_tool_reports_missing_snippet():
    handle = FakeHandle({"src/x.c": "int a = 1;\n"})
    with use_container(handle):
        out = DockerEditFile().run("src/x.c", "not present", "x")
    assert "not found" in out["result"]
    assert handle.writes == []  # nothing written on a failed match


def test_edit_tool_reports_ambiguous_snippet():
    handle = FakeHandle({"src/x.c": "dup\ndup\n"})
    with use_container(handle):
        out = DockerEditFile().run("src/x.c", "dup", "x")
    assert "2 times" in out["result"]
    assert handle.writes == []


def test_edit_tool_surfaces_read_error():
    class Broken(FakeHandle):
        def read_file(self, path):
            return "(error: could not read src/x.c: exit 1)\n"

    handle = Broken()
    with use_container(handle):
        out = DockerEditFile().run("src/x.c", "a", "b")
    assert out["result"].startswith("(error:")
    assert handle.writes == []


def test_tools_require_active_container():
    with pytest.raises(RuntimeError):
        DockerShell().run("ls")
    with pytest.raises(RuntimeError):
        active_container()


def test_context_resets_after_block():
    with use_container(FakeHandle()):
        pass
    with pytest.raises(RuntimeError):
        active_container()


def test_dockershell_serialization_round_trip():
    comp = DockerShell(timeout=45, max_output_chars=1234)
    restored = DockerShell.from_dict(comp.to_dict())
    assert restored.timeout == 45
    assert restored.max_output_chars == 1234


def test_dockershell_passes_short_output_through():
    with use_container(FakeHandle()):  # FakeHandle echoes "ran: <command>" (well under the cap)
        out = DockerShell().run("echo hi")["output"]
    assert out == "(exit 0)\nran: echo hi"
    assert "truncated" not in out


def test_dockershell_truncates_long_output_keeping_head_and_tail():
    class Flood:
        def exec(self, command, timeout=None):
            return (0, "START" + ("x" * 5000) + "END")

    with use_container(Flood()):
        out = DockerShell(max_output_chars=200).run("dump")["output"]
    assert out.startswith("(exit 0)\nSTART")  # head kept (exit line always intact)
    assert out.rstrip().endswith("END")  # tail kept
    assert "truncated" in out and "from the middle" in out
    assert len(out) < 600  # bounded: ~200 chars of body + notice, not 5000+


# --------------------------------------------------------------------------- #
# Debugger tool
# --------------------------------------------------------------------------- #
_CRASH_TRANSCRIPT = """Reading symbols from ./build/tool...
[Thread debugging using libthread_db enabled]
Program received signal SIGSEGV, Segmentation fault.
0x0000555555 in parse (buf=0x0, len=8) at src/parse.c:42
42	    return buf[len];

===== BACKTRACE =====
#0  parse (buf=0x0, len=8) at src/parse.c:42
#1  main () at src/main.c:10

===== LOCALS =====
No locals.

===== ARGS =====
buf = 0x0
len = 8
"""


def test_debugger_builds_gdb_invocation_with_missing_guard():
    handle = GdbHandle(output="Program received signal SIGSEGV")
    with use_container(handle):
        DockerDebugger().run("./build/tool /testcase/poc")
    cmd = handle.last_command
    # Guards on gdb's presence, runs it in batch mode, wires the capture script and the target argv.
    assert "command -v gdb" in cmd
    assert "gdb -q -batch" in cmd
    assert "-ex run" in cmd
    assert "-ex backtrace" in cmd
    assert "--args ./build/tool /testcase/poc" in cmd


def test_debugger_uses_its_own_timeout():
    handle = GdbHandle()
    with use_container(handle):
        DockerDebugger(timeout=90).run("./build/tool poc")
    assert handle.last_timeout == 90


def test_debugger_appends_custom_gdb_commands_after_defaults():
    handle = GdbHandle()
    with use_container(handle):
        DockerDebugger().run("./build/tool poc", gdb_commands=["p somevar", "info registers"])
    cmd = handle.last_command
    assert "-ex 'p somevar'" in cmd
    assert "-ex 'info registers'" in cmd
    # Custom commands run after the built-in capture (so they act at the crash frame).
    assert cmd.index("-ex 'info args'") < cmd.index("-ex 'p somevar'")


def test_debugger_translates_leading_env_assignments():
    handle = GdbHandle()
    with use_container(handle):
        DockerDebugger().run("UBSAN_OPTIONS=abort_on_error=1 ./build/tool /testcase/poc")
    cmd = handle.last_command
    # The assignment becomes a gdb `set environment`, applied before `run`, and is not argv[0].
    assert "-ex 'set environment UBSAN_OPTIONS=abort_on_error=1'" in cmd
    assert "--args ./build/tool /testcase/poc" in cmd
    assert cmd.index("set environment") < cmd.index("-ex run")


def test_debugger_reports_missing_gdb():
    with use_container(GdbHandle(output=_NO_GDB_SENTINEL)):
        report = DockerDebugger().run("./build/tool poc")["report"]
    assert report.startswith("(error: gdb is not installed")


def test_debugger_rejects_empty_command():
    with use_container(GdbHandle()):
        report = DockerDebugger().run("   ")["report"]
    assert report.startswith("(error: empty command")


def test_debugger_formats_crash_report():
    with use_container(GdbHandle(output=_CRASH_TRANSCRIPT)):
        report = DockerDebugger().run("./build/tool /testcase/poc")["report"]
    # A one-line signal summary is surfaced up top...
    assert report.startswith("[gdb] crashed with SIGSEGV")
    # ...loader/thread chatter is dropped...
    assert "Reading symbols from" not in report
    assert "libthread_db" not in report
    # ...and the useful crash context survives.
    assert "src/parse.c:42" in report
    assert "buf = 0x0" in report


def test_debugger_drops_dwarf_version_noise():
    transcript = (
        "Dwarf Error: DW_FORM_strx1 found in non-DWO CU [in module ./build/tool]\n"
        "Program received signal SIGABRT, Aborted.\n#0  abort () at abort.c:1"
    )
    with use_container(GdbHandle(output=transcript)):
        report = DockerDebugger().run("./build/tool poc")["report"]
    assert "Dwarf Error" not in report
    assert report.startswith("[gdb] crashed with SIGABRT")


def test_debugger_reports_ptrace_failure_as_emulation_hint():
    # The gdb transcript seen when debugging an emulated x86_64 binary on an arm64 host.
    transcript = (
        "Couldn't get registers: Input/output error.\n"
        "Python Exception <class 'gdb.error'> Couldn't get registers: Input/output error.:"
    )
    with use_container(GdbHandle(output=transcript)):
        report = DockerDebugger().run("./build/tool poc")["report"]
    assert report.startswith("[gdb] could not inspect the process (ptrace failed)")
    assert "emulation" in report


def test_debugger_flags_run_that_did_not_crash():
    transcript = "Starting program: /build/tool\n[Inferior 1 (process 7) exited normally]\nNo stack."
    with use_container(GdbHandle(output=transcript)):
        report = DockerDebugger().run("./build/tool poc")["report"]
    assert "without crashing" in report


def test_debugger_truncates_to_tail():
    transcript = "HEAD-NOISE\n" + "x" * 500 + "\nTAIL-SIGNAL src/parse.c:42"
    with use_container(GdbHandle(output=transcript)):
        report = DockerDebugger(max_output_chars=80).run("./build/tool poc")["report"]
    assert "truncated to the last 80 chars" in report
    assert "src/parse.c:42" in report  # the tail is kept
    assert "HEAD-NOISE" not in report  # the head is dropped


def test_debugger_serialization_round_trip():
    comp = DockerDebugger(timeout=77, max_output_chars=1234)
    restored = DockerDebugger.from_dict(comp.to_dict())
    assert restored.timeout == 77
    assert restored.max_output_chars == 1234


# --------------------------------------------------------------------------- #
# Code navigation tool
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mode,flag", [("definition", "-1"), ("callers", "-3"), ("references", "-0")]
)
def test_navigator_maps_mode_to_cscope_flag(mode, flag):
    handle = GdbHandle()  # generic capturing handle (records the shell command)
    with use_container(handle):
        CodeNavigator().run("opj_j2k_decode", mode=mode)
    assert f"cscope -dL {flag} opj_j2k_decode" in handle.last_command
    assert r'grep -nE "\bopj_j2k_decode\b"' in handle.last_command  # fallback carries the symbol


def test_navigator_parses_cscope_output():
    out = "CSCOPE_OK\nsrc/f.c func 42 int func(void) {\nsrc/g.c caller 9 func();"
    with use_container(GdbHandle(output=out)):
        result = CodeNavigator().run("func", mode="callers")["result"]
    assert result.startswith("[cscope] callers of 'func' — 2 hit(s)")
    assert "src/f.c:42  [func]  int func(void) {" in result
    assert "src/g.c:9  [caller]  func();" in result


def test_navigator_parses_grep_fallback_output():
    out = "GREP_FALLBACK\nsrc/f.c:42:int func(void)\nsrc/g.c:9:  func();"
    with use_container(GdbHandle(output=out)):
        result = CodeNavigator().run("func", mode="definition")["result"]
    assert "lexical fallback" in result
    assert "src/f.c:42  int func(void)" in result
    assert "src/g.c:9  func();" in result


def test_navigator_rejects_non_identifier_without_running():
    handle = GdbHandle()
    with use_container(handle):
        result = CodeNavigator().run("foo(); evil")["result"]
    assert result.startswith("(error:") and "identifier" in result
    assert handle.last_command is None  # validated before touching the container


def test_navigator_rejects_unknown_mode_without_running():
    handle = GdbHandle()
    with use_container(handle):
        result = CodeNavigator().run("func", mode="sideways")["result"]
    assert "unknown mode" in result
    assert handle.last_command is None


def test_navigator_reports_zero_hits():
    with use_container(GdbHandle(output="CSCOPE_OK\n")):
        result = CodeNavigator().run("nope", mode="definition")["result"]
    assert "no definition found for 'nope'" in result


def test_navigator_truncates_to_max_results():
    hits = "\n".join(f"src/f.c fn {i} line{i}" for i in range(10))
    with use_container(GdbHandle(output=f"CSCOPE_OK\n{hits}")):
        result = CodeNavigator(max_results=3).run("fn", mode="references")["result"]
    assert "10 hit(s) (showing first 3)" in result
    assert result.count("src/f.c:") == 3


def test_navigator_serialization_round_trip():
    comp = CodeNavigator(timeout=200, max_results=15)
    restored = CodeNavigator.from_dict(comp.to_dict())
    assert restored.timeout == 200
    assert restored.max_results == 15
