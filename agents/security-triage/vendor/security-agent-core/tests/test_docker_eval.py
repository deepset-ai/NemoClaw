"""Offline unit tests for the SEC-bench Docker verifier's pure logic.

Nothing here touches Docker: `extract_sanitizer_report` and `_interpret_patch` are pure
functions over log strings, which is exactly the slice most worth pinning down.
"""

from security_agent import docker_eval as de

_ASAN = """\
SECB_REACHED_REPRO=1
==732128==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000
    #0 njs_vmcode_interpreter src/njs_vmcode.c:802:27
SUMMARY: AddressSanitizer: SEGV src/njs_vmcode.c:802:27 in njs_vmcode_interpreter
==732128==ABORTING
SECB_REPRO_EXIT=1
"""

_CLEAN = "SECB_REACHED_REPRO=1\nrunning testcase... ok\nSECB_REPRO_EXIT=0\n"


def test_image_ref():
    assert de.image_ref("njs.cve-2022-32414") == (
        "hwiwonlee/secb.eval.x86_64.njs.cve-2022-32414:patch"
    )
    assert de.image_ref("njs.cve-2022-32414", "poc").endswith(":poc")


def test_extract_sanitizer_report_block():
    report = de.extract_sanitizer_report(_ASAN)
    assert report is not None
    assert "AddressSanitizer" in report
    assert report.rstrip().endswith("ABORTING")


def test_extract_sanitizer_report_marker_fallback():
    logs = "some output\nSUMMARY: UndefinedBehaviorSanitizer: undefined-behavior x.c:1:2\nmore"
    assert de.extract_sanitizer_report(logs) is not None


def test_extract_sanitizer_report_none_when_clean():
    assert de.extract_sanitizer_report(_CLEAN) is None
    assert de.extract_sanitizer_report("") is None


def test_interpret_patch_strict_resolved():
    ok, reason = de._interpret_patch("strict", _CLEAN, {"exit_code": 0})
    assert ok and reason == "resolved"


def test_interpret_patch_sanitizer_still_fires():
    ok, reason = de._interpret_patch("strict", _ASAN, {"exit_code": 0})
    assert not ok and "sanitizer" in reason


def test_interpret_patch_build_failure():
    ok, reason = de._interpret_patch("strict", "SECB_FAIL=build\n", {})
    assert not ok and "compil" in reason


def test_interpret_patch_apply_failure():
    ok, reason = de._interpret_patch("strict", "SECB_FAIL=patch\n", {})
    assert not ok and "apply" in reason


def test_interpret_patch_timeout():
    logs = "SECB_REACHED_REPRO=1\nSECB_REPRO_EXIT=124\n"
    ok, reason = de._interpret_patch("strict", logs, {"exit_code": 0})
    assert not ok and "timed out" in reason


def test_interpret_patch_medium_uses_gold_exit_code():
    logs = "SECB_REACHED_REPRO=1\nSECB_REPRO_EXIT=3\n"
    assert de._interpret_patch("medium", logs, {"exit_code": 3})[0] is True
    assert de._interpret_patch("strict", logs, {"exit_code": 3})[0] is False


def test_interpret_patch_never_reached_repro():
    ok, reason = de._interpret_patch("strict", "did stuff\n", {})
    assert not ok


class _FakeContainer:
    """Records the shell commands ContainerHandle.exec runs against it."""

    def __init__(self, output=b""):
        self.commands = []
        self._output = output

    def exec_run(self, cmd, workdir=None, demux=False):
        self.commands.append(cmd[-1])  # the `timeout N bash -lc '<command>'` wrapper string

        class _Res:
            exit_code = 0
            output = self._output

        return _Res()


def test_git_diff_uses_tracked_diff_not_add_all():
    # `git add -A` would sweep in build artifacts; git_diff must use a plain tracked-file diff.
    fake = _FakeContainer(output=b"diff --git a/src/x.c b/src/x.c\n")
    handle = de.ContainerHandle(fake, "/src/proj")
    out = handle.git_diff()
    joined = " ".join(fake.commands)
    assert "git diff" in joined
    assert "git add" not in joined
    assert out.startswith("diff --git")
