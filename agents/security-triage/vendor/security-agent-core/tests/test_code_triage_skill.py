"""Contract tests for the code-triage skill (skills/code-triage/scripts).

The stdlib-only tests pin the stage-1 output contract, the fail-open stage-2
behavior, and the classify/prioritize logic re-derived from T3MP3ST's code-ingest.
The tree-sitter/Bandit path is exercised only when those optional deps are present,
matching the skill's own fail-open design.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "skills" / "code-triage" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402
import deepen  # noqa: E402
import sweep  # noqa: E402

HAS_TREE_SITTER = importlib.util.find_spec("tree_sitter_language_pack") is not None
HAS_BANDIT = importlib.util.find_spec("bandit") is not None


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "node_modules").mkdir()

    (tmp_path / "src" / "upload.py").write_text(
        "import subprocess\n"
        "\n"
        '@app.route("/upload", methods=["POST"])\n'
        "def handle_upload(request):\n"
        '    name = request.args.get("file")\n'
        "    subprocess.run([name])\n"
        "\n"
        "def validate_token(token):\n"
        "    return token == SECRET\n"
        "\n"
        "def fetch_remote(url):\n"
        "    import requests\n"
        "    return requests.get(url)\n"
        "\n"
        "def helper(x):\n"
        "    return x + 1\n"
    )
    (tmp_path / "src" / "cmd.c").write_text(
        "int run_it(char *path) {\n    system(path);\n    return 0;\n}\n"
    )
    # excluded by DEFAULT_EXCLUDES
    (tmp_path / "tests" / "test_x.py").write_text("import subprocess\n")
    (tmp_path / "node_modules" / "junk.js").write_text("x.system(1)\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Stage 1 — sweep
# ---------------------------------------------------------------------------

def test_sweep_finds_cross_language_sinks_and_honors_excludes(repo: Path):
    results = {fh.path: fh for fh in sweep.sweep(repo)}
    paths = set(results)

    # test dirs and node_modules are excluded; source files with sinks are flagged.
    assert "src/upload.py" in paths
    assert "src/cmd.c" in paths
    assert not any("test" in p or "node_modules" in p for p in paths)

    py_labels = {h.label for h in results["src/upload.py"].hits}
    assert "subprocess" in py_labels
    assert "requests.*" in py_labels

    c_labels = {h.label for h in results["src/cmd.c"].hits}
    assert "system()" in c_labels  # bare-call guard matches C free call


def test_sweep_bare_call_guard_skips_method_calls(tmp_path: Path):
    # `foo.system(x)` (method call) must NOT match the guarded bare-call `system(`.
    (tmp_path / "a.py").write_text("def f():\n    self.system('x')\n")
    results = sweep.sweep(tmp_path)
    labels = {h.label for fh in results for h in fh.hits}
    assert "system()" not in labels


# ---------------------------------------------------------------------------
# Stage 2 — fail-open path (no tree-sitter): raw-hit candidates
# ---------------------------------------------------------------------------

def test_deepen_warns_when_files_do_not_resolve(repo: Path):
    # A fully-unresolved --files list (e.g. an unsplit shell variable, or typos) must
    # produce a loud warning, not an empty warning-free result that reads as "clean".
    out = deepen.deepen(repo, files=["src/upload.py src/cmd.c"])  # one unsplit token
    assert out["candidates"] == []
    assert any("none of the" in w and "--files" in w for w in out["warnings"])

    # A partially-unresolved list still runs on the good entries, but flags the rest.
    out2 = deepen.deepen(repo, files=["src/upload.py", "does/not/exist.py"])
    assert out2["candidates"]  # upload.py still triaged
    assert any("did not" in w and "resolve" in w for w in out2["warnings"])


def test_deepen_failopen_emits_raw_hit_candidates(repo: Path, monkeypatch):
    # Force the tree-sitter-absent branch regardless of the environment.
    monkeypatch.setattr(
        deepen, "parse_functions",
        lambda path, content: ([], "tree-sitter unavailable (forced); function resolution skipped"),
    )
    out = deepen.deepen(repo)

    assert any("tree-sitter unavailable" in w for w in out["warnings"])
    # duplicate warnings are collapsed
    assert len(out["warnings"]) == len(set(out["warnings"]))

    by_file = {c["file"]: c for c in out["candidates"]}
    assert "src/upload.py" in by_file
    up = by_file["src/upload.py"]
    assert up["function"] is None
    assert up["exposure"] == "attack_surface"
    assert up["finding"]["tool"] == "regex-fallback"


# ---------------------------------------------------------------------------
# Pure classify / prioritize logic (stdlib only)
# ---------------------------------------------------------------------------

def _fn(name="f", params=None, body="", decorators=""):
    return common.Function(
        file="x.py", name=name, params=params or [], body=body,
        line_start=1, line_end=2, decorators=decorators,
    )


def test_classify_precedence():
    pats = common.load_patterns()
    # entry point wins even with a sink in the body
    assert deepen.classify(_fn(name="handle_x", body="os.system(x)"), True, True, pats) == "exposed_externally"
    # security-control name beats a body sink
    assert deepen.classify(_fn(name="validate_input", body="os.system(x)"), False, False, pats) == "security_control"
    # a body sink beats mere reachability
    assert deepen.classify(_fn(name="worker", body="os.system(x)"), False, True, pats) == "attack_surface"
    # reachable but no sink / control / entry
    assert deepen.classify(_fn(name="worker", body="return 1"), False, True, pats) == "exposed_internally"
    # nothing at all
    assert deepen.classify(_fn(name="worker", body="return 1"), False, False, pats) == "neutral"


def test_prioritize_formula_matches_source():
    # exposed_externally(100) + 2 signals(20) + reachable depth0 bonus(30) + ssrf(20)
    assert deepen.prioritize("exposed_externally", True, 0, 2, True) == 170
    # attack_surface(80), unreachable -> no reach bonus, 1 signal(10)
    assert deepen.prioritize("attack_surface", False, -1, 1, False) == 90
    # reachable depth 2 bonus = max(0, 30-10) = 20
    assert deepen.prioritize("exposed_internally", True, 2, 0, False) == 70
    # security_control tier is preserved (review finding #6)
    assert deepen.prioritize("security_control", False, -1, 0, False) == 40


def test_verified_bonus_lifts_ast_confirmed_findings():
    # a Bandit finding adds a severity-scaled bonus so it can outrank a naming guess
    assert deepen.verified_bonus({"tool": "bandit", "severity": "high"}) == 50
    assert deepen.verified_bonus({"tool": "bandit", "severity": "medium"}) == 30
    assert deepen.verified_bonus({"tool": "bandit", "severity": "low"}) == 5
    assert deepen.verified_bonus({"tool": "regex-fallback", "severity": None}) == 0
    assert deepen.verified_bonus(None) == 0
    # a medium Bandit hit on attack_surface (80+30=110) now beats a sink-less entry
    # point (exposed_externally base 100)
    verified = deepen.prioritize("attack_surface", False, -1, 0, False, verified_bonus=30)
    bare_entry = deepen.prioritize("exposed_externally", False, -1, 0, False)
    assert verified > bare_entry


def test_name_only_entry_penalty_ranks_below_decorator_and_sinks():
    # a decorated entry point (no penalty) outranks a bare name-only entry point
    decorated = deepen.prioritize("exposed_externally", False, -1, 0, False)
    name_only = deepen.prioritize(
        "exposed_externally", False, -1, 0, False, entry_penalty=deepen.NAME_ONLY_ENTRY_PENALTY
    )
    assert decorated > name_only
    # and the penalized name-only guess drops below a concrete sink (attack_surface)
    concrete_sink = deepen.prioritize("attack_surface", False, -1, 1, False)
    assert concrete_sink > name_only


def test_http_sink_patterns_require_call_not_annotation():
    # httpx/urllib patterns must fire on a call site, not a type annotation or import,
    # which were the top false positives on a real Python library.
    pats = common.load_patterns()
    labels = {label for label, rx in pats.sink_evidence}
    assert "httpx" in labels and "urllib" in labels

    def matched(body):
        return {label for label, rx in pats.sink_evidence if rx.search(body)}

    # false-positive shapes: no sink label
    assert "httpx" not in matched("def f(resp: httpx.Response) -> httpx.Response: ...")
    assert "httpx" not in matched("import httpx")
    assert "urllib" not in matched("from urllib.parse import urljoin")
    # real call sites: sink label fires
    assert "httpx" in matched("async with httpx.AsyncClient() as c: ...")
    assert "httpx" in matched("r = httpx.get(url)")
    assert "urllib" in matched("urllib.request.urlopen(url)")


def test_sql_execute_sink_matches_any_receiver():
    # the SQL-exec sink must fire on the common `c.execute(...)` / `session.execute(...)`
    # idioms, not only the literal `cursor.execute` — the miss that hid every SQLi
    # file on a real vulnerable app.
    pats = common.load_patterns()

    def matched(body):
        return {label for label, rx in pats.sink_evidence if rx.search(body)}

    assert ".execute()" in matched("c = conn.cursor(); c.execute(sql)")
    assert ".execute()" in matched("cursor.execute(query)")
    assert ".execute()" in matched("session.execute(text(q))")
    assert ".execute()" not in matched("return self.executed_count")  # not a call


def test_socket_sink_requires_a_network_call():
    # socket. must fire on a real network call, not on a string literal or a
    # dataflow framework's pipeline "socket" objects (both were false positives).
    pats = common.load_patterns()

    def matched(body):
        return {label for label, rx in pats.sink_evidence if rx.search(body)}

    # false-positive shapes
    assert "socket" not in matched('log.warning("uses WebSockets (socket.io) sticky")')
    assert "socket" not in matched("for s in self.socket.senders: ...")
    assert "socket" not in matched("out = component.output_sockets[name]")
    # real network calls
    assert "socket" in matched("s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)")
    assert "socket" in matched("c = socket.create_connection((host, port))")
    assert "socket" in matched("addrs = socket.getaddrinfo(host, port)")


def test_attach_finding_picks_highest_severity():
    # a function with a medium AND a high bandit issue must surface the HIGH one,
    # not whichever comes first by line (severity misrepresentation otherwise).
    from pathlib import Path as _P
    fn = _fn(name="do_GET")
    fn.line_start, fn.line_end = 18, 80
    fn.file = "/x/dsvw.py"
    key = str(_P("/x/dsvw.py").resolve())
    bandit_by_file = {key: [
        {"tool": "bandit", "test_id": "B608", "severity": "medium", "line": 24},
        {"tool": "bandit", "test_id": "B602", "severity": "high", "line": 33},
    ]}
    finding = deepen.attach_finding(fn, [], bandit_by_file)
    assert finding["severity"] == "high" and finding["test_id"] == "B602"
    assert finding["n_findings"] == 2  # reader is told there's more than one


def test_ssrf_idor_shape():
    pats = common.load_patterns()
    assert deepen.has_ssrf_idor(_fn(params=["url"], body="requests.get(url)"), pats)
    # generic-verb dispatch: requests.request("GET", url) must count as an outbound call
    assert deepen.has_ssrf_idor(_fn(params=["url"], body='requests.request("GET", url)'), pats)
    assert not deepen.has_ssrf_idor(_fn(params=["n"], body="requests.get(x)"), pats)  # no risky param
    assert not deepen.has_ssrf_idor(_fn(params=["url"], body="return url"), pats)  # no outbound request


def test_name_entry_point_gated_on_web_context():
    pats = common.load_patterns()
    handler = _fn(name="_resolve_handler", body="return self.handlers[ct]")
    decorated = _fn(name="view", decorators="@app.route('/x')", body="...")

    # library (no web context): name-based match suppressed, decorator still fires
    assert deepen.is_entry_point(handler, pats, web_context=False) is False
    assert deepen.is_entry_point(decorated, pats, web_context=False) is True
    # web service (web context): name-based match applies
    assert deepen.is_entry_point(handler, pats, web_context=True) is True


# ---------------------------------------------------------------------------
# Full path — only when the optional deps are installed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter-language-pack not installed")
def test_full_path_entry_point_and_security_control(repo: Path):
    out = deepen.deepen(repo)
    by_fn = {c["function"]: c for c in out["candidates"]}

    assert by_fn["handle_upload"]["is_entry_point"] is True
    assert by_fn["handle_upload"]["exposure"] == "exposed_externally"
    # the security_control tier survives (review finding #6)
    assert by_fn["validate_token"]["exposure"] == "security_control"
    # noise is dropped
    assert "helper" not in by_fn
    # sorted by priority, entry point first
    scores = [c["priority_score"] for c in out["candidates"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(not (HAS_TREE_SITTER and HAS_BANDIT), reason="bandit/tree-sitter not installed")
def test_full_path_bandit_finding_attached(repo: Path):
    out = deepen.deepen(repo)
    by_fn = {c["function"]: c for c in out["candidates"]}
    finding = by_fn["handle_upload"]["finding"]
    assert finding is not None and finding["tool"] == "bandit"
    assert finding["test_id"].startswith("B")


@pytest.mark.skipif(not (HAS_TREE_SITTER and HAS_BANDIT), reason="bandit/tree-sitter not installed")
def test_bandit_confirmed_sink_surfaces_even_without_regex_match(tmp_path: Path):
    # The file is flagged by stage 1 via run_cmd's subprocess sink. make_token has
    # no regex sink, only a Bandit-detectable weak-hash issue (B324) — it must still
    # surface (not be filtered as neutral) once stage 2 looks at the file.
    (tmp_path / "svc.py").write_text(
        "import subprocess, hashlib\n"
        "def run_cmd(c):\n"
        "    subprocess.run(c)\n"
        "def make_token(seed):\n"
        "    return hashlib.md5(seed).hexdigest()\n"
    )
    out = deepen.deepen(tmp_path)
    by_fn = {c["function"]: c for c in out["candidates"]}
    assert "make_token" in by_fn, "a Bandit-confirmed issue must not be filtered as noise"
    cand = by_fn["make_token"]
    assert cand["sink_matches"] == []  # no regex sink
    assert cand["finding"]["tool"] == "bandit"
    assert cand["exposure"] == "attack_surface"  # elevated from neutral by the AST check


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter-language-pack not installed")
def test_web_context_gating_end_to_end(tmp_path: Path):
    library = (
        "import subprocess\n"
        "def run_thing(c):\n"
        "    subprocess.run(c)\n"
        "def _resolve_handler(ct):\n"
        "    return HANDLERS[ct]\n"
    )
    # Library (no web framework): the name-based entry point is suppressed, so the
    # sink-less _resolve_handler is not surfaced as exposed_externally.
    (tmp_path / "lib.py").write_text(library)
    out = deepen.deepen(tmp_path)
    by_fn = {c["function"]: c for c in out["candidates"]}
    assert "_resolve_handler" not in by_fn
    assert any("no web-framework signal" in w for w in out["warnings"])

    # Add a web framework import: the same function now counts as an entry point.
    (tmp_path / "lib.py").write_text("from fastapi import APIRouter\n" + library)
    out2 = deepen.deepen(tmp_path)
    by_fn2 = {c["function"]: c for c in out2["candidates"]}
    assert by_fn2.get("_resolve_handler", {}).get("exposure") == "exposed_externally"
    assert not any("no web-framework signal" in w for w in out2["warnings"])


@pytest.mark.skipif(not (HAS_TREE_SITTER and HAS_BANDIT), reason="bandit/tree-sitter not installed")
def test_module_level_bandit_finding_surfaces_at_module_scope(tmp_path: Path):
    # app.run(debug=True) is B201 (HIGH) at module level, outside any function. It
    # must still surface as a <module>-scope candidate, not be silently dropped.
    (tmp_path / "srv.py").write_text(
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "def index():\n"
        "    return 'ok'\n"
        "app.run(host='0.0.0.0', debug=True)\n"
    )
    out = deepen.deepen(tmp_path, files=["srv.py"])
    module_cands = [c for c in out["candidates"] if c["function"] == "<module>"]
    assert module_cands, "a module-level Bandit finding must surface at module scope"
    mc = module_cands[0]
    assert mc["finding"]["tool"] == "bandit"
    assert mc["finding"]["severity"] == "high"  # B201 debug=True
    assert mc["exposure"] == "attack_surface"


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="tree-sitter-language-pack not installed")
def test_decorated_entry_outranks_bare_name_only_entry(tmp_path: Path):
    # Web-context repo (fastapi import): both a decorated route and a bare getter
    # match the entry-point heuristic, but the decorated one must rank higher.
    (tmp_path / "app.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/models')\n"
        "def on_message(msg):\n"
        "    return handle(msg)\n"
        "def get_console():\n"
        "    return Console()\n"
    )
    # pass the file explicitly: it carries no sink, so a self-sweep wouldn't flag it,
    # and this test is about entry-point ranking, not sink discovery.
    out = deepen.deepen(tmp_path, files=["app.py"])
    by_fn = {c["function"]: c for c in out["candidates"]}
    assert "on_message" in by_fn and "get_console" in by_fn
    assert by_fn["on_message"]["priority_score"] > by_fn["get_console"]["priority_score"]


@pytest.mark.skipif(not (HAS_TREE_SITTER and HAS_BANDIT), reason="bandit/tree-sitter not installed")
def test_low_severity_bandit_hit_is_not_elevated(tmp_path: Path):
    # compute_total has only a bare `assert` (B101, low severity); the file is flagged
    # by run_cmd's subprocess sink. The assert must NOT be elevated to attack_surface
    # and must rank low, so common assert noise doesn't drown out real findings.
    (tmp_path / "svc.py").write_text(
        "import subprocess\n"
        "def run_cmd(c):\n"
        "    subprocess.run(c)\n"
        "def compute_total(items):\n"
        "    assert items\n"
        "    return sum(items)\n"
    )
    out = deepen.deepen(tmp_path)
    by_fn = {c["function"]: c for c in out["candidates"]}
    if "compute_total" in by_fn:  # kept as a candidate because Bandit confirmed it
        cand = by_fn["compute_total"]
        assert cand["exposure"] == "neutral"  # low severity -> not elevated
        assert cand["priority_score"] < by_fn["run_cmd"]["priority_score"]
