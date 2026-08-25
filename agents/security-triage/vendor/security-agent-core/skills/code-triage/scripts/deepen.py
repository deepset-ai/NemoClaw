#!/usr/bin/env python3
"""Stage 2 — deepen the stage-1 candidates into a ranked shortlist.

For each file stage 1 flagged, resolve functions (tree-sitter), classify each by
exposure, approximate reachability from entry points, verify sinks (Bandit as a
library for Python; a scoped regex re-check otherwise), and score. Emits a sorted,
top-N JSON shortlist with `file`/`line_start`/`line_end` so a reader opens only what
matters — never full file contents.

The scoring is *re-derived* from T3MP3ST's prioritize()/classify() (build-plan
finding #6): the full five-tier exposure ladder is kept, including `security_control`.

Fail-open everywhere: no tree-sitter -> keep raw hit lines (function=null); no
Bandit -> scoped regex fallback. Degradations are reported in `warnings`, never
swallowed. There is no Semgrep tier yet: `--semgrep` is a reserved placeholder that
only warns (a Semgrep tier would need its own sandbox --binary allowlist rule).

Usage:
    deepen.py <repo_root> [--files f1 f2 ...] [--top N] [--max-hops N]
                          [--semgrep] [--max-file-bytes N] [--sarif]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from common import (
    Candidate,
    Function,
    MAX_FILE_BYTES,
    Patterns,
    crawl,
    cwe_for_sink_labels,
    load_patterns,
    parse_functions,
    read_text,
)

MAX_HOPS_DEFAULT = 2

# Every `name(` call shape in a body — the callee names, read out in one pass.
_CALL_RE = re.compile(r"\b(\w+)\s*\(")
MAX_REACH_FUNCS = 5000  # runaway guard for the grep-based reachability walk
TOP_N_DEFAULT = 40

# A repo shows "web context" if a known web framework or a route decorator appears
# in the scanned files. Used to gate the NAME-based entry-point heuristic
# (`_handler$`, `^process`, …): in a plain importable library nothing is a network
# route, so a function named `_resolve_handler` is not "exposed externally". Matched
# against raw file text (imports live at module top, not in function bodies).
_WEB_FRAMEWORK_RE = re.compile(
    r"\b(flask|fastapi|starlette|aiohttp|tornado|bottle|sanic|falcon|pyramid|quart"
    r"|litestar|werkzeug|django|connexion|grpc)\b"
    r"|@(app|router|bp|blueprint)\.(route|get|post|put|delete|patch)"
    r"|\bwsgi\b|\basgi\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Classification + scoring — re-derived from code-ingest.ts classify()/prioritize().
# ---------------------------------------------------------------------------

def is_entry_point(fn: Function, pats: Patterns, web_context: bool = True) -> bool:
    # Decorator-based entry points (@app.route, @grpc…) are always trusted — they
    # are precise. The NAME-based heuristic (`_handler$`, `^process`, …) only fires
    # when the repo shows a web-framework signal; in a plain library an internally-
    # named dispatch function is not a network route (see _WEB_FRAMEWORK_RE).
    if fn.decorators and any(r.search(fn.decorators) for r in pats.entry_point_decorators):
        return True
    if not web_context:
        return False
    return any(r.search(fn.name) for r in pats.entry_point_names)


def is_decorator_entry(fn: Function, pats: Patterns) -> bool:
    """A route/RPC decorator is present — a high-precision entry-point signal."""
    return bool(
        fn.decorators and any(r.search(fn.decorators) for r in pats.entry_point_decorators)
    )


def name_matches_entry_point(fn: Function, pats: Patterns) -> bool:
    return any(r.search(fn.name) for r in pats.entry_point_names)


def compute_sink_matches(fn: Function, pats: Patterns) -> list[str]:
    """Labels of every sink pattern present in the function body (order-stable,
    de-duplicated)."""
    seen: list[str] = []
    for label, regex in pats.sink_evidence:
        if regex.search(fn.body) and label not in seen:
            seen.append(label)
    return seen


def has_ssrf_idor(fn: Function, pats: Patterns) -> bool:
    """Accepts an identifier-shaped param AND makes an outbound request."""
    if not any(pats.risky_params.search(p) for p in fn.params):
        return False
    return bool(pats.outbound_request.search(fn.body))


def classify(fn: Function, entry: bool, reachable: bool, pats: Patterns) -> str:
    """Exactly one exposure, source precedence:
    exposed_externally > security_control (by name) > attack_surface (by body/param)
    > exposed_internally (reachable) > neutral."""
    if entry:
        return "exposed_externally"
    if pats.security_controls.search(fn.name):
        return "security_control"
    if pats.dangerous_sink.search(fn.body) or has_ssrf_idor(fn, pats):
        return "attack_surface"
    if reachable:
        return "exposed_internally"
    return "neutral"


# Score bonus for a Bandit-confirmed (AST-verified) finding, by severity. Without
# this, a verified finding scores on exposure base alone and loses to a speculative
# naming-pattern guess — on a real library every Bandit hit sank below rank 42,
# never reaching a top-N shortlist. Low is deliberately tiny: B101 (assert_used) is
# low-severity and very common, and must not flood the top.
_VERIFIED_BONUS = {"high": 50, "medium": 30, "low": 5}

# Penalty for an entry point matched by NAME only (no decorator) with no other
# corroborating signal — a plain `get_config`/`handle_x` helper in a web-framework
# repo. The name heuristic is low-precision; without this a bare undecorated getter
# ties a genuinely decorated route (both exposed_externally=100, +30 for the depth-0
# self-seed = 130). The penalty (larger than that self-seed bonus) drops the guess
# to internal-helper level (~70) so a decorator-confirmed route and any concrete
# sink both outrank it.
NAME_ONLY_ENTRY_PENALTY = 60


def verified_bonus(finding: dict | None) -> int:
    if not finding or finding.get("tool") != "bandit":
        return 0
    return _VERIFIED_BONUS.get((finding.get("severity") or "").lower(), 0)


def prioritize(
    exposure: str,
    reachable: bool,
    reach_depth: int,
    n_signals: int,
    ssrf_idor: bool,
    verified_bonus: int = 0,
    entry_penalty: int = 0,
) -> int:
    from common import EXPOSURE_BASE

    score = EXPOSURE_BASE[exposure]
    score += 10 * n_signals
    if reachable:
        score += max(0, 30 - reach_depth * 5)
    score -= entry_penalty
    if ssrf_idor:
        score += 20
    score += verified_bonus
    return max(0, score)


# ---------------------------------------------------------------------------
# Reachability — grep-based forward BFS from entry points, hop- and size-capped.
# ---------------------------------------------------------------------------

def compute_reachability(
    functions: list[Function], pats: Patterns, max_hops: int, web_context: bool = True
) -> tuple[dict[int, int], list[str]]:
    """Return {index_in_functions: hop_distance} for functions reachable from an
    entry point within max_hops, plus any cap warnings. A function G is reachable
    at depth d+1 if some depth-d function's body calls G's name (`\\bG\\s*\\(`).

    This is a bounded approximation of the source's single-pass call graph — capped
    so it can't go quadratic on a large repo."""
    warnings: list[str] = []
    depth: dict[int, int] = {}

    # name -> indices (a name can be defined more than once).
    by_name: dict[str, list[int]] = {}
    for i, fn in enumerate(functions):
        by_name.setdefault(fn.name, []).append(i)

    frontier: list[int] = []
    for i, fn in enumerate(functions):
        if is_entry_point(fn, pats, web_context):
            depth[i] = 0
            frontier.append(i)

    hop = 0
    while frontier and hop < max_hops:
        hop += 1
        next_frontier: list[int] = []
        for caller_idx in frontier:
            body = functions[caller_idx].body
            # Read the called names out of the body once, instead of testing every
            # known name against it: O(len(body)) per caller, not O(len(by_name)).
            for name in set(_CALL_RE.findall(body)):
                for callee_idx in by_name.get(name, ()):
                    if callee_idx in depth:
                        continue
                    depth[callee_idx] = hop
                    next_frontier.append(callee_idx)
                    if len(depth) >= MAX_REACH_FUNCS:
                        warnings.append(
                            f"reachability capped at {MAX_REACH_FUNCS} functions; "
                            "some depths may be missing"
                        )
                        return depth, warnings
        frontier = next_frontier

    return depth, warnings


# ---------------------------------------------------------------------------
# Sink verification.
# ---------------------------------------------------------------------------

def bandit_findings(py_files: list[Path]) -> tuple[dict[str, list[dict]], list[str]]:
    """Run Bandit as a library over the given .py files. Returns
    {abs_path: [finding, ...]} plus warnings. Fail-open: a missing/broken Bandit
    yields ({}, [warning]) and the caller falls back to regex."""
    if not py_files:
        return {}, []
    try:
        from bandit.core import config as b_config
        from bandit.core import manager as b_manager
    except Exception as exc:
        return {}, [f"bandit unavailable ({exc.__class__.__name__}); used regex fallback for Python"]

    try:
        conf = b_config.BanditConfig()
        mgr = b_manager.BanditManager(conf, "file")
        mgr.discover_files([str(p) for p in py_files])
        mgr.run_tests()
        out: dict[str, list[dict]] = {}
        for issue in mgr.get_issue_list():
            fname = str(Path(issue.fname).resolve())
            sev = getattr(issue.severity, "name", str(issue.severity))
            cwe_obj = getattr(issue, "cwe", None)
            cwe_id = getattr(cwe_obj, "id", None)
            out.setdefault(fname, []).append(
                {
                    "tool": "bandit",
                    "test_id": issue.test_id,
                    "severity": str(sev).lower(),
                    "cwe": f"CWE-{cwe_id}" if cwe_id else None,
                    "line": issue.lineno,
                }
            )
        return out, []
    except Exception as exc:
        return {}, [f"bandit run failed ({exc.__class__.__name__}); used regex fallback for Python"]


_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1, "": 0, None: 0}


def _finding_from_issues(issues: list[dict]) -> dict:
    """Build a bandit finding from a set of issues covering one candidate, choosing
    the HIGHEST-severity issue (ties: lowest line). Picking the first-by-line issue
    would let a medium SQLi mask a HIGH command-injection in the same function —
    a reader triaging by the displayed severity alone would badly under-rate it."""
    worst = max(issues, key=lambda i: (_SEVERITY_RANK.get(i.get("severity"), 0), -i.get("line", 0)))
    finding = {
        "tool": "bandit",
        "test_id": worst["test_id"],
        "severity": worst["severity"],
        "cwe": worst.get("cwe"),
    }
    if len(issues) > 1:
        # tell the reader the badge is the worst of several, not the only one
        finding["n_findings"] = len(issues)
    return finding


def attach_finding(
    fn: Function, sink_matches: list[str], bandit_by_file: dict[str, list[dict]]
) -> dict | None:
    """Pick the finding for a function: the highest-severity Bandit issue whose line
    falls inside the function wins (AST-verified); otherwise a lower-confidence
    regex-fallback marker when any sink matched. None when nothing matched."""
    key = str(Path(fn.file).resolve())
    in_range = [
        issue for issue in bandit_by_file.get(key, [])
        if fn.line_start <= issue.get("line", -1) <= fn.line_end
    ]
    if in_range:
        return _finding_from_issues(in_range)
    if sink_matches:
        return {
            "tool": "regex-fallback",
            "test_id": None,
            "severity": None,
            "cwe": cwe_for_sink_labels(sink_matches),
        }
    return None


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def resolve_target_files(
    repo_root: Path, files: list[str] | None, max_file_bytes: int, pats: Patterns
) -> tuple[list[Path], list[str]]:
    """The files to deepen: an explicit --files list, or (when omitted) every file
    a fresh stage-1 sweep flags. Returns (paths, warnings).

    When --files entries don't resolve, that is reported loudly: a caller that
    passed an unsplit shell variable or a typo'd path would otherwise get an empty,
    warning-free result indistinguishable from a genuinely clean repo."""
    if files:
        resolved: list[Path] = []
        unresolved: list[str] = []
        for f in files:
            p = Path(f)
            if not p.is_absolute():
                p = repo_root / f
            if p.is_file():
                resolved.append(p)
            else:
                unresolved.append(f)

        warnings: list[str] = []
        if unresolved:
            if not resolved:
                warnings.append(
                    f"none of the {len(files)} --files entries resolved to a file "
                    f"(first bad entry: {unresolved[0]!r}); is the list a single "
                    "unsplit shell variable, or are the paths wrong?"
                )
            else:
                warnings.append(
                    f"{len(unresolved)} of {len(files)} --files entries did not "
                    f"resolve and were skipped (e.g. {unresolved[0]!r})"
                )
        return resolved, warnings

    # No explicit list — sweep the repo and take files with at least one hit.
    flagged: list[Path] = []
    for path in crawl(repo_root, max_file_bytes=max_file_bytes):
        content = read_text(path)
        if content is None:
            continue
        if pats.dangerous_sink.search(content):
            flagged.append(path)
    return flagged, []


def deepen(
    repo_root: Path,
    files: list[str] | None = None,
    top_n: int = TOP_N_DEFAULT,
    max_hops: int = MAX_HOPS_DEFAULT,
    use_semgrep: bool = False,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> dict:
    pats = load_patterns()
    repo_root = repo_root.resolve()
    warnings: list[str] = []

    targets, resolve_warnings = resolve_target_files(repo_root, files, max_file_bytes, pats)
    warnings.extend(resolve_warnings)

    # 1. Resolve functions per file; remember files that failed to parse so their
    #    raw hits survive as function=null candidates. While reading, detect whether
    #    the repo has a web-framework signal (gates the name-based entry-point rule).
    all_functions: list[Function] = []
    raw_hit_files: list[Path] = []
    web_context = False
    for path in targets:
        content = read_text(path)
        if content is None:
            continue
        if not web_context and _WEB_FRAMEWORK_RE.search(content):
            web_context = True
        fns, warn = parse_functions(path, content)
        if warn:
            warnings.append(warn)
        if fns:
            all_functions.extend(fns)
        elif pats.dangerous_sink.search(content):
            raw_hit_files.append(path)

    # A route decorator anywhere also establishes web context.
    if not web_context:
        web_context = any(is_decorator_entry(fn, pats) for fn in all_functions)
    if not web_context and any(name_matches_entry_point(fn, pats) for fn in all_functions):
        warnings.append(
            "no web-framework signal found; the name-based entry-point heuristic "
            "(_handler/_view/handle*/process*/…) is disabled to avoid mislabeling "
            "library dispatch functions as externally exposed (decorator-based entry "
            "points still apply)"
        )

    # 2. Reachability across all resolved functions.
    depth, reach_warnings = compute_reachability(all_functions, pats, max_hops, web_context)
    warnings.extend(reach_warnings)

    # 3. Sink verification — Bandit (library) for Python, regex fallback elsewhere.
    #    Run on every .py TARGET (not just files with extracted functions) so a
    #    module-only file's issues are still caught and can surface at module scope.
    py_files = sorted({p for p in targets if p.suffix == ".py"})
    bandit_by_file, bandit_warnings = bandit_findings(py_files)
    warnings.extend(bandit_warnings)

    if use_semgrep:
        # Placeholder, not a capability: no Semgrep tier exists yet, so there is
        # nothing to probe for on PATH and nothing to install. Say that plainly —
        # the previous "not on PATH" wording implied installing it would help.
        warnings.append(
            "--semgrep is not implemented yet; ran the bandit/regex path. No "
            "candidate will carry finding.tool == 'semgrep'."
        )

    # 4. Classify + score every resolved function.
    candidates: list[Candidate] = []
    for i, fn in enumerate(all_functions):
        entry = is_entry_point(fn, pats, web_context)
        decorator_entry = is_decorator_entry(fn, pats)
        reachable = i in depth
        reach_depth = depth.get(i, -1)
        sink_matches = compute_sink_matches(fn, pats)
        ssrf = has_ssrf_idor(fn, pats)
        exposure = classify(fn, entry, reachable, pats)
        finding = attach_finding(fn, sink_matches, bandit_by_file)

        bandit_confirmed = bool(finding and finding.get("tool") == "bandit")
        severity = (finding.get("severity") or "").lower() if finding else ""
        # A medium/high Bandit-confirmed sink is real attack surface even when it
        # used no pattern in sinks.json (e.g. B615 unsafe download, B506 yaml.load) —
        # trust the AST check over the regex list. Low-severity hits (B101 assert,
        # B110 try/except/pass) are NOT elevated: they stay neutral and rank low, so
        # they don't drown out real findings.
        if bandit_confirmed and severity in ("medium", "high") and exposure == "neutral":
            exposure = "attack_surface"

        # skip pure noise: neutral, no sinks, not reachable, not an entry point,
        # and nothing a tool confirmed.
        if (
            exposure == "neutral"
            and not sink_matches
            and not reachable
            and not entry
            and not bandit_confirmed
        ):
            continue

        # A name-only entry point (no decorator) whose sole basis for
        # exposed_externally is the name prefix — carrying no sink or SSRF signal —
        # is docked, so a decorated route or a concrete sink outranks it. (Not gated
        # on `reachable`: every entry point is trivially depth-0 reachable, so that
        # would never fire; a sink/SSRF is the real corroboration.)
        name_only_bare_entry = (
            exposure == "exposed_externally"
            and not decorator_entry
            and not sink_matches
            and not ssrf
        )
        entry_penalty = NAME_ONLY_ENTRY_PENALTY if name_only_bare_entry else 0

        n_signals = len(sink_matches) + (1 if ssrf else 0)
        score = prioritize(
            exposure, reachable, reach_depth, n_signals, ssrf,
            verified_bonus(finding), entry_penalty,
        )

        candidates.append(
            Candidate(
                file=_rel(fn.file, repo_root),
                function=fn.name,
                line_start=fn.line_start,
                line_end=fn.line_end,
                sink_matches=sink_matches,
                is_entry_point=entry,
                exposure=exposure,
                hop_distance=reach_depth,
                ssrf_idor_flag=ssrf,
                priority_score=score,
                finding=finding,
            )
        )

    # 4b. Module-scope Bandit findings: issues on statements outside every extracted
    #     function (imports, `app.config[...] = ...`, `app.run(debug=True)`) would
    #     otherwise be dropped — that hid a HIGH-severity B201 debug-server exposure
    #     on a real app. Surface one candidate per file for the worst uncovered issue,
    #     but only medium/high (module-level lows like B404 are noise).
    fn_ranges: dict[str, list[tuple[int, int]]] = {}
    for fn in all_functions:
        fn_ranges.setdefault(str(Path(fn.file).resolve()), []).append((fn.line_start, fn.line_end))
    for key, issues in bandit_by_file.items():
        uncovered = [
            i for i in issues
            if not any(s <= i.get("line", -1) <= e for s, e in fn_ranges.get(key, []))
        ]
        if not uncovered:
            continue
        finding = _finding_from_issues(uncovered)
        if finding["severity"] not in ("medium", "high"):
            continue
        worst_line = max(
            uncovered, key=lambda i: (_SEVERITY_RANK.get(i.get("severity"), 0), -i.get("line", 0))
        ).get("line", 1)
        candidates.append(
            Candidate(
                file=_rel(key, repo_root),
                function="<module>",
                line_start=worst_line,
                line_end=worst_line,
                sink_matches=[],
                is_entry_point=False,
                exposure="attack_surface",
                hop_distance=-1,
                ssrf_idor_flag=False,
                priority_score=prioritize("attack_surface", False, -1, 0, False, verified_bonus(finding)),
                finding=finding,
            )
        )

    # 5. Raw-hit fallback candidates (files tree-sitter couldn't parse): one
    #    lower-confidence attack_surface candidate per file so nothing is silently
    #    dropped.
    for path in raw_hit_files:
        content = read_text(path)
        if content is None:
            continue
        labels: list[str] = []
        first_line = None
        last_line = 1
        for lineno, line in enumerate(content.splitlines(), start=1):
            for label, regex in pats.sink_evidence:
                if regex.search(line):
                    if label not in labels:
                        labels.append(label)
                    first_line = lineno if first_line is None else first_line
                    last_line = lineno
        if not labels:
            continue
        score = prioritize("attack_surface", False, -1, len(labels), False)
        candidates.append(
            Candidate(
                file=_rel(str(path), repo_root),
                function=None,
                line_start=first_line or 1,
                line_end=last_line,
                sink_matches=labels,
                is_entry_point=False,
                exposure="attack_surface",
                hop_distance=-1,
                ssrf_idor_flag=False,
                priority_score=score,
                finding={
                    "tool": "regex-fallback",
                    "test_id": None,
                    "severity": None,
                    "cwe": cwe_for_sink_labels(labels),
                },
            )
        )

    # Collapse duplicate warnings (e.g. one "tree-sitter unavailable" per file)
    # while preserving first-seen order.
    warnings = list(dict.fromkeys(warnings))

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    if len(candidates) > top_n:
        warnings.append(
            f"output capped to top {top_n} of {len(candidates)} candidates; "
            "raise --top to see more"
        )
        candidates = candidates[:top_n]

    return {
        "warnings": warnings,
        "candidates": [c.to_dict() for c in candidates],
    }


def _rel(path_str: str, repo_root: Path) -> str:
    try:
        return str(Path(path_str).resolve().relative_to(repo_root))
    except ValueError:
        return path_str


# ---------------------------------------------------------------------------
# SARIF 2.1.0 rendering — lets any SARIF consumer (GitHub code scanning, IDEs,
# a downstream harness) ingest the shortlist. CWE classes become rule tags
# (external/cwe/CWE-NN) so consumers can group by weakness.
# ---------------------------------------------------------------------------

_SARIF_LEVEL = {"high": "error", "medium": "warning", "low": "note"}


def _rule_id(cand: dict) -> str:
    finding = cand.get("finding") or {}
    if finding.get("tool") == "bandit" and finding.get("test_id"):
        return finding["test_id"]
    if finding.get("cwe"):
        return finding["cwe"]
    return f"code-triage/{cand['exposure']}"


def _rule_desc(cand: dict, rid: str) -> str:
    finding = cand.get("finding") or {}
    if finding.get("tool") == "bandit":
        return f"Bandit {rid} ({finding.get('cwe') or 'security'}), AST-verified"
    if rid.startswith("CWE-"):
        return f"Regex-matched sink, unverified ({rid})"
    return f"Triage lead by exposure: {cand['exposure']}"


def to_sarif(result: dict, repo_root: Path) -> dict:
    """Render a deepen() result as a single SARIF 2.1.0 run. Each candidate becomes a
    result located at its file/line range; each distinct rule carries the CWE as an
    `external/cwe/CWE-NN` tag. deepen warnings become tool-execution notifications."""
    rules: dict[str, dict] = {}
    results: list[dict] = []

    for c in result["candidates"]:
        finding = c.get("finding") or {}
        rid = _rule_id(c)
        cwe = finding.get("cwe")
        level = _SARIF_LEVEL.get((finding.get("severity") or "").lower(), "note")

        if rid not in rules:
            tags = ["security"] + ([f"external/cwe/{cwe}"] if cwe else [])
            rules[rid] = {
                "id": rid,
                "name": rid.replace("/", "-"),
                "shortDescription": {"text": _rule_desc(c, rid)},
                "defaultConfiguration": {"level": level},
                "properties": {"tags": tags},
            }

        fn = c.get("function") or "<module>"
        sinks = ", ".join(c.get("sink_matches") or []) or "no direct sink"
        tool_note = f"; {finding['tool']} {finding.get('test_id') or ''}".rstrip() if finding else ""
        message = f"{c['exposure']}: {fn} — {sinks}{tool_note}" + (f" [{cwe}]" if cwe else "")

        props = {
            "priority_score": c["priority_score"],
            "exposure": c["exposure"],
            "is_entry_point": c["is_entry_point"],
            "hop_distance": c["hop_distance"],
            "ssrf_idor_flag": c["ssrf_idor_flag"],
            "function": c.get("function"),
            "tool": finding.get("tool"),
        }
        if cwe:
            props["cwe"] = cwe
        if finding.get("n_findings"):
            props["n_findings"] = finding["n_findings"]

        results.append({
            "ruleId": rid,
            "level": level,
            "message": {"text": message},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": c["file"], "uriBaseId": "SRCROOT"},
                    "region": {
                        "startLine": max(1, c["line_start"]),
                        "endLine": max(1, c["line_end"]),
                    },
                }
            }],
            "properties": props,
        })

    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "code-triage",
                "informationUri": "https://github.com/deepset-ai/security-agent",
                "version": "1.0.0",
                "rules": list(rules.values()),
            }},
            "originalUriBaseIds": {"SRCROOT": {"uri": repo_root.resolve().as_uri() + "/"}},
            "invocations": [{
                "executionSuccessful": True,
                "toolExecutionNotifications": [
                    {"level": "note", "message": {"text": w}} for w in result.get("warnings", [])
                ],
            }],
            "results": results,
        }],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-2 deepen + rank triage candidates.")
    parser.add_argument("repo_root", help="Path to the repository.")
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Candidate files from stage 1 (relative to repo_root or absolute). "
             "If omitted, deepen.py runs its own stage-1 sweep.",
    )
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT, help="Keep top-N candidates.")
    parser.add_argument("--max-hops", type=int, default=MAX_HOPS_DEFAULT, help="Reachability hop cap.")
    parser.add_argument(
        "--semgrep", action="store_true",
        help="Reserved — a Semgrep tier is not implemented yet; the flag only emits a warning.",
    )
    parser.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES, help="Skip files larger than this.")
    parser.add_argument(
        "--sarif", action="store_true",
        help="Emit SARIF 2.1.0 instead of the default JSON (for GitHub code scanning, IDEs, etc.).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    if not repo_root.is_dir():
        print(json.dumps({"error": f"not a directory: {repo_root}"}), file=sys.stderr)
        return 2

    result = deepen(
        repo_root,
        files=args.files,
        top_n=args.top,
        max_hops=args.max_hops,
        use_semgrep=args.semgrep,
        max_file_bytes=args.max_file_bytes,
    )
    output = to_sarif(result, repo_root) if args.sarif else result
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
