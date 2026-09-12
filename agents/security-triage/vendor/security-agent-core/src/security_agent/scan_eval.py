"""Score a repository scan against verified findings, in failure labels rather than a number.

The shape follows the harness evaluators in Haystack's agent-pack (haystack-core-integrations
#3932/#3947): an eval case names what a correct run must produce, a run is reduced to a tuple of
**failure labels** (a case passes when the tuple is empty), and quality is the fraction of cases
that passed. Labels are what an optimizer can reason over — `missed_finding:filters.py:119 x3`
says where the harness falls short; `0.43` does not.

The labels, and why each exists:

- ``no_report`` — the final answer held no parseable report. Nothing else can be scored.
- ``scan_incomplete`` — the run errored or ran out of steps and had its answer forced. Like the
  agent-pack's ``usage_incomplete`` gate: a candidate whose runs cannot be fully accounted for
  cannot win, whatever it scored, because a truncated scan looks exactly like a silently failing one.
- ``missed_finding:<file>:<line>`` — a verified finding the report does not contain (same file,
  line within tolerance).
- ``unverifiable_citation:<file>:<line>`` — a reported finding whose file does not exist in the
  repository or whose line is past its end. The security analogue of a citation that resolves to no
  document: a hallucination gate that needs no ground truth.
- ``noise_finding:<class>`` — a finding of a class the reviewer prompt excludes (asserts, broad
  excepts, style). Counted once per class so a report with twelve asserts is one label, not twelve.
- ``tool_calls_over_budget:<calls>/<limit>`` — more tool calls than the case allows.

Nothing here knows about Docker, NemoClaw or the model: it takes the JSON record
``security_agent.repo_scan`` writes and a case, and returns labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_TOOL_BUDGET = 40
DEFAULT_LINE_TOLERANCE = 3

# Classes of finding the reviewer prompt tells the agent not to report. Matched on the title,
# case-insensitively; each regex is a class so the label stays coarse.
NOISE_CLASSES: dict[str, re.Pattern[str]] = {
    "assert": re.compile(r"\bassert\b", re.I),
    "broad_except": re.compile(r"broad (exception|except)|bare except|except(ion)? (suppress|swallow)", re.I),
    "style": re.compile(r"\b(type hint|docstring|naming|formatting|unused (import|variable))\b", re.I),
}


@dataclass(frozen=True)
class ExpectedFinding:
    """One verified defect the scan must report. `why` is for the reader and the optimizer's digest."""

    file: str
    line: int
    why: str = ""
    cwe: Optional[str] = None
    tolerance: int = DEFAULT_LINE_TOLERANCE

    def matches(self, finding: dict[str, Any]) -> bool:
        reported_file = str(finding.get("file") or "")
        if not (reported_file == self.file or reported_file.endswith("/" + self.file)):
            return False
        try:
            line = int(finding.get("line"))
        except (TypeError, ValueError):
            return False
        return abs(line - self.line) <= self.tolerance


@dataclass(frozen=True)
class ScanEvalCase:
    """A repository with its verified findings.

    :param repo: Name of the repository directory (inside the sandbox: `/sandbox/work/<repo>`).
    :param expected: The findings a correct report contains.
    :param source: Host path of the checkout, used to verify cited file:line pairs. Optional; without it
        the `unverifiable_citation` label is never produced.
    :param tool_budget: Maximum tool calls for one scan.
    """

    repo: str
    expected: tuple[ExpectedFinding, ...]
    source: Optional[str] = None
    tool_budget: int = DEFAULT_TOOL_BUDGET

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScanEvalCase":
        expected = tuple(
            ExpectedFinding(
                file=str(item["file"]),
                line=int(item["line"]),
                why=str(item.get("why", "")),
                cwe=item.get("cwe"),
                tolerance=int(item.get("tolerance", DEFAULT_LINE_TOLERANCE)),
            )
            for item in data.get("expected", [])
        )
        return cls(
            repo=str(data["repo"]),
            expected=expected,
            source=data.get("source"),
            tool_budget=int(data.get("tool_budget", DEFAULT_TOOL_BUDGET)),
        )

    def fingerprint_dict(self) -> dict[str, Any]:
        """Stable content for the measurement-context digest (no host paths)."""
        return {
            "repo": self.repo,
            "expected": sorted((e.file, e.line, e.tolerance) for e in self.expected),
            "tool_budget": self.tool_budget,
        }


@dataclass
class ScanCaseMetrics:
    repo: str
    failures: tuple[str, ...]
    recall: float
    n_findings: int
    n_noise: int
    tool_calls: int
    steps: Optional[int]
    seconds: Optional[float]
    digest: Optional[dict[str, Any]] = None

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "passed": self.passed,
            "failures": list(self.failures),
            "recall": round(self.recall, 3),
            "n_findings": self.n_findings,
            "n_noise": self.n_noise,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "seconds": self.seconds,
            **({"digest": self.digest} if self.digest is not None else {}),
        }


def tool_calls_of(record: dict[str, Any]) -> list[tuple[str, Any]]:
    """(tool name, arguments) for every tool call in the record's trajectory, in order."""
    calls: list[tuple[str, Any]] = []
    for message in record.get("trajectory") or []:
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                function = call.get("function") or {}
                calls.append((str(function.get("name") or call.get("tool_name") or "?"), function.get("arguments")))
    return calls


def _file_line_count(source: Path, file: str) -> Optional[int]:
    """Lines in `file` under `source`, None when the file is missing. `file` may carry the repo prefix."""
    for candidate in (source / file, *(source / Path(*Path(file).parts[i:]) for i in range(1, len(Path(file).parts)))):
        if candidate.is_file():
            try:
                return sum(1 for _ in candidate.open(errors="replace"))
            except OSError:
                return None
    return None


def score_record(record: dict[str, Any], case: ScanEvalCase, *, keep_digest: bool = True) -> ScanCaseMetrics:
    """Reduce one scan record to labels. Never raises on a malformed record: that is `no_report`."""
    failures: list[str] = []
    report = record.get("report") if isinstance(record.get("report"), dict) else None
    findings = [f for f in (report or {}).get("findings") or [] if isinstance(f, dict)]
    calls = tool_calls_of(record)

    if record.get("error") or record.get("forced_final_answer"):
        failures.append("scan_incomplete")
    if report is None:
        failures.append("no_report")

    hits = 0
    for expected in case.expected:
        if any(expected.matches(f) for f in findings):
            hits += 1
        else:
            failures.append(f"missed_finding:{expected.file}:{expected.line}")
    recall = hits / len(case.expected) if case.expected else 1.0

    noise_classes: set[str] = set()
    for finding in findings:
        title = str(finding.get("title") or "")
        for name, pattern in NOISE_CLASSES.items():
            if pattern.search(title):
                noise_classes.add(name)
    failures.extend(f"noise_finding:{name}" for name in sorted(noise_classes))
    n_noise = sum(1 for f in findings if any(p.search(str(f.get("title") or "")) for p in NOISE_CLASSES.values()))

    if case.source:
        source = Path(case.source)
        for finding in findings:
            file = str(finding.get("file") or "")
            try:
                line = int(finding.get("line"))
            except (TypeError, ValueError):
                line = 0
            total = _file_line_count(source, file) if file else None
            if total is None or line < 1 or line > total:
                failures.append(f"unverifiable_citation:{file}:{line}")

    if len(calls) > case.tool_budget:
        failures.append(f"tool_calls_over_budget:{len(calls)}/{case.tool_budget}")

    metrics = ScanCaseMetrics(
        repo=case.repo,
        failures=tuple(failures),
        recall=recall,
        n_findings=len(findings),
        n_noise=n_noise,
        tool_calls=len(calls),
        steps=record.get("steps"),
        seconds=record.get("seconds"),
    )
    if keep_digest and failures:
        metrics.digest = digest_record(record, case)
    return metrics


# --------------------------------------------------------------------------- #
# Digest: what the optimizer reads about a failing case
# --------------------------------------------------------------------------- #
MAX_TOOL_STEPS = 30
MAX_ARGUMENT_CHARS = 600
MAX_RESULT_CHARS = 400
MAX_ANSWER_CHARS = 800


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [{len(text) - limit} more characters omitted]"


def digest_record(record: dict[str, Any], case: ScanEvalCase) -> dict[str, Any]:
    """Tool calls with arguments and (truncated) results, the reported findings, the expected ones.

    Arguments are kept longer than results: they are what the agent chose, and the highest-value
    evidence per character. Every cut is marked so a reader cannot mistake truncation for absence.
    """
    steps: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    for message in record.get("trajectory") or []:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            step = {"tool": function.get("name"), "arguments": _truncate(function.get("arguments") or "", MAX_ARGUMENT_CHARS)}
            pending[str(call.get("id") or len(steps))] = step
            steps.append(step)
        if message.get("role") == "tool":
            step = pending.pop(str(message.get("tool_call_id") or ""), None)
            if step is not None:
                step["result"] = _truncate(message.get("content") or "", MAX_RESULT_CHARS)
    omitted = max(0, len(steps) - MAX_TOOL_STEPS)
    report = record.get("report") if isinstance(record.get("report"), dict) else {}
    return {
        "repo": case.repo,
        "steps": record.get("steps"),
        "forced_final_answer": bool(record.get("forced_final_answer")),
        "error": record.get("error"),
        "tool_steps": steps[:MAX_TOOL_STEPS],
        **({"omitted_tool_calls": omitted} if omitted else {}),
        "reported_findings": [
            {k: _truncate(f.get(k), 200) for k in ("file", "line", "severity", "cwe", "title") if k in f}
            for f in (report.get("findings") or [])
            if isinstance(f, dict)
        ][:20],
        "expected_findings": [{"file": e.file, "line": e.line, "cwe": e.cwe, "why": e.why} for e in case.expected],
        "answer_excerpt": _truncate(record.get("raw_answer") or "", MAX_ANSWER_CHARS),
    }


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
@dataclass
class ScanMetrics:
    """One measurement of a configuration over every eval case. `quality` is in [0, 1]."""

    quality: float
    cases: list[ScanCaseMetrics] = field(default_factory=list)

    @property
    def incomplete(self) -> bool:
        return any("scan_incomplete" in c.failures or "no_report" in c.failures for c in self.cases)

    @property
    def mean_recall(self) -> float:
        return sum(c.recall for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def total_tool_calls(self) -> int:
        return sum(c.tool_calls for c in self.cases)

    @property
    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.cases:
            for label in c.failures:
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_dict(self, *, with_digests: bool = True) -> dict[str, Any]:
        cases = [c.to_dict() for c in self.cases]
        if not with_digests:
            for c in cases:
                c.pop("digest", None)
        return {
            "quality": round(self.quality, 4),
            "mean_recall": round(self.mean_recall, 4),
            "incomplete": self.incomplete,
            "total_tool_calls": self.total_tool_calls,
            "failure_counts": self.failure_counts,
            "cases": cases,
        }


def aggregate(case_metrics: Iterable[ScanCaseMetrics]) -> ScanMetrics:
    cases = list(case_metrics)
    quality = sum(1 for c in cases if c.passed) / len(cases) if cases else 0.0
    return ScanMetrics(quality=quality, cases=cases)


def headline(metrics: ScanMetrics) -> str:
    """One line per measurement, the way the optimizer sees history."""
    clean = sum(1 for c in metrics.cases if c.passed)
    parts = [
        f"quality {metrics.quality:.2f} ({clean}/{len(metrics.cases)} repos clean)",
        f"recall {metrics.mean_recall:.2f}",
        f"tool calls {metrics.total_tool_calls}",
    ]
    if metrics.incomplete:
        parts.append("INCOMPLETE")
    if metrics.failure_counts:
        parts.append("failures " + ", ".join(f"{k} x{v}" for k, v in list(metrics.failure_counts.items())[:8]))
    return " | ".join(parts)
