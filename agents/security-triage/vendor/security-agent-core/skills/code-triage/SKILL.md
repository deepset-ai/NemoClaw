---
name: code-triage
description: >-
  Narrow an unfamiliar repo down to the functions most likely to matter for a
  vulnerability hunt, before reading anything in full. Use this before grepping ad
  hoc or reading files one by one in a repo you haven't explored yet. Two stages: a
  cheap whole-repo sweep for dangerous-sink patterns, then a deeper tree-sitter /
  Bandit pass on only the flagged files, returning a ranked JSON shortlist of
  functions with file / line-range / why — never full file contents. Triggers on:
  "where are the vulnerabilities", "triage this repo", "what should I look at
  first", "find the attack surface", "which functions are risky", auditing a CVE
  checkout, or starting a white-box review of unfamiliar code. PREREQUISITE — this
  skill runs helper scripts, so the loading agent must independently have a
  shell/code-execution surface (OpenClaw code-execution, or SEC-bench's DockerShell)
  plus the skill venv's python3; a SkillToolset-only agent can read these scripts but
  cannot run them.
---

# Code triage

Turn "here is a repo I've never seen" into "here are the ~40 functions worth
reading first, ranked, with a reason for each" — without pulling whole files into
context. Two stages:

1. **`scripts/sweep.py`** — one cheap Python-`re` pass over the whole repo, flagging
   files that contain a dangerous-sink pattern (subprocess, `eval(`, `os.system`,
   outbound HTTP, C `system()`/`popen()`, Go `exec.Command`, Java `ProcessBuilder`,
   …). Cross-language. Output is a list of files + which patterns matched on which
   lines. No file contents.
2. **`scripts/deepen.py`** — on the flagged files only: resolve functions with
   tree-sitter, classify each by exposure (entry point → security control → attack
   surface → reachable → neutral), approximate reachability from entry points, verify
   Python sinks with Bandit (regex fallback otherwise), score, and return a ranked
   top-N JSON shortlist.

## How to run this skill (REQUIRED — read first)

This skill is **helper scripts, not a callable tool**. `SkillToolset` only lets an
agent `load_skill` / `read_skill_file` — it has no execution surface. To run the
scripts you (the consuming agent) must independently have a shell / code-execution
tool, and you invoke the scripts through the skill venv's `python3`.

Let `D` be this skill's directory (e.g.
`/sandbox/.openclaw/workspace/skills/code-triage` in the NemoClaw sandbox). Run
through your shell/exec tool:

```bash
# Stage 1 — whole-repo sweep
$D/venv/bin/python3 $D/scripts/sweep.py <repo_root>

# Stage 2 — deepen the flagged files into a ranked shortlist
$D/venv/bin/python3 $D/scripts/deepen.py <repo_root> --files <f1> <f2> ...
```

Under OpenClaw's JS `tool_search_code`, that is:

```js
const D = '/sandbox/.openclaw/workspace/skills/code-triage';
return await openclaw.tools.call('openclaw:core:exec', {
  command: `${D}/venv/bin/python3 ${D}/scripts/sweep.py /sandbox/.openclaw/workspace/target-repo`
});
```

The venv's `python3` is allowlisted by the standard skill-install policy, so no
extra `openshell policy --binary` rule is needed for the default path.

## Recommended workflow

1. Run `sweep.py <repo_root>`. Read the `files[].path` list.
2. Run `deepen.py <repo_root> --files <the flagged paths>`. (If you skip `--files`,
   `deepen.py` runs its own sweep first — simpler, slightly slower.) Pass the paths
   as **literal space-separated argv tokens**, not a single shell variable — some
   shells (zsh) don't word-split an unquoted `$VAR`, which would send the whole list
   as one bogus filename. If any `--files` entry doesn't resolve, `deepen.py` says so
   in `warnings` (a fully unresolved list is called out loudly), so an empty result
   is never silently confused with "clean repo".
3. Treat the returned JSON as the answer. **Do not** paste full file contents into
   context. Use each candidate's `file` / `line_start` / `line_end` to open only the
   ranges you decide are worth reading in full. Sort by `priority_score`, but **read
   `sink_matches` too, don't trust score order blindly**: an entry point with no
   sinks scores 100+ on exposure alone and can outrank a concrete dangerous call —
   its risk usually lives one hop away in a callee this pass doesn't inline. Prefer
   candidates that pair a high score with a non-empty `sink_matches`.
4. If `warnings` reports a tool as unavailable (e.g. tree-sitter or Bandit missing),
   proceed anyway — the ranking is still usable, just lower-confidence for the
   affected files. Check each candidate's `finding.tool` to see whether a hit was
   AST-verified (`bandit`) or a lower-confidence `regex-fallback`.

## Output contract

`sweep.py`:

```json
{
  "backend_used": "python-re",
  "root": "/abs/path/to/repo",
  "files": [
    { "path": "src/upload.py", "hits": [ { "line": 6, "label": "subprocess" } ] }
  ]
}
```

`deepen.py`:

```json
{
  "warnings": ["bandit unavailable (ImportError); used regex fallback for Python"],
  "candidates": [
    {
      "file": "src/upload.py",
      "function": "handle_upload",
      "line_start": 4,
      "line_end": 7,
      "sink_matches": ["subprocess"],
      "is_entry_point": true,
      "exposure": "exposed_externally",
      "hop_distance": 0,
      "ssrf_idor_flag": false,
      "finding": { "tool": "bandit", "test_id": "B607", "severity": "low" },
      "priority_score": 140
    }
  ]
}
```

Field notes:

- `exposure` — one of `exposed_externally` (entry point), `attack_surface` (dangerous
  sink or SSRF/IDOR shape), `exposed_internally` (reachable from an entry point),
  `security_control` (auth/validation function by name — kept deliberately, since
  bypass bugs live here), `neutral`. Entry points come from route decorators
  (`@app.route`, gRPC, …), always trusted, plus a name heuristic (`_handler$`,
  `^process`, …) that only applies when the repo shows a web-framework signal — in a
  plain importable library nothing is a network route, and a `warnings` line notes
  when that heuristic was disabled.
- `hop_distance` — shortest hops from an entry point (0 = is one; `-1` = not reached
  within the hop cap). A bounded grep-based approximation, not a full call graph.
- `ssrf_idor_flag` — takes an identifier-shaped param (`url`/`path`/`id`/…) **and**
  makes an outbound request.
- `finding.tool` — `bandit` (AST-verified) or `regex-fallback` (lower confidence);
  those are the only two values emitted today. Lets a reader tell a verified hit from
  a pattern guess (`priority_score` already rewards the verified ones). When a
  candidate has more than one Bandit issue, `finding.severity`/`test_id` are the
  **highest-severity** one and `finding.n_findings` is the total (so a "medium" badge
  can't hide a HIGH issue in the same function).
- `function` — the function name, `null` for a raw-hit fallback (unparsed file), or
  `"<module>"` for a Bandit issue on module-level code (imports, config,
  `app.run(debug=True)`) that lives outside any function.
- `priority_score` — exposure base + 10·risk-signals + reachability bonus + SSRF/IDOR
  bonus. Higher = look first.

Useful flags on `deepen.py`: `--top N` (shortlist size, default 40), `--max-hops N`
(reachability cap, default 2), `--max-file-bytes N`.

`deepen.py` also accepts `--semgrep`, but **it does nothing yet** — there is no Semgrep
tier in this build, so the flag only adds a warning saying so. Don't reach for it
expecting a second verification tier, and don't install Semgrep hoping to switch it on;
adding the tier would also mean giving the sandbox its own `--binary` allowlist rule for
the Semgrep CLI.

## Limitations — what this does NOT find

This is a **triage narrower**, not a scanner or a proof of safety. It ranks where to
look first; it does not decide what is exploitable. Validated against deliberately-
vulnerable apps (DSVW, vulpy, PyGoat), it reliably surfaces single-call injection
sinks but has real blind spots. **An empty or low-signal result means "no matched
patterns," never "no vulnerabilities."** Concretely, it will miss or under-rank:

- **Multi-step / dataflow vulnerabilities** that don't reduce to one recognizable
  call. Example: SSTI where a handler writes attacker-controlled content to a
  template file and a *different* function later renders it — each half looks benign
  in isolation. Reachability is a capped 2-hop name-based approximation, not real
  taint tracking, so data flow across functions is largely invisible.
- **Pure logic / access-control bugs** with no dangerous call: auth bypass via a
  wildcard/glob match, missing authorization checks, IDOR, broken session logic.
  There is no sink to match and Bandit has no rule, so these score as `neutral` and
  are dropped as noise.
- **Non-Python languages get regex only.** Bandit (the AST-verified tier) is
  Python-only; C/C++/Go/Java/JS candidates always show `regex-fallback`, a
  lower-confidence pattern guess with no semantic verification.
- **Anything outside the pattern list.** Sink detection is pattern-based: an
  obfuscated/aliased sink, a network call via an unlisted library or method, or a
  dangerous construct with no listed pattern will not be flagged. The list is a
  heuristic starting point, not exhaustive.
- **Name-heuristic noise at the top of the ranking.** Entry-point-by-name
  (`handle`/`get_`/`process…`) false-positives in web repos — notably Django's
  mandatory `BaseCommand.handle()`, which flags every management command as
  externally exposed. **Read `sink_matches` and `finding`; do not trust
  `priority_score` order blindly** (see the workflow note above).

Treat the output as "read these first," then apply real analysis. A confident
security judgment still requires reading the code.

## Setup

The scripts need `tree-sitter`, `tree-sitter-language-pack`, and `bandit` in the
skill's venv (see `requirements.txt`). Install once at setup:

```bash
python3 -m venv $D/venv
$D/venv/bin/pip install -r $D/requirements.txt
```

Everything runs under that venv's `python3` (already allowlisted). No network is
needed at triage time — only at setup. If the deps are absent the scripts still run:
sweep is stdlib-only, and deepen degrades to raw hit lines + regex checks, reporting
the degradation in `warnings`.

## Provenance

Upstream is [T3MP3ST](https://github.com/elder-plinius/T3MP3ST), **AGPL-3.0** — the full
reuse map and the licensing caveat live in this repo's `NOTICE`, next to the redamon entry.

Sink/entry-point/security-control patterns and the exposure/scoring model are ported
from T3MP3ST's `src/recon/code-ingest.ts` (`DANGEROUS_SINK_RE`, `OUTBOUND_REQUEST_RE`,
`RISKY_PARAM_RE`, `classify()`, `prioritize()`) and its tree-sitter function
extraction (`src/recon/ts-parse.ts`). Most patterns are kept verbatim, look-behind
guards included; the scoring is re-derived and keeps the full five-tier exposure
ladder. Two Python-precision divergences from the verbatim source: the `httpx` and
`urllib` sink patterns now require a call/attribute shape rather than a bare module
name (the bare forms matched type annotations and imports as fake call sites — the
top false positives on a real Python library), and AST-verified Bandit findings get
a severity-scaled score bonus so they can outrank speculative naming-pattern guesses.
