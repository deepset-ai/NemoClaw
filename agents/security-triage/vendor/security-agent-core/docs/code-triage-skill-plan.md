# Code-triage skill — build plan

Decisions locked in from the research phase (`code-scanner-plan.md`) plus this round's calls:

- Ships as a **Haystack 3.0 `Skill`**, loaded via `SkillToolset` + `FileSystemSkillStore` — not a
  hand-wired `Tool`, not a Claude-Code-only artifact. Directly usable by any Haystack `Agent` that
  has the skill store in its `tools=`, including SEC-bench's container-agent.
- **Fallback-in-script**: no dependency on the SEC-bench per-CVE Docker images having anything
  preinstalled. Every external tool is optional at runtime; the script degrades instead of failing.
- **Vendor `ripgrep`** (MIT/Unlicense, static musl Linux binary) directly in the skill directory for
  speed, since it's a single small static binary — no image-build changes needed.
- Two-stage flow: cheap whole-repo sweep, then tree-sitter/Bandit/Semgrep only on files the sweep
  flagged. Output is structured JSON (file/function/line-range/why), never full file contents.

No code written yet — this is the concrete build plan; implementation is the next session.

## Review findings (2026-08-10) — fix before building

Scrutinized against the actual T3MP3ST source and the real Haystack 3.0 skill API. The
architecture holds up — `FileSystemSkillStore` (`haystack.skill_stores.file_system`) and
`SkillToolset` (`haystack.tools`) exist in the 3.0 build the repo pins, and shipping this as a
progressive-disclosure skill is the right call.

**Deployment target: NemoClaw / OpenShell sandbox** (strict Linux, amd64, python 3.13, per the
`haystack-rag-openclaw-guide.md` reference deployment). This is not a generic "any container"
environment, and that assumption simplifies the plan substantially. Concretely:

- Skills get a **per-skill venv** with deps pip-installed at setup time (the `install.sh` pattern
  from the RAG demo). So tree-sitter / Bandit / PyYAML can simply be *required* — the elaborate
  "everything optional, degrade to stdlib" ladder was hedging against an environment we don't have.
- Execution is **policy-gated by a binary allowlist**: a process runs only if its path is allowlisted
  (`openshell policy … --binary <path>`). The venv's `python3` is allowlisted; **any other binary you
  shell out to is not, until you add a policy rule for it.** This makes in-process library calls
  strictly cheaper than external CLIs, and makes a vendored binary a policy problem, not just an arch
  problem.
- The skill runs **in-sandbox against the sandbox filesystem** — so the target repo lives in the
  sandbox workspace and the host-vs-container question below is answered: in-sandbox.

Six issues to resolve first, in priority order:

1. **Use Python `re` as the single stage-1 backend and keep T3MP3ST's patterns verbatim — don't
   introduce ripgrep at all.** The whole three-backend / regex-portability problem only existed to
   accommodate `rg`, whose Rust engine can't run the source's look-behind guards
   (`(?<![\w.])system\(`, `(?<![\w.])popen\(`, …). Python's `re` supports fixed-width look-behind, so
   in a controlled venv the cleanest path is: one backend, Python `re`, patterns **copied verbatim**
   from `DANGEROUS_SINK_RE`/`OUTBOUND_REQUEST_RE` including the guards that were deliberately added to
   keep the corpus rankings stable. This gives full source fidelity *and* removes the "stable contract
   across backends" problem by having one backend. (This reverses the earlier "strip the look-arounds
   for portability" fix — that was only needed if rg stayed in the picture.)

2. **Drop vendored ripgrep entirely.** In the sandbox it's blocked twice over: an un-allowlisted
   binary won't execute under policy, and a static musl `rg` without PCRE2 can't run the patterns
   anyway. For single-repo SEC-bench CVE checkouts (not monorepos) `pathlib.rglob` + `re` is fast
   enough. If a real monorepo ever proves it too slow, adding `rg` means (a) a PCRE2 build and (b) a
   policy `--binary` allowlist entry — revisit then, not now. Deletes the vendoring section and its
   build step.

3. **Prefer in-process library checks over external CLIs in stage 2 — for policy reasons, not just
   tidiness.** Bandit-as-a-library and tree-sitter run under the already-allowlisted venv `python3`.
   **Semgrep-as-a-CLI is a separate binary that needs its own `--binary` allowlist rule** to run in
   the sandbox at all. So Bandit (Python) + tree-sitter (Python) are the primary path; Semgrep stays
   an *optional* upgrade that also requires a policy change, and the regex-in-Python fallback covers
   the case where it isn't allowlisted.

4. **`SkillToolset` provides only `load_skill`/`read_skill_file` — no execution surface.** The skill's
   instructions ("run `scripts/sweep.py`") only work if the *consuming* agent independently carries an
   execution surface (OpenClaw's `tool_search_code` / code-execution, or SEC-bench's `DockerShell`); a
   bare `Agent(tools=skills_toolset)` cannot run anything. State this prerequisite in `SKILL.md`, and
   note that under NemoClaw the scripts execute via the skill's **venv python**, which must be
   allowlisted (it is, by the standard skill-install policy).

5. **Config format is now a minor call, not a hard floor.** With a controlled venv you can pip-install
   PyYAML, so the stdlib-floor argument is moot. JSON is still marginally simpler (no dep, no parser
   ambiguity) and the config is trivial — keep `sinks.json`, but this is preference, not a blocker.

6. **Re-derived scoring silently drops the `security_control` exposure tier.** Source classifies
   `entry-point > security_control (by name) > attack_surface (by body) > reachable > neutral` with
   bases `100 / 40 / 80 / 50 / 10` (note: precedence order ≠ base-score order — a name-matched auth
   function is tagged `security_control` at base 40 *before* its body is checked for sinks). The plan
   lists only `entry point > sink > reachable > neutral`. For a vuln hunt, auth/validation functions
   are exactly where bypass bugs live — either keep the tier or drop it consciously, not by omission.

## Directory layout

```
security-agent-core/
└── skills/
    └── code-triage/
        ├── SKILL.md
        ├── config/
        │   └── sinks.json            # sink/entry-point/security-control patterns (JSON, no PyYAML dep)
        └── scripts/
            ├── common.py             # shared dataclasses + JSON schema, pattern loader
            ├── sweep.py              # stage 1
            └── deepen.py             # stage 2
```

`FileSystemSkillStore("skills/")` in security_agent points at the `skills/` parent; the agent
discovers `code-triage` by its `SKILL.md` frontmatter.

## `SKILL.md`

Frontmatter: `name: code-triage`, description tuned for discovery — e.g. *"Narrow an unfamiliar
repo down to the functions most likely to matter for a vulnerability hunt, before reading anything
in full. Use this before grepping ad hoc or reading files one by one in a repo you haven't explored
yet."*

**Consuming-agent prerequisite (state this in the frontmatter/body):** this skill executes helper
scripts, so the agent loading it must independently carry an execution surface — OpenClaw's
code-execution / `tool_search_code` in the NemoClaw sandbox, or SEC-bench's `DockerShell` — plus the
skill venv's (allowlisted) `python3`. `SkillToolset` itself only exposes `load_skill` /
`read_skill_file`; an agent with `tools=skills_toolset` and nothing else can only read these scripts,
not run them.

Instructions body (what the agent is told to do):
1. Run `scripts/sweep.py <repo_root>` through your shell tool.
2. If it returns candidate files, run `scripts/deepen.py <repo_root> --files <candidates>`.
3. Treat the returned JSON as the answer — do not paste full file contents into context. Use the
   `file`/`line_start`/`line_end` fields to read only what you decide is worth reading in full.
4. If `deepen.py` reports a tool as unavailable (see `warnings` field in its output), proceed anyway
   — the ranking is still usable, just lower-confidence for that language/check.

## `config/sinks.json`

Same categories T3MP3ST hardcodes as regex constants, kept as data instead of code so the list can
grow without touching scripts. JSON rather than YAML is a minor convenience (no PyYAML needed), not a
hard requirement in the NemoClaw venv — see review finding #5.

**Patterns are copied verbatim from the source, look-behind guards included (review finding #1).**
Because stage 1 runs on a single Python `re` backend (no ripgrep), the source's `(?<![\w.])system\(`
/ `(?<![\w.])popen\(` guards work as-is and should be preserved — they were deliberately added to
keep the Python corpus's rankings stable, and dropping them would loosen precision for no benefit.
The example below is abbreviated; the shipped file mirrors `DANGEROUS_SINK_RE` /
`OUTBOUND_REQUEST_RE` / `RISKY_PARAM_RE` exactly.

```json
{
  "sinks": [
    { "label": "subprocess",        "pattern": "subprocess" },
    { "label": "eval",              "pattern": "\\beval\\(" },
    { "label": "pickle_loads",      "pattern": "pickle\\.loads" },
    { "label": "outbound_request",  "pattern": "requests\\.(get|post|put)|fetch\\(|axios[.(]|http\\.(Get|Post)" }
  ],
  "entry_points":      [ { "pattern": "@(app|router)\\.(route|get|post)" }, { "pattern": "^(handle|on_|process_)" } ],
  "security_controls": [ { "pattern": "valid|auth|sanitiz|csrf|check_" } ],
  "risky_params":      [ { "pattern": "url|uri|endpoint|host|addr|path|file|name|id$|_id" } ]
}
```

Full list ported (look-around-stripped) from `code-ingest.ts`'s `DANGEROUS_SINK_RE` /
`OUTBOUND_REQUEST_RE` / `RISKY_PARAM_RE`.

## `scripts/sweep.py` — stage 1

1. Single backend: `pathlib.rglob` + Python `re`, one file at a time. No ripgrep, no backend
   resolution — the sandbox's policy allowlist and the Rust-regex look-behind gap make `rg` more
   trouble than it's worth here (review findings #1, #2). Still emit a `backend_used` field for the
   output contract's stability, hardcoded to `"python-re"` for now.
2. Load `sinks.json`, run every pattern against the repo.
3. Collapse hits to unique files, each with its matched `(line, pattern_label)` list.
4. Emit JSON: `{backend_used, files: [{path, hits: [{line, label}]}]}`.

## `scripts/deepen.py` — stage 2

Input: the file list from stage 1 (or run stage 1 itself if not given one).

1. **Function resolution** — try `tree_sitter` + `tree_sitter_language_pack` per file; on
   ImportError, unsupported language, or a parse error, fail open: keep the raw hit line with
   `function: null` rather than dropping it (same philosophy as T3MP3ST's `ts-parse.ts`).
2. **Sink verification** — for `.py` files, call `bandit` as a library (runs under the allowlisted
   venv python); on ImportError, fall back to re-matching `sinks.json` patterns against just the
   resolved function's body (the T3MP3ST-equivalent check, scoped small instead of whole-repo). For
   everything else, `semgrep` is an *optional* upgrade — but it's an external CLI binary that needs
   its own OpenShell `--binary` allowlist rule to run in the sandbox (review finding #3), so treat it
   as opt-in and default to the Python-regex fallback when it's absent or not allowlisted.
3. **Lazy reachability** — for each flagged function, one or two hops of "who calls this name"
   via the same search backend from stage 1. Cap hop count and total files pulled in, so this can't
   runaway-expand on a monorepo — and when a cap is hit, record it in `warnings` rather than
   truncating silently. (This is a per-candidate grep-based approximation of reachability, not the
   single-pass call graph the source builds; the cap keeps it from going quadratic on large repos.)
4. **Score** — weighted formula, re-derived (not copied) from T3MP3ST's `prioritize()`: exposure
   base + risk-signal count + reachability bonus + risky-param/outbound-request bonus. Keep the full
   exposure ladder including `security_control` (see review finding #6), matching the source's five
   tiers: `entry-point (100) > attack_surface/sink (80) > exposed_internally/reachable (50) >
   security_control (40) > neutral (10)`. Note the source's classification *precedence* (name-based
   `security_control` is decided before body-based `attack_surface`) differs from this base-score
   *order*; re-derive deliberately rather than reproduce that quirk by accident.
5. Emit sorted JSON, capped to top-N.

### Output schema (the contract, stable regardless of which backend ran)

```json
{
  "warnings": ["semgrep not found on PATH, used regex fallback for src/parser.c"],
  "candidates": [
    {
      "file": "src/upload.py",
      "function": "handle_upload",
      "line_start": 120,
      "line_end": 145,
      "sink_matches": ["subprocess", "eval("],
      "is_entry_point": true,
      "hop_distance": 0,
      "ssrf_idor_flag": false,
      "finding": {"tool": "bandit", "test_id": "B602", "severity": "high"},
      "priority_score": 132
    }
  ]
}
```

`finding.tool` is one of `bandit` / `semgrep` / `regex-fallback` — downstream consumers (and anyone
scoring how well this worked) can see when the check was AST-verified vs. a lower-confidence regex
hit, instead of the quality difference being invisible.

## On ripgrep (dropped — review findings #1, #2)

Not used and not vendored. In the NemoClaw sandbox an external `rg` binary is blocked by the policy
allowlist until explicitly permitted, a static musl build without PCRE2 can't run the source's
look-behind patterns, and for single-repo CVE checkouts `pathlib.rglob` + Python `re` is fast
enough. Revisit only if a real monorepo target proves Python too slow — and then it's a PCRE2 build
*plus* an OpenShell `--binary` policy rule, decided at that point.

## Runtime dependencies (NemoClaw venv)

Installed into the skill's venv at setup (the RAG demo's `install.sh` pattern), so they're guaranteed
at runtime rather than probed for: `tree-sitter` + `tree-sitter-language-pack`, `bandit`. `semgrep`
is optional and, being a CLI binary, also needs a policy `--binary` rule — leave it out of the
default install. No network is needed at triage time (only at install time), which fits the sandbox's
locked-down network policy.

## Build order

1. `common.py` + `sinks.json` — shared schema and pattern data (patterns copied verbatim from source,
   look-behind guards intact), nothing runs yet.
2. `sweep.py`, single Python-`re` backend. Pin the JSON output contract here with a fixture repo.
3. `deepen.py` with regex-fallback only (no Bandit/Semgrep) — function resolution + regex-scoped sink
   check + scoring. This alone is already useful and testable.
4. Add Bandit (library, Python targets) as an upgrade to step 3; keep Semgrep out until/unless a
   target needs it and a policy rule is added. Prove the Bandit path optional (test with it absent).
5. `SKILL.md`, wire into a `FileSystemSkillStore`, smoke-test with a Haystack `Agent` that has
   **both** the `SkillToolset` **and** an execution surface — a `tools=skills_toolset`-only agent can
   load the skill but can't run its scripts (finding #4).
6. Point it at 2-3 of SEC-bench's own C/C++ CVE checkouts as the first real test — also the moment to
   sanity-check tree-sitter-language-pack's C/C++ coverage on real targets.
7. Wire into the SEC-bench / OpenClaw agent `tools=` last, once 1-6 are confirmed working standalone.

## Still open (non-blocking, revisit later)

- Priority-score weights are a first guess; expect to tune them against real SEC-bench instances
  rather than get them right analytically up front.

## Resolved by this review (previously "still open")

- **Host vs. container execution** — answered by the deployment target: the skill runs *in-sandbox*
  against the sandbox workspace filesystem, with its scripts and deps in the skill venv. The target
  repo is checked out into the workspace; there's no separate per-CVE container to ship deps into.
