# Code scanner (white-box triage) plan

Supersedes `web-scanner-plan.md` for this effort — that doc covered live-target web recon
(T3MP3ST's `scanner`/WEAPONIZE archetype); this one covers **static source-code triage**
(T3MP3ST's `code-ingest.ts` pipeline), which is what's actually wanted. Left the other doc in
place for reference; delete it if it's not useful.

No code yet — research + decision doc, open-source-only per the brief.

## What's being ported

T3MP3ST's pipeline (`src/recon/code-ingest.ts` + `ts-parse.ts`) does NOT decide "this is a
vulnerability." It answers a narrower question: *given a whole repo, which ~50 functions should
an LLM read first?* Five stages:

1. **Crawl + parse** — walk the repo, extract every function/method/class as a `CodeBlock` (name,
   params, decorators, line range, body). Real tree-sitter ASTs for JS/TS/Go/Java/C/C++; Python
   uses a regex "AST-lite" fallback.
2. **Call graph** — resolve intra-repo callers/callees by name.
3. **Entry points + reachability** — regex-match names/decorators for HTTP-handler shapes
   (`handle*`, `@app.route`, etc.), then BFS from those through the call graph for hop-distance.
4. **Classify** — one exposure label per block, fixed precedence: entry point > name matches a
   security-control pattern (`valid|auth|sanitiz|...`) > body matches a dangerous-sink pattern
   (`eval(`, `subprocess`, `pickle.loads`, ...) or an SSRF/IDOR shape (identifier-param + outbound
   request) > reachable-from-entry-point > neutral.
5. **Prioritize** — a weighted score (exposure base + risk-signal count + reachability bonus +
   SSRF/IDOR bonus) that sorts the shortlist, then a token-budget packer trims it to what fits an
   LLM context window.

Steps 3 and 4 are plain regex, not semantic analysis — no taint tracking, no alias resolution. The
doc-comments in T3MP3ST's own source are explicit that this is a recall-oriented pre-filter, not a
verdict.

## Why this is a specific fit here, not just a generic "nice to have"

`SecBench.run` (`src/security_agent/benchmarks/secbench.py`) hands the coding agent a real CVE
checked out in a Docker container and lets it explore with `DockerShell`/`DockerReadFile`/
`DockerEditFile`/`DockerDebugger` (`container_tools.py`) — there's no pre-ranking today; the agent
finds the vulnerable code by its own exploration (grep, read, guess). SEC-bench's targets are C/C++,
which is exactly one of T3MP3ST's tree-sitter-covered languages. A triage pass that crawls the
checked-out repo *before* (or alongside) the agent's first turn and hands it "here are the 20
highest-exposure functions, ranked" is a plausible way to cut down blind exploration on a benchmark
this project already runs — not a hypothetical use case.

Separately, `PrimeVul` already does function-level classification but on pre-sampled/pre-extracted
functions — it never has to find the function in a whole repo. A repo-scale triage stage is what
would let either benchmark's agent operate on a full checkout instead of a pre-cut snippet, if that
ever becomes a target shape.

## Per-stage recommendation

| Stage | Recommendation | License | Why |
|---|---|---|---|
| Crawl + parse (multi-language AST extraction) | **py-tree-sitter** (`tree-sitter/py-tree-sitter`) + **tree-sitter-language-pack** (`kreuzberg-dev/tree-sitter-language-pack`, 300+ grammars incl. C/C++/Python/JS/TS/Go/Java) | MIT | Direct Python analog of what `ts-parse.ts` does with `web-tree-sitter` (WASM) — native bindings here, no WASM layer needed. Official project, actively maintained. |
| Call graph | Build directly from the tree-sitter parse (same approach T3MP3ST takes: name-resolve calls within extracted blocks) rather than a separate tool | N/A (your code) | Keeps call-graph nodes in the same shape as the blocks you're about to score — bolting on an external call-graph tool means reconciling two different block identities. `code2flow` (`scottrogowski/code2flow`, MIT) is worth a look if a standalone graph *artifact* (not feeding a scorer) is ever wanted — but it only covers Python/JS/Ruby/PHP, missing C/C++/Go/Java. |
| Dangerous-sink / security-control detection | **Bandit** (`PyCQA/bandit`) for Python specifically (real AST-based checks — hardcoded secrets, `subprocess`/`eval`/unsafe YAML/SQL-string-building — strictly more accurate than T3MP3ST's own regex for the languages it covers); **Semgrep Community Edition** (`semgrep/semgrep`) for the multi-language case, including C/C++ | Bandit: Apache-2.0. Semgrep engine: LGPL-2.1; Semgrep-maintained community rules: **Semgrep Rules License v1.0** (free for internal/non-SaaS use — fine here, restricted only for competing SaaS products) | Both are AST/semantic-pattern based, not regex — a real upgrade over `DANGEROUS_SINK_RE`. Semgrep in particular has framework-aware rules (Flask/Django/Express route decorators) that would also strictly improve the entry-point detection T3MP3ST does by bare name/decorator regex. |
| Entry-point detection | Semgrep's framework-specific rulesets (route decorators) where available; regex name-heuristics (`handle*`, `*_handler`) as a fallback for languages/frameworks Semgrep doesn't have a rule for | (see above) | Same reasoning — decorator-aware detection beats name-guessing, but won't have 100% framework coverage, so keep the cheap regex fallback T3MP3ST uses. |
| Reachability + priority scoring | **Roll your own** — this is bespoke "rank for LLM context budget" logic; no existing OSS project does this specific job | N/A (your code) | BFS over your own call graph, weighted scoring tuned to what matters for triage (proximity to entry point, sink density, SSRF/IDOR shape) — same as T3MP3ST's `prioritize()`, just re-derived, not copied line-for-line given the AGPL-3.0 question flagged for T3MP3ST itself. |
| Token-budget packing | Roll your own (or check if Haystack already has a document/context truncation utility that fits — worth checking before writing one) | N/A | Same reasoning; also small enough that reuse risk/benefit is low either way. |

## One tool to explicitly rule out

**CodeQL** is not usable for this unless the target codebase is itself OSS on GitHub — GitHub's
CodeQL license restricts free use to open-source codebases/academic research; analyzing arbitrary
(closed-source, e.g. a private CVE checkout or someone's proprietary repo) code requires a
commercial license. Since triage targets here (SEC-bench CVE checkouts, PrimeVul samples, or any
future arbitrary-repo use) aren't guaranteed to be OSS, don't build around it.

## Suggested architecture (Python)

```
security_agent/
└── triage/                      # new — mirrors code-ingest's stage names, not its code
    ├── crawl.py                 # walk repo -> file list (extIn/excludeGlobs, same knobs as IngestConfig)
    ├── extract.py               # tree-sitter -> CodeBlock-equivalent dataclass per language
    ├── callgraph.py             # name-resolve calls within extracted blocks
    ├── classify.py              # Bandit/Semgrep findings + entry-point rules -> Exposure label
    ├── prioritize.py            # weighted score, sorted shortlist
    └── pack.py                  # token-budget trim for the target LLM context window
```

Each stage's output shape should be a plain dataclass/dict, not a Bandit- or Semgrep-specific
object — that keeps `classify.py` swappable (e.g. add a language Semgrep doesn't cover well without
touching the rest of the pipeline).

## Next steps

1. Confirm SEC-bench is the first real consumer (or decide it's exploratory/library-only for now) —
   that decides whether stage 1 is "crawl a Docker container's filesystem via `DockerReadFile`" or
   "crawl a local clone," which changes `crawl.py`'s I/O layer.
2. Prototype stages 1-2 (crawl/extract/callgraph) against a couple of SEC-bench's own C/C++ CVE
   checkouts — cheapest way to validate tree-sitter-language-pack's C/C++ grammar coverage on real
   targets before investing in classify/prioritize.
3. Wire in Bandit (Python targets) and/or Semgrep (C/C++ and everything else) for stage 3 instead of
   writing sink regexes from scratch.
4. Write `prioritize.py` — this is the one stage that's genuinely novel; expect to iterate on the
   scoring weights against real SEC-bench instances rather than guessing them up front.
5. Decide the actual integration point: a new `Tool` the SEC-bench container-agent calls at the start
   of its run (e.g. `DockerTriageRepo`, following `container_tools.py`'s existing `Docker*` naming),
   vs. a pre-processing pass that runs before the agent even starts and gets injected into its first
   message.

## Sources

- [tree-sitter/py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter) — MIT
- [kreuzberg-dev/tree-sitter-language-pack](https://github.com/kreuzberg-dev/tree-sitter-language-pack) — grammar bundle, 300+ languages
- [scottrogowski/code2flow](https://github.com/scottrogowski/code2flow) — MIT (rewritten from LGPL in 2021); Python/JS/Ruby/PHP only
- [PyCQA/bandit](https://github.com/PyCQA/bandit) — Apache-2.0
- [semgrep/semgrep](https://github.com/semgrep/semgrep) — LGPL-2.1 (engine)
- [Semgrep Rules License v1.0](https://semgrep.dev/legal/rules-license/) — community rules, free for internal/non-SaaS use
- [github/codeql](https://github.com/github/codeql) — free only for OSS/academic use; commercial license needed for closed-source targets
- [Technologicat/pyan](https://github.com/Technologicat/pyan) — Python-only call graph, license unverified, not recommended over the above
