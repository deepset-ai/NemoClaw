# Web scanner tool: build-vs-reuse plan

Goal: give `security_agent` a live web-recon/vuln-scanning tool set equivalent to T3MP3ST's
`scanner` archetype (WEAPONIZE phase) — header/CSP/cookie/CORS/clickjacking/open-redirect/HTTP-method
checks, tech fingerprinting, a TCP port scan, a TLS scan, and (optionally) a `nuclei` pass — without
rewriting what a maintained open-source project already does well.

No code yet. This is the research + decision doc; see "Next steps" for what happens after a license
posture is picked.

## Where T3MP3ST's version stands, for reference

13 of its 14 scanner-archetype tools are dependency-free — plain `fetch`/`tls.connect`/raw TCP socket
plus string/regex analysis, no external binary. Only `nuclei_scan` shells out to a CLI. That's *why*
a from-scratch Python port is a viable option here, not just "reuse a library because writing it is
hard" — writing it is not hard, the question is purely whether an existing OSS project already covers
the check more thoroughly (real bypass techniques for CORS, real CVE checks for TLS) than a from-scratch
version would.

## Per-check recommendation

| Check(s) | Recommendation | License | Why |
|---|---|---|---|
| Security headers, CSP, cookies, HSTS, redirects, CAA, SRI | **Mozilla HTTP Observatory** (`mozilla/http-observatory`) | MPL-2.0 (weak/file-level copyleft) | One library covers 4+ of our checks at once, more thorough than our own header/CSP/cookie regex would be (SRI/CAA/redirect checks we don't have at all), actively maintained by Mozilla. |
| Clickjacking (`X-Frame-Options` / CSP `frame-ancestors`) | Same as above — HTTP Observatory's header analyzer already flags this | MPL-2.0 | No separate tool needed; don't add a second dependency for one header check. |
| CORS misconfiguration | **Corsy** (`s0md3v/Corsy`) or `chenjj/CORScanner` | GPL-3.0 | Purpose-built, covers known CORS bypass techniques (reflected origin, null origin, subdomain trust, etc.) that are easy to under-cover in a quick from-scratch check. |
| Open redirect, HTTP methods (`OPTIONS`/`TRACE`), API endpoint discovery | **Roll your own** (stdlib `requests`/`httpx`, ~50-100 lines each) | N/A (your code) | No mature, narrowly-scoped OSS project stands out here; T3MP3ST's own versions of these are this size. `s0md3v/Arjun` (GPL-3.0) or `maurosoria/dirsearch` (GPL-3.0) are options if endpoint/parameter discovery needs to be more thorough than a static wordlist. |
| Technology / version fingerprinting | Community Wappalyzer forks — `enthec/webappanalyzer` or `tunetheweb/wappalyzer` (fingerprint data) + `s0md3v/wappalyzer-next` (Python-usable) | GPL-3.0 | Original Wappalyzer went closed-source/commercial in 2023; these forks continue from the last public GPL-3.0 fingerprint set. Verify the specific fork's LICENSE file before use — provenance here is a bit patchworked. |
| Port scan (open/closed TCP) | **Roll your own** — Python stdlib `socket.connect_ex` + `asyncio`/thread pool | N/A (your code) | Same shape as T3MP3ST's own version (~60 lines, no dependency). Only reach for `python-nmap` if you specifically want nmap's service/version fingerprinting and are fine requiring the `nmap` binary. |
| TLS/SSL — lite (cert expiry, self-signed, weak key/cipher/protocol) | **Roll your own** — Python stdlib `ssl` + `socket` | N/A (your code) | Same shape as T3MP3ST's own version (~50 lines). Zero dependency, zero license question. |
| TLS/SSL — deep (Heartbleed, ROBOT, CCS-injection, full Mozilla-config compliance) | **sslyze** (`nabla-c0d3/sslyze`) | **AGPL-3.0** | Battle-tested, but same copyleft tier as T3MP3ST itself — pulling it in raises the identical distribution-time question you were already weighing about T3MP3ST's own code. Only add this tier if the deep CVE-style checks are actually wanted; the lite tier covers what T3MP3ST's own scanner does. |
| Nuclei pass (template-based vuln scanning) | Subprocess the `nuclei` binary directly (same as T3MP3ST) — **skip `PyNuclei`** | MIT (nuclei itself) | `PyNuclei` is a small unofficial wrapper with low adoption around a CLI whose interface changes; a direct `subprocess.run(["nuclei", "-target", ..., "-jsonl"])` + JSON-lines parse (~20 lines) is more robust and is exactly what T3MP3ST does. Optional tool — skip entirely if you don't want the extra binary dependency. |

## License posture — the actual decision to make first

Everything above sorts into three tiers:

1. **No license question at all** — open-redirect, HTTP-methods, API-endpoint-discovery, port scan,
   TLS-lite. Pure stdlib, your own code, ship it regardless of what else you decide.
2. **Permissive-ish, low friction** — HTTP Observatory (MPL-2.0, file-level copyleft: modifying *their*
   files requires sharing those changes back, but linking/importing it from your own code doesn't
   spread), nuclei binary (MIT).
3. **Copyleft, needs a decision** — Corsy / Arjun / dirsearch / Wappalyzer forks (GPL-3.0), sslyze
   (AGPL-3.0). Vendoring or importing GPL/AGPL code into `security_agent` would pull its copyleft terms
   onto however you distribute the combined result — same category of question already flagged for
   borrowing from T3MP3ST itself. Worth deciding once, for all of them, rather than per-library.

Recommendation: start with tier 1 + HTTP Observatory (tier 2) — that alone covers headers, CSP,
cookies, clickjacking, open redirect, HTTP methods, endpoint discovery, port scan, and TLS-lite, with
zero or minimal copyleft exposure. Add a tier-3 dependency only for the specific checks where a
from-scratch version would be genuinely weaker (CORS bypass coverage, deep TLS CVE checks,
tech-fingerprint breadth) — and decide that with eyes open to what GPL/AGPL means for how
`security_agent` gets distributed.

## Fitting into security-agent-core's existing shape

Per the project's own convention (`README.md`): the agent's tools are Python components wired through
a seed YAML (`seeds/<name>.yaml`), same pattern `container_tools.py` uses for the SEC-bench tool set.
A new `security_agent/web_scan_tools.py` (or similar) would follow that pattern once implementation
starts — each check as a Haystack `Tool`-wrapped function, registered in a new seed rather than bolted
onto `primevul`/`secbench`'s existing seeds.

Separately: neither existing benchmark (`primevul` = function classification, `secbench` = Docker CVE
patching) fits "did the scan find the right live-web issue." Whether this needs its own `Benchmark`
(ground-truth vulnerable targets + scoring, so the optimizer can hill-climb it like `primevul`/`secbench`)
or ships as a plain agent capability with no eval harness yet is a separate decision from "does the
tool exist" — flagged here, not resolved.

## Next steps (once ready to write code)

1. Pick the license posture (permissive-only vs. willing to take on GPL/AGPL deps) — determines which
   rows above are actually available.
2. Build the zero-dependency tier first (open redirect, HTTP methods, endpoint discovery, port scan,
   TLS-lite) — no license blocker, and it mirrors what already works in T3MP3ST.
3. Wire in HTTP Observatory for headers/CSP/cookies/clickjacking.
4. Only then decide on the copyleft-tier additions (CORS, tech fingerprinting, deep TLS, nuclei) —
   per-library, now that the boundary is visible.
5. Wrap the chosen set as Haystack `Tool`s, register in a new seed YAML.
6. Decide benchmark-or-not for the new capability.

## Sources

- [mozilla/http-observatory](https://github.com/mozilla/http-observatory) — MPL-2.0
- [nabla-c0d3/sslyze](https://github.com/nabla-c0d3/sslyze) — AGPL-3.0 ([licensing clarification issue](https://github.com/nabla-c0d3/sslyze/issues/198))
- [s0md3v/Corsy](https://github.com/s0md3v/Corsy) — GPL-3.0
- [s0md3v/Arjun](https://github.com/s0md3v/Arjun) — GPL-3.0
- [maurosoria/dirsearch](https://github.com/maurosoria/dirsearch) — GPL-3.0
- [enthec/webappanalyzer](https://github.com/topics/wappalyzer-alternative), [s0md3v/wappalyzer-next](https://github.com/s0md3v/wappalyzer-next) — GPL-3.0 (fingerprint data provenance from pre-2023 Wappalyzer)
- [projectdiscovery/nuclei](https://github.com/projectdiscovery/nuclei) — MIT
- [kushvaibhav/PyNuclei](https://github.com/kushvaibhav/PyNuclei) — unofficial wrapper, not recommended (see table)
- [EONRaider/Simple-Async-Port-Scanner](https://github.com/EONRaider/Simple-Async-Port-Scanner) — reference only; recommendation is to roll a stdlib version instead
