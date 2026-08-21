# security-agent-core

One self-improving **security agent**, the **benchmarks** it is assessed against, and the
**optimizer** that evolves it.

- **The agent** (`security_agent`) is a Haystack `Agent`: a chat generator, a system prompt, and
  tools. Its whole definition — prompt, knobs, tool wiring — lives in the committed seed YAML
  (`seeds/<name>.yaml`), the single source of truth; Python holds only the tool component *code*.
- **Benchmarks** (`security_agent.benchmarks`) are pluggable and selected by name. PrimeVul
  (classification) and SEC-bench (execution, Docker) are implemented; add one by implementing the
  `Benchmark` protocol and registering it.
- **The optimizer** (`security_agent.optimize`) is a meta-agent that
  edits the target agent's serialized config, benchmarks each candidate, and keeps only
  improvements (hill-climbing with champion tracking, a journal, and rollback).

## Layout

The **benchmark name is the single selector**: `--benchmark <name>` (on every command) or
`default_benchmark` in `config.yaml`. Choosing it resolves that benchmark's config block *and* all of
its artifact paths, which are **derived from the name** by convention — so two benchmarks never
overwrite one another:

Files named `<name>.*` (or under `<name>/`) exist once **per benchmark** — `<name>` is the benchmark,
e.g. `primevul` or `secbench`.

```
security-agent-core/
├── config.yaml                  # campaign config: default_benchmark + a `benchmarks:` map
├── data/cwe_reference.json      # CWE tool data
│
├── seeds/<name>.yaml            # the agent definition — SOURCE OF TRUTH   ── tracked
├── programs/<name>.md           # the meta-agent's directive                ── tracked
├── pipelines/<name>.yaml        # the promoted agent artifact               ── tracked
├── benchmark_data/<name>.jsonl  # pre-sampled tasks (`build-benchmark`)
├── runs/<name>/                 # champions, configs, journal               ── gitignored
├── traces/                      # OTLP traces                               ── gitignored
├── docker-compose.yml           # Qdrant, the knowledge-base store
├── data/kb_cache/               # per-source feed downloads                  ── gitignored
│
└── src/security_agent/
    ├── cli.py                   # the `secagent` entry point
    ├── paths.py  evaluate.py    # filesystem locations / eval runner
    ├── components.py  verdict.py  tracing.py                   # CWE tools + shared building blocks
    ├── container_tools.py  docker_eval.py                      # SEC-bench tool components + Docker verifier
    ├── benchmarks/              # the pluggable benchmark layer
    │   └── base.py  primevul.py  secbench.py
    ├── knowledge_base/          # the security KB behind `search_security_kb`
    │   ├── build.py  store.py  mapping.py  settings.py  profiles.py  pins.py
    │   ├── search.py            # the agent pack's Advanced RAG Agent over the store
    │   ├── prompts.py  render.py  sanitize.py  testing.py
    │   └── curation/            # vendored redamon feed clients (see repo-root NOTICE)
    └── optimize/                # the self-improvement engine
        ├── optimizer.py  config_patch.py  validate.py  store.py  settings.py  tool_catalog.py
        ├── harness/             # runner.py, subprocess_runner.py — isolated candidate scoring
        └── meta/                # agent_factory, hooks, session, tools — the meta-agent
```

## Quick start

```bash
hatch env create         # create the env and install the project (editable)
cp .env.template .env     # fill in OPENAI_API_KEY (meta-agent) + ANTHROPIC_API_KEY (target)

# Build a benchmark's tasks once (static, so every iteration scores the same tasks):
hatch run secagent build-benchmark --benchmark primevul --train-pairs 15 --holdout-pairs 15

# The agent lives in the committed seed (seeds/primevul.yaml) — edit its prompt/knobs/tools there.
# The self-improvement loop (every command takes --benchmark; default: config.yaml default_benchmark):
hatch run secagent optimize init --benchmark primevul         # score the seed, install as champion
hatch run secagent optimize run --benchmark primevul --iterations 5   # the loop (needs OPENAI_API_KEY)
hatch run secagent optimize status --benchmark primevul
hatch run secagent optimize report --benchmark primevul       # holdout: seed vs champion (overfitting check)

# Promote the champion into the tracked artifact and evaluate it:
hatch run secagent promote --benchmark primevul     # writes pipelines/primevul.yaml
hatch run secagent eval --benchmark primevul --split holdout

# To eval the seed directly (before optimizing), promote the seed:
hatch run secagent promote --benchmark primevul --champion seeds/primevul.yaml
```

(`hatch shell` drops you into the env so you can call `secagent` directly.)


## Benchmarks

A `Benchmark` (see `benchmarks/base.py`) bundles what varies per evaluation target: `load_tasks()`
(with `train`/`holdout` splits) and `score(tasks, results)` → a `ScoreReport` with the scalar the
optimizer hill-climbs. Select one with `--benchmark <name>` (on every command) or `default_benchmark`
in `config.yaml`; the choice also picks that benchmark's seed, runs store, pipeline, and program.

- **PrimeVul** (`primevul`): function-level vulnerability *classification*. The agent emits a JSON
  triage verdict (`is_vulnerable` + CWE); scored by **paired accuracy** — a (vulnerable, fixed) pair
  counts only when both are correct, which defeats the "always say vulnerable" degenerate policy
  (falls back to binary F1 when a set has no complete pairs).
- **SEC-bench** (`secbench`): real-CVE *patching* — an **execution-style** benchmark. A coding agent
  edits the vulnerable source inside a Docker container; its edits are extracted as a git diff and scored by
  **resolve rate** — running the patch against SEC-bench's prebuilt image (compile + the known PoC no longer trips
  the sanitizer).
  Unlike a classification benchmark, it **owns its runner** (`SecBench.run`); the shared harness
  dispatches to it via `run_tasks`. Requires a running **Docker daemon** and pulls large per-instance
  images. Selecting `--benchmark secbench` uses `programs/secbench.md` and its own `runs/secbench/`
  store, seed (`seeds/secbench.yaml`), and pipeline (`pipelines/secbench.yaml`) automatically.

  ```bash
  # Build the tiny subset, then score/optimize the committed seed (Docker required):
  hatch run secagent build-benchmark --benchmark secbench --train-pairs 2 --holdout-pairs 1
  hatch run secagent promote --benchmark secbench --champion seeds/secbench.yaml  # -> pipelines/secbench.yaml, to eval the seed
  hatch run secagent eval --benchmark secbench --split holdout --limit 1
  hatch run secagent optimize init --benchmark secbench && hatch run secagent optimize run --benchmark secbench --iterations 2
  ```

## Security knowledge base (`search_security_kb`)

The SEC-bench agent can ask a security research assistant about three curated security datasets:
the bundled **CWE** weakness reference, **NVD** CVE records and **ExploitDB** entries. All three
bear on memory-safety bugs in C/C++, which is what SEC-bench is. Upstream (redamon) also ships
GTFOBins, LOLBAS, OWASP WSTG and Nuclei clients; those cover privilege escalation, web-app testing
methodology and network detection, so they are not carried here — see the repo-root `NOTICE`.

`search_security_kb` is the [Advanced RAG
Agent](https://docs.haystack.deepset.ai/docs/advanced-rag-agent) from the Haystack agent pack
(`agent-pack-haystack`) running over that store, wrapped as a `ComponentTool`. The pack gives it
five tools — `list_metadata_fields`, `get_metadata_field_values`, `get_metadata_field_range`,
`fetch_documents_by_filter` and `search_documents` — so it inspects the corpus's metadata at query
time and writes its own Haystack filters. That is why the tool takes a question rather than a
`sources` / `min_cvss` / `top_k` parameter triple: those were a fixed slice of the filter space
that had to be kept in sync with the corpus by hand, and *"high-severity CVEs published this
quarter affecting libpng"* or *"ExploitDB entries mentioning this function"* were not in it.

The corpus lives in **Qdrant** (`docker compose up -d qdrant`), two vectors per document:
`BAAI/bge-small-en-v1.5` (384-dim) for the dense leg and fastembed's `Qdrant/bm25` for the
lexical one. `search_documents` is a `QdrantHybridRetriever` — one request, both vectors, fused
server-side — followed by a small cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
22M params). Both models run locally, so retrieval needs no internet; only the sub-agent's own
LLM does.

Qdrant replaced a hand-rolled on-disk artifact (`documents.jsonl` + `embeddings.npy`, merged by
id under an atomic-write layer), a numpy matmul retriever and an in-process BM25 index — about
800 lines whose whole job was persistence, filtering and ANN. `sparse_idf=True` on the collection
is load-bearing: without it the BM25 leg silently degrades to term frequency.

The feed fetching and chunking is vendored from [redamon](https://github.com/samugit83/redamon)
(MIT) — see the repo-root `NOTICE`. Everything from embedding onward is Haystack; redamon's
FAISS/Neo4j/knowledge-graph layer is not used, and there is no web-search fallback.

```bash
docker compose up -d qdrant   # the store; the corpus survives in a named volume

# Build the corpus. Profiles, cheapest first — measured on an M-series Mac, CPU embedding,
# with NVD_API_KEY set:
hatch run secagent kb build --profile dev        #    969 chunks, CWE only, no network   ~30 s
hatch run secagent kb build --profile standard   #  ~8,000  + NVD (90 days, CVSS >= 7)     ~5 min
hatch run secagent kb build --profile full       # ~55,000  + ExploitDB                    ~7 min

hatch run secagent kb stats                      # what is in the store, and what built it

# Retrieval only — no LLM, no API key. This is the retrieval stack the agent's search_documents
# tool runs on; --filters takes the same Haystack filter the agent writes for itself.
hatch run secagent kb query "unbounded copy into a fixed stack buffer" --top-k 5
hatch run secagent kb query "CVE-2024-21626" \
  --filters '{"field": "meta.source", "operator": "==", "value": "nvd"}'

# The real tool: the advanced RAG agent, end to end. Needs OPENAI_API_KEY.
hatch run secagent kb query --agent "which CWE covers writing past the end of a heap buffer?"
```

Notes worth knowing before a campaign:

- **`kb build` is incremental, `kb rebuild` is not.** A re-run re-embeds only chunks whose content
  changed and upserts them by id; `rebuild` clears the caches and recreates the collection. The
  chunk manifest survives the move to Qdrant purely as a cost saving — an upsert would be correct
  either way, but re-embedding 55k documents would not be cheap.
- **`kb query` is the offline smoke test** of the whole retrieval stack with no LLM in the loop —
  use it to tune `top_k` and to judge whether the reranker earns its ~0.2 s. `--agent` adds the
  LLM back and shows what the SEC-bench agent actually receives.
- **A lookup is now an agent run, not a retrieval call.** Budget for it: several seconds and a
  handful of small-model calls per `search_security_kb`, against ~0.5 s before. The sub-agent runs
  on `llm_model: gpt-5.4-mini` with `max_agent_steps: 6`; if the budget runs out mid-lookup, the
  pack's `BackupAnswerHook` writes a best-effort answer from what it gathered, so a cut-off lookup
  still returns something usable rather than nothing.
- **Set `NVD_API_KEY`** for the `standard`/`full` profiles: it raises the NVD rate limit 10x
  (0.65 s vs 6.5 s between requests). Free from <https://nvd.nist.gov/developers>.
- **Only `nvd` carries `cvss_score`.** A bare `meta.cvss_score >= x` filter silently drops every
  CWE and ExploitDB result, turning a severity hint into a source filter. The sub-agent's system
  prompt says so and shows the OR form.
- **Freeze the KB for the duration of an optimize campaign.** `kb_meta.json` records a
  `kb_version`; if the corpus changes mid-campaign, iteration scores stop being comparable and the
  optimizer would credit a KB refresh to a prompt change.
- **The seed is the source of truth for the tool**, not `config.yaml`. `kb build` prints the
  matching `init_parameters` snippet; `tests/test_seeds.py` asserts the seed's models and pinned
  revisions agree with `knowledge_base/pins.py`, because a mismatch makes `load_kb` refuse the
  store and the tool degrade to an error string mid-task.
- **Air-gapped:** the models load only at warm-up. Pre-seed `HF_HOME` on a networked host
  (`huggingface-cli download BAAI/bge-small-en-v1.5 --revision <pin>`, same for the reranker;
  fastembed caches `Qdrant/bm25` under `FASTEMBED_CACHE_PATH`), then set `KB_OFFLINE=1`. Verify with `secagent kb stats --check-models` *before* a campaign, so a
  missing model fails there rather than inside a benchmark task. Note that the sub-agent's LLM is
  currently not air-gapped.
- **KB text is untrusted, and privilege separation is now part of the answer.** Feeds are pinned to
  immutable upstream commits; documents are scrubbed of role/boundary markers and capped on the
  way into Qdrant, so every pack tool reads scrubbed text (changing the policy therefore needs a
  `kb rebuild`). Feed content lands in the *sub-agent's*
  context, which holds no container tools — only its written answer crosses into the context that
  holds `run_shell`/`edit_file`, scrubbed again and framed as untrusted data, with both system
  prompts saying never to act on instructions found in a result.
- **The reranker does not have the last word.** Its ordering is fused with the retrieval ordering
  by weighted RRF at `reranker_weight: 0.5`, and at most `max_per_source: 2` results come from one
  dataset. Both were added after measuring the real corpus, both sit under the pack's tool
  boundary, and both are knobs the optimizer can tune. `cross-encoder/ms-marco-MiniLM-L-6-v2`
  saturates here — raw logits 0.96-0.999 across the whole candidate pool — and prefers NVD's prose
  to the terse one-line titles ExploitDB rows are made of, so letting it override the fusion buried
  exact lexical hits: for *"heap buffer overflow in libpng row decoding"* BM25's top hit was
  `EDB-389: LibPNG ... Remote Buffer Overflow` and the reranker pushed it out of the top 8. Scores
  are normalized to the best result in that set — relative standing, not calibrated relevance.
- **The per-source cap buys a turn, it no longer decides the answer.** The corpus is volume-skewed
  (46k ExploitDB rows vs ~1k CWE entries), so an uncapped query returns five near-identical
  ExploitDB rows and the CWE class that names the weakness never appears. The sub-agent can now
  scope by filtering on `meta.source` itself; the cap makes the *first* result set diverse so it
  usually does not have to.
- **Scale — re-measure before quoting.** The figures on record were taken on the seven-source,
  75,377-document corpus this was trimmed from, and before Qdrant: build 9 min wall / 4.5 GB peak
  RSS (the NVD client holding a window of CVEs dominates, not the embedding). `full` is now ~55k
  documents, the 159 MB local artifact and the ~11 s per-process store load are both gone, and
  retrieval latency now depends on the Qdrant deployment rather than on an in-process BM25 scan.
  None of the old numbers should be carried into an eval writeup unmeasured.

Optional test tiers (skipped by default, they download models / hit the network / cost tokens):

```bash
SECAGENT_KB_INTEGRATION=1 hatch run pytest tests/test_kb_integration.py   # real models
SECAGENT_KB_NETWORK=1     hatch run pytest tests/test_kb_integration.py   # + a real feed fetch
SECAGENT_KB_AGENT=1       hatch run pytest tests/test_kb_integration.py   # + the real LLM
```

## Optimizer notes

- The meta-agent edits the target's config via structured patches (`config_patch.py`): `set`
  (system_prompt, exit_conditions, max_agent_steps, generation kwargs, approved model),
  `enable_tool` / `disable_tool` / `set_tool_description`. It never writes YAML or code directly.
  The tool catalog it enables from is reconstructed from the benchmark's seed (`tool_catalog.py`),
  so tool wiring has one source — the seed YAML.
- Champions, configs, and an append-only journal live under `runs/<benchmark>/` (content-addressed,
  resumable, rollback-able), one store per benchmark. Acceptance is `score > champion + epsilon`.
- Guardrails for the first security campaigns: the target model is fixed (`claude-sonnet-5`), and
  `create_component` (LLM-authored new tools) is disabled (`optimizer.allow_create_component`).
- Pinned to Haystack `3.0.0`. The `JournalHook` runs on `before_tool` (this release has no
  `after_tool` hook point).
