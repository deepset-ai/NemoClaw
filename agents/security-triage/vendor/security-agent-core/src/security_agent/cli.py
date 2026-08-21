"""Unified `secagent` CLI for the security agent.

Subcommands:
  build-benchmark   pre-sample a benchmark's tasks into benchmark_data/<name>.jsonl
  eval              run the agent over a benchmark split and print metrics + traces
  optimize ...      the self-improvement loop (init, run, status, report, rollback)
  promote           write the champion config into the benchmark's pipeline (pipelines/<name>.yaml)
  kb ...            build, inspect and query the security knowledge base (build, rebuild, stats,
                    query) that backs the agent's search_security_kb tool. Needs Qdrant:
                    `docker compose up -d qdrant`

Every command takes `--benchmark <name>` (default: `default_benchmark` in config.yaml). The
benchmark is the single selector: it resolves the whole config block plus all per-benchmark
artifact paths (seed, runs store, promoted pipeline, meta-agent program).

The meta-agent (optimize run) uses OpenAI (OPENAI_API_KEY); the target agent uses
Anthropic (ANTHROPIC_API_KEY). Both are loaded from the project-root .env.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from security_agent import paths
from security_agent.optimize.settings import Settings, load_settings
from security_agent.optimize.store import RunStore


def _store(settings: Settings) -> RunStore:
    return RunStore(settings.path(settings.runs_dir))


# --------------------------------------------------------------------------- #
# build-benchmark
# --------------------------------------------------------------------------- #
def cmd_build_benchmark(settings: Settings, args) -> int:
    from security_agent.benchmarks import get_benchmark

    name = settings.benchmark
    bench = get_benchmark(name)
    if not hasattr(bench, "build"):
        raise SystemExit(f"Benchmark '{name}' has no builder.")
    out = bench.build(train_pairs=args.train_pairs, holdout_pairs=args.holdout_pairs, seed=args.seed)
    n = sum(1 for line in Path(out).read_text().splitlines() if line.strip())
    print(f"Wrote {n} tasks to {out}")
    return 0


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def _summarize_traces(trace_file: Path) -> None:
    if not trace_file.exists():
        print("No trace file was written.")
        return
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for line in trace_file.read_text().splitlines():
        if not line.strip():
            continue
        for rs in json.loads(line).get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    by_trace[span["traceId"]].append(span)
    total = sum(len(v) for v in by_trace.values())
    print(f"\n=== traces: {len(by_trace)} runs, {total} spans -> {trace_file} ===")


def cmd_eval(settings: Settings, args) -> int:
    from haystack import Pipeline
    from haystack.dataclasses import ChatMessage

    from security_agent import tracing
    from security_agent.benchmarks import get_benchmark, split as split_tasks
    from security_agent.benchmarks.base import TaskResult
    from security_agent.optimize.validate import register_allowlist

    name = settings.benchmark
    bench = get_benchmark(name)
    tasks = bench.load_tasks()
    if args.split != "all":
        tasks = split_tasks(tasks, args.split)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit(f"No tasks for benchmark={name} split={args.split}.")

    pipe_path = Path(args.pipeline) if args.pipeline else settings.path(settings.pipeline)
    if not pipe_path.exists():
        raise SystemExit(
            f"No pipeline at {pipe_path}. Run `secagent promote` first (promotes the champion; "
            f"pass `--champion {settings.seed_config}` to promote the seed directly)."
        )

    provider = tracing.setup_tracing(str(paths.TRACE_FILE))
    register_allowlist()

    print(f"Evaluating {pipe_path.name} on {name} (split={args.split}, {len(tasks)} tasks)\n")
    results: list[TaskResult] = []
    if hasattr(bench, "run"):
        # Execution-style benchmark (e.g. SEC-bench): it owns its runner. Hand it the serialized
        # Agent config extracted from the pipeline artifact; it drives its own sandbox and scores.
        import yaml

        pipe_doc = yaml.safe_load(pipe_path.read_text()) or {}
        agent_config = (pipe_doc.get("components") or {}).get("agent")
        if agent_config is None:
            raise SystemExit(f"{pipe_path} has no 'agent' component for benchmark '{name}'.")
        results = bench.run(agent_config, tasks, settings)
        for i, r in enumerate(results, 1):
            err = f"  ERROR={r.error}" if r.error else ""
            print(f"[{i}/{len(tasks)}] {r.task_id}{err}")
    else:
        pipe = Pipeline.loads(pipe_path.read_text())
        for i, task in enumerate(tasks, 1):
            try:
                out = pipe.run({"agent": {"messages": [ChatMessage.from_user(task.question)]}})
                agent_out = out.get("agent", {}) or {}
                last = agent_out.get("last_message")
                answer = (getattr(last, "text", None) if last is not None else "") or ""
                results.append(TaskResult(task_id=task.id, answer=answer, steps=agent_out.get("step_count")))
                err = ""
            except Exception as exc:  # noqa: BLE001 - a task failure must not abort the eval
                results.append(TaskResult(task_id=task.id, answer="", error=f"{type(exc).__name__}: {exc}"))
                err = f"  ERROR={type(exc).__name__}"
            print(f"[{i}/{len(tasks)}] {task.id}{err}")

    provider.force_flush()
    report = bench.score(tasks, results, step_cost_lambda=settings.step_cost_lambda)
    print(f"\n=== {name} metrics ===")
    print(json.dumps(report.metrics, indent=2))
    print(f"\nscore={report.score:.4f}  mean_steps={report.mean_steps:.1f}")
    _summarize_traces(paths.TRACE_FILE)
    return 0


# --------------------------------------------------------------------------- #
# optimize ...
# --------------------------------------------------------------------------- #
def cmd_optimize(settings: Settings, args) -> int:
    from security_agent import tracing
    from security_agent.optimize import optimizer

    action = args.action
    store = _store(settings)

    # init/run execute agents in-process (the meta-agent, and the SEC-bench target agent during
    # scoring) — capture their spans to a dedicated per-benchmark file so campaigns are observable
    # (status/report/rollback don't run agents, so they skip tracing).
    provider = None
    if action in ("init", "run"):
        trace_file = paths.optimize_trace_file(settings.benchmark)
        print(f"Tracing this run to {trace_file}")
        provider = tracing.setup_tracing(str(trace_file))

    try:
        return _cmd_optimize_dispatch(action, store, settings, args, optimizer)
    finally:
        if provider is not None:
            provider.force_flush()


def _cmd_optimize_dispatch(action, store, settings, args, optimizer) -> int:
    if action == "init":
        champion = optimizer.ensure_champion(store, settings)
        print(f"Champion: {champion['hash']} score={champion['score']:.3f}")
        return 0

    if action == "run":
        if args.no_meta:
            champion = optimizer.ensure_champion(store, settings)
            benchmark = optimizer.get_bench(settings)
            report = optimizer.score_config(
                store.load_champion_config(),
                optimizer.load_train_tasks(settings, benchmark),
                settings,
                benchmark,
            )
            print(report.summary())
            return 0
        optimizer.run_campaign(store, settings, iterations=args.iterations)
        return 0

    if action == "status":
        champion = store.champion()
        if champion is None:
            print("No champion yet. Run `secagent optimize init`.")
            return 1
        print(f"Champion: {champion['hash']} score={champion['score']:.3f} metrics={champion['metrics']}")
        for record in store.read_journal()[-args.last :]:
            line = {k: record.get(k) for k in ("type", "iteration", "hash", "score", "accepted", "reason") if k in record}
            timings = record.get("timings") or []
            secs = sum((t.get("agent_s") or 0) + (t.get("verify_s") or 0) for t in timings)
            if secs:
                line["cost_s"] = round(secs)
            print(json.dumps(line, default=str))
        return 0

    if action == "report":
        print(optimizer.holdout_report(store, settings))
        return 0

    if action == "rollback":
        target = args.hash
        scored = [r for r in store.read_journal() if r.get("hash") == target and r.get("score") is not None]
        if not scored:
            print(f"No journal record with a score for config '{target}'.", file=sys.stderr)
            return 1
        store.load_config(target)  # fail early if the config file is missing
        score = scored[-1]["score"]
        store.set_champion(target, score, {"rolled_back": True})
        store.append_journal({"type": "rollback", "hash": target, "score": score})
        print(f"Champion set to {target} (score {score}).")
        return 0

    raise SystemExit(f"Unknown optimize action '{action}'.")


# --------------------------------------------------------------------------- #
# promote
# --------------------------------------------------------------------------- #
def cmd_promote(settings: Settings, args) -> int:
    import yaml
    from haystack import Pipeline

    from security_agent.optimize.validate import load_agent

    champ = Path(args.champion) if args.champion else settings.path(settings.runs_dir) / "champion.yaml"
    out = Path(args.out) if args.out else settings.path(settings.pipeline)
    if not champ.exists():
        raise SystemExit(f"No champion at {champ}. Run `secagent optimize init` first.")
    agent = load_agent(yaml.safe_load(champ.read_text()))
    pipe = Pipeline()
    pipe.add_component("agent", agent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pipe.dumps())
    print(f"Promoted {champ} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# kb — the security knowledge base backing the search_security_kb tool
# --------------------------------------------------------------------------- #
def cmd_kb(settings: Settings, args) -> int:
    """Build, inspect and query the security knowledge base.

    Deliberately not `parents=[common]`: the knowledge base is benchmark-independent. `settings`
    is accepted for signature uniformity with the other commands and is unused — KB settings come
    from the `knowledge_base:` block via `knowledge_base.settings.load_kb_settings`.
    """
    from security_agent.knowledge_base.settings import load_kb_settings

    kb_settings = load_kb_settings(
        args.project.resolve(),
        profile=getattr(args, "profile", None),
        device=getattr(args, "device", None),
        batch_size=getattr(args, "batch_size", None),
        st_batch_size=getattr(args, "st_batch_size", None),
        nvd_lookback_days=getattr(args, "nvd_days", None),
        nvd_min_cvss=getattr(args, "nvd_min_cvss", None),
        offline=True if getattr(args, "offline", False) else None,
        qdrant_url=getattr(args, "qdrant_url", None),
        qdrant_index=getattr(args, "qdrant_index", None),
    )

    if args.action in ("build", "rebuild"):
        from security_agent.knowledge_base.build import build_kb, format_stats

        if args.verbose:
            import logging

            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        stats = build_kb(
            kb_settings,
            profile=args.profile,
            source=args.source,
            rebuild=args.action == "rebuild",
            nvd_api_key=args.nvd_key,
        )
        print(format_stats(stats))
        if stats.get("total"):
            print("\nMatching seed snippet (keep seeds/<benchmark>.yaml in sync with this build):")
            print(_kb_seed_snippet(kb_settings))
        # Non-zero on a partial build so scripts and CI notice a missing source.
        return 1 if stats["errors"] else 0

    if args.action == "stats":
        from security_agent.knowledge_base.store import KbStoreError, read_meta

        try:
            meta = read_meta(settings=kb_settings)
        except KbStoreError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"store:     {kb_settings.qdrant_url}/{kb_settings.qdrant_index}")
        for key in ("kb_version", "built_at", "profile", "embedding_model", "embedding_revision",
                    "sparse_model", "dim", "device", "reranker_model", "total"):
            if key in meta:
                print(f"{key + ':':<18} {meta[key]}")
        for source, count in sorted((meta.get("counts") or {}).items()):
            print(f"  {source:<12} {count}")
        if args.check_models:
            return _kb_check_models(kb_settings, meta)
        return 0

    if args.action == "query":
        from security_agent.knowledge_base import render
        from security_agent.knowledge_base.search import SecurityKbSearch
        from security_agent.knowledge_base.store import KbStoreError

        search = SecurityKbSearch(
            qdrant_url=kb_settings.qdrant_url,
            qdrant_index=kb_settings.qdrant_index,
            embedding_model=kb_settings.embedding_model,
            query_prefix=kb_settings.query_prefix,
            sparse_model=kb_settings.sparse_model,
            reranker_model=kb_settings.reranker_model,
            device=kb_settings.device,
            top_k=args.top_k,
            use_reranker=not args.no_rerank,
            offline=kb_settings.offline,
        )
        # Default: the retrieval pipeline alone, so the retrieval stack stays inspectable with no
        # LLM, no API key and no sub-agent in the way. `--agent` runs the real tool — the
        # advanced RAG agent — which is what the SEC-bench agent actually calls.
        if args.agent:
            print(search.run(query=args.query)["results"])
            return 0

        try:
            filters = json.loads(args.filters) if args.filters else None
        except json.JSONDecodeError as exc:
            print(f"--filters is not valid JSON: {exc}", file=sys.stderr)
            return 2
        try:
            documents = search.retrieve(args.query, filters=filters, top_k=args.top_k)
        except KbStoreError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(render.render_listing(documents, query=args.query))
        return 0

    raise SystemExit(f"Unknown kb action '{args.action}'.")


def _kb_seed_snippet(kb_settings) -> str:
    """The tool's init_parameters as they must appear in the seed YAML."""
    import yaml

    from security_agent.knowledge_base import pins

    return yaml.safe_dump(
        {
            "qdrant_url": kb_settings.qdrant_url,
            "qdrant_index": kb_settings.qdrant_index,
            "embedding_model": kb_settings.embedding_model,
            "embedding_revision": pins.MODEL_PINS.get(kb_settings.embedding_model),
            "query_prefix": kb_settings.query_prefix,
            "sparse_model": kb_settings.sparse_model,
            "reranker_model": kb_settings.reranker_model,
            "reranker_revision": pins.MODEL_PINS.get(kb_settings.reranker_model),
            "device": kb_settings.device,
        },
        sort_keys=False,
    ).rstrip()


def _kb_check_models(kb_settings, meta: dict) -> int:
    """Warm both retrieval models so an air-gapped deployment fails here, not mid-task.

    `warm_up_retrieval`, not `warm_up`: this check is about the HuggingFace models being present
    on disk, and building the knowledge-base agent on top would additionally demand
    OPENAI_API_KEY — turning an offline pre-flight check into a networked one.
    """
    from security_agent.knowledge_base.search import SecurityKbSearch

    search = SecurityKbSearch(
        qdrant_url=kb_settings.qdrant_url,
        qdrant_index=kb_settings.qdrant_index,
        embedding_model=meta.get("embedding_model", kb_settings.embedding_model),
        query_prefix=meta.get("query_prefix", kb_settings.query_prefix),
        sparse_model=meta.get("sparse_model", kb_settings.sparse_model),
        reranker_model=meta.get("reranker_model", kb_settings.reranker_model),
        device=kb_settings.device,
        offline=kb_settings.offline,
    )
    search.warm_up_retrieval()
    if search.error:
        print(f"\nmodels: FAILED\n{search.error}", file=sys.stderr)
        return 1
    print("\nmodels: ok (embedder and reranker loaded)")
    return 0


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="secagent", description="Self-improving security agent.")
    parser.add_argument(
        "--project", type=Path, default=paths.PROJECT_ROOT, help="Project root (default: the package's)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Every command shares one benchmark selector; it resolves the whole config block plus all
    # derived per-benchmark paths (seed, runs store, pipeline, program). Default: config.yaml.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--benchmark", default=None, help="Benchmark name (default: config.yaml default_benchmark)."
    )

    p = sub.add_parser("build-benchmark", parents=[common], help="Pre-sample a benchmark's tasks to benchmark_data/.")
    p.add_argument("--train-pairs", type=int, default=15)
    p.add_argument("--holdout-pairs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_build_benchmark)

    p = sub.add_parser("eval", parents=[common], help="Evaluate the agent on a benchmark split.")
    p.add_argument("--split", default="all", choices=("train", "holdout", "all"))
    p.add_argument("--limit", type=int, default=None, help="Truncate to the first N tasks.")
    p.add_argument("--pipeline", default=None, help="Pipeline YAML to evaluate (default: pipelines/<benchmark>.yaml).")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("optimize", help="The self-improvement loop.")
    osub = p.add_subparsers(dest="action", required=True)
    osub.add_parser("init", parents=[common], help="Score the seed and install it as champion.")
    q = osub.add_parser("run", parents=[common], help="Run optimization iterations.")
    q.add_argument("--iterations", type=int, default=None)
    q.add_argument("--no-meta", action="store_true", help="Just score the current champion once.")
    q = osub.add_parser("status", parents=[common], help="Show champion and recent journal entries.")
    q.add_argument("--last", type=int, default=10)
    osub.add_parser("report", parents=[common], help="Holdout report for seed vs champion.")
    q = osub.add_parser("rollback", parents=[common], help="Point the champion at a previously scored config.")
    q.add_argument("hash")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("promote", parents=[common], help="Write the champion into pipelines/<benchmark>.yaml.")
    p.add_argument("--champion", default=None, help="Champion YAML (default: runs/<benchmark>/champion.yaml).")
    p.add_argument("--out", default=None, help="Output pipeline YAML (default: pipelines/<benchmark>.yaml).")
    p.set_defaults(func=cmd_promote)

    # kb: the security knowledge base behind the search_security_kb tool. No `--benchmark` —
    # one knowledge base serves every benchmark.
    p = sub.add_parser("kb", help="Build, inspect and query the security knowledge base.")
    ksub = p.add_subparsers(dest="action", required=True)

    # Where the corpus lives. Every kb action needs it, so it is its own parent parser.
    kb_store_args = argparse.ArgumentParser(add_help=False)
    kb_store_args.add_argument(
        "--qdrant-url", default=None,
        help="Qdrant endpoint (default: config.yaml knowledge_base.qdrant_url, or KB_QDRANT_URL). "
             "`docker compose up -d qdrant` serves the default.",
    )
    kb_store_args.add_argument(
        "--qdrant-index", default=None, help="Collection name (default: security_kb)."
    )

    kb_build_args = argparse.ArgumentParser(add_help=False, parents=[kb_store_args])
    kb_build_args.add_argument(
        "--profile", default=None, choices=("dev", "standard", "full"),
        help="Which sources to ingest (default: config.yaml knowledge_base.profile).",
    )
    kb_build_args.add_argument(
        "--source", default=None, help="Ingest a single source instead of a whole profile."
    )
    kb_build_args.add_argument("--device", default=None, choices=("cpu", "mps", "cuda"))
    kb_build_args.add_argument("--batch-size", type=int, default=None, help="Documents per pipeline run.")
    kb_build_args.add_argument("--st-batch-size", type=int, default=None, help="sentence-transformers encode batch.")
    kb_build_args.add_argument("--nvd-days", type=int, default=None, help="NVD lookback window in days.")
    kb_build_args.add_argument("--nvd-min-cvss", type=float, default=None, help="NVD CVSS floor.")
    kb_build_args.add_argument("--nvd-key", default=None, help="NVD API key (default: NVD_API_KEY).")
    kb_build_args.add_argument("--offline", action="store_true", help="Never download models (local HF cache only).")
    kb_build_args.add_argument("-v", "--verbose", action="store_true")

    ksub.add_parser("build", parents=[kb_build_args], help="Fetch, chunk, embed and merge into the store.")
    ksub.add_parser(
        "rebuild", parents=[kb_build_args],
        help="Like build, but clear the caches and replace the store instead of merging.",
    )
    q = ksub.add_parser("stats", parents=[kb_store_args], help="Show what is in the built store.")
    q.add_argument(
        "--check-models", action="store_true",
        help="Also load the embedder and reranker, so an air-gapped setup fails here and not mid-task.",
    )
    q = ksub.add_parser(
        "query", parents=[kb_store_args],
        help="Run the retrieval pipeline from the shell (no LLM involved).",
    )
    q.add_argument("query")
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument(
        "--filters", default=None,
        help='A Haystack filter as JSON, e.g. \'{"field": "meta.source", "operator": "==", '
             '"value": "cwe"}\'. The same grammar the knowledge-base agent writes, so this '
             "reproduces what a filtered lookup actually retrieves.",
    )
    q.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder (latency comparison).")
    q.add_argument(
        "--agent", action="store_true",
        help="Ask the advanced RAG agent instead of printing raw retrieval results — the real "
             "search_security_kb tool. Needs OPENAI_API_KEY; --filters does not apply "
             "(the agent writes its own).",
    )
    p.set_defaults(func=cmd_kb)

    args = parser.parse_args(argv)
    project_root = args.project.resolve()

    # Constructing the target/meta chat generators needs their API key present even for
    # build/promote (the value is never serialized). Load the project .env; existing env wins.
    try:
        from dotenv import load_dotenv

        load_dotenv(project_root / ".env")
    except ImportError:
        pass

    settings = load_settings(project_root, benchmark=getattr(args, "benchmark", None))
    return args.func(settings, args)


if __name__ == "__main__":
    sys.exit(main())
