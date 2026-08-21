"""`search_security_kb`: an advanced RAG agent over the security knowledge base.

`SecurityKbSearch` is a plain `@component` that owns an `Agent` built by
`haystack_integrations.agent_pack.advanced_rag.create_advanced_rag_agent` — the Advanced RAG
Agent from the Haystack agent pack (https://docs.haystack.deepset.ai/docs/advanced-rag-agent).
The pack supplies the loop and five tools; this module supplies the corpus, the retriever, and
the untrusted-content boundary:

    create_advanced_rag_agent(document_store=<Qdrant collection>, retriever=<pipeline below>)
      ├── list_metadata_fields / get_metadata_field_values / get_metadata_field_range  (pack)
      ├── fetch_documents_by_filter                                                    (pack)
      └── search_documents = PipelineTool over

              dense_embedder  ─┐
                               ├─> QdrantHybridRetriever ─┬─> shaper
              sparse_embedder ─┘                          │
                               └─> ranker (cross-encoder) ┘

Why the pack rather than the hand-rolled `query -> hybrid retrieval -> rendered chunks` tool this
started as: the corpus is heterogeneous and heavily metadata-bearing (three datasets, and
`cve_id` / `cwe_id` / `cvss_score` / `severity` / `published_date` / `platform` on top), which is
the shape the Advanced RAG Agent's docs call out as its case. The old tool exposed `sources` and
`min_cvss` as hand-written enum parameters — a fixed slice of the filter space kept in sync with
the corpus by hand. The pack's metadata tools let the sub-agent discover the fields and their
actual values at query time and write a real Haystack filter.

It also changes what the outer agent gets back: a written, cited briefing rather than five
rendered chunks it has to read and reconcile itself, produced by a loop that can search again
when the first attempt misses.

Retrieval is `QdrantHybridRetriever`: one call, both vectors, Qdrant fusing them server-side.
The lexical leg is a sparse BM25 vector (see `knowledge_base.store`) rather than an in-process
BM25 index, which is what replaced ~800 lines of hand-rolled artifact I/O, a numpy matmul
retriever and a row-index — Qdrant does persistence, filtering and ANN itself.

Two things are deliberately kept from the previous, measurement-driven retrieval stage; both sit
below the pack's tool boundary (see `KbResultShaper`):

* The cross-encoder's order is **fused with** the retrieval order (weighted RRF, the reranker at
  `reranker_weight=0.5`) rather than replacing it. Measured on `"heap buffer overflow in libpng
  row decoding"`, the lexical leg's top hits were exactly right (`EDB-389: LibPNG Graphics
  Library - Remote Buffer Overflow`) and the cross-encoder pushed them out of the top 8 entirely:
  its raw logits across the whole candidate pool were 0.96-0.999, i.e. saturated, and it
  systematically prefers NVD's prose descriptions over the terse one-line titles ExploitDB rows
  are made of.
* A per-source cap on the returned set. The corpus is volume-skewed (46k ExploitDB rows, ~12k
  NVD, ~1k CWE), so without it a technique query returns five near-identical CVE records in one
  shot. The sub-agent *can* recover by filtering on `meta.source` and searching again, but the
  cap makes the first result set diverse and saves the round trip.

Why `SecurityKbSearch` stays a plain component that builds all of this in `warm_up()`, rather
than the seed carrying the pack's `Agent` directly:

* Everything the two transformer models and the Qdrant connection cost must stay out of
  `from_dict()` — i.e. out of `secagent promote`, `optimize`'s config validation, and
  `tests/test_seeds.py`.
* The optimizer patches flat `init_parameters`. A dozen scalars it can tune (`top_k`,
  `reranker_weight`, `max_per_source`, `llm_model`, `max_agent_steps`, ...) is a better knob
  surface than a nested Agent-inside-a-tool blob.

Dropped from redamon's original retrieval pipeline, for the record: Neo4j fulltext (the sparse
leg is the keyword leg), the per-source score boosts (a cross-encoder judges query/document
relevance directly, and static boosts would fight the optimizer's tuning), and the 0.35
sufficiency threshold (it gated a Tavily web-search fallback that does not exist here).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, Document, SparseEmbedding

from security_agent.knowledge_base import prompts, render
from security_agent.knowledge_base.store import KbStoreError, check_model, make_store, read_meta

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_INDEX = "security_kb"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# The sub-agent's LLM. Deliberately the same tier as the SEC-bench target agent: this loop runs
# inside one of that agent's tool calls, so its cost is multiplied by every lookup.
DEFAULT_LLM_MODEL = "gpt-5.4-mini"

# sentence-transformers `encode` on a shared model is not documented thread-safe, and both the
# outer agent and the sub-agent run tool calls concurrently. The lock covers only the two model
# calls (see `SerializedTextEmbedder` / `SerializedRanker`), so two concurrent knowledge-base
# lookups still overlap on everything else — the LLM calls, BM25, and the matmul.
_MODEL_LOCK = threading.Lock()

# Built retrieval pipelines and agents, keyed by the configuration each depends on. Rebuilding
# one per task would re-read the artifact and re-load both transformer models:
# benchmarks/secbench.py calls load_agent() + agent.warm_up() inside its per-task loop,
# in-process. Two caches rather than one because the retrieval half is usable — and warmed —
# without the LLM half; see `warm_up_retrieval`.
_PIPELINES: dict[tuple, Any] = {}
_AGENTS: dict[tuple, Any] = {}
_CACHE_LOCK = threading.Lock()


def clear_agent_cache() -> None:
    """Drop cached pipelines and agents (tests, and after a rebuild)."""
    with _CACHE_LOCK:
        _PIPELINES.clear()
        _AGENTS.clear()


# Reciprocal-rank-fusion constant, the standard value from Cormack et al. 2009 (and what
# DocumentJoiner uses internally for its own RRF mode).
RRF_K = 60


def fuse_by_rank(
    orders: list[list[Document]],
    k: int = RRF_K,
    weights: Optional[list[float]] = None,
) -> list[Document]:
    """Weighted reciprocal rank fusion over several orderings of the same documents.

    Used to combine the hybrid-retrieval order with the cross-encoder's order, so neither can
    single-handedly bury the other's best hit. Returns new Documents whose `score` is the fused
    score; ties break on id so the ordering is deterministic across runs.
    """
    weights = weights or [1.0] * len(orders)
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for order, weight in zip(orders, weights):
        for rank, doc in enumerate(order, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + weight / (k + rank)
            documents.setdefault(doc.id, doc)
    ranked = sorted(documents.values(), key=lambda d: (-scores[d.id], d.id))
    return [
        Document(id=d.id, content=d.content, meta=dict(d.meta), score=scores[d.id])
        for d in ranked
    ]


def cap_per_source(documents: list[Document], top_k: int, max_per_source: int) -> list[Document]:
    """Take the best `top_k`, allowing at most `max_per_source` from any one dataset.

    The corpus is volume-skewed, so without this a technique query comes back as five
    near-identical CVE records and the sub-agent has to spend another turn filtering on
    `meta.source` to see the CWE class that actually answers it. Documents
    deferred by the cap are used to backfill if the capped set cannot reach `top_k`, so the cap
    never costs results.
    """
    if max_per_source <= 0:
        return documents[:top_k]
    kept: list[Document] = []
    deferred: list[Document] = []
    seen: dict[str, int] = {}
    for doc in documents:
        source = (doc.meta or {}).get("source", "unknown")
        if seen.get(source, 0) < max_per_source:
            seen[source] = seen.get(source, 0) + 1
            kept.append(doc)
        else:
            deferred.append(doc)
        if len(kept) == top_k:
            return kept
    return (kept + deferred)[:top_k]


def normalize_scores(documents: list[Document]) -> list[Document]:
    """Rescale scores so the best result is 1.0.

    Raw fused scores are reciprocal-rank sums (~0.03 for everything), which renders as an
    uninformative `score=0.03` on every line of a tool result. Normalizing makes the column mean
    something readable — relative standing within this result set. It is deliberately NOT
    presented as a calibrated relevance probability: the cross-encoder's own sigmoid saturates
    near 1.0 across the whole candidate pool on this corpus, which is why it is not used directly.
    """
    if not documents:
        return documents
    best = max((d.score or 0.0) for d in documents) or 1.0
    return [
        Document(id=d.id, content=d.content, meta=d.meta, score=(d.score or 0.0) / best)
        for d in documents
    ]


# ---------------------------------------------------------------------------
# Model factories — the seam the offline tests replace (see knowledge_base.testing), so no test
# downloads a transformer or reaches an LLM. Same idea as
# `build.build_indexing_pipeline(embedder=...)`.
# ---------------------------------------------------------------------------

def make_document_store(config: "SecurityKbSearch"):
    """Connect to the collection holding the corpus.

    A factory like the model ones, because an in-memory Qdrant is per-client: a test has to hand
    the component the same store instance the corpus was built into, or it would open a second,
    empty database.
    """
    from security_agent.knowledge_base.settings import load_kb_settings

    return make_store(
        load_kb_settings(
            qdrant_url=config.qdrant_url,
            qdrant_index=config.qdrant_index,
            embedding_model=config.embedding_model,
        )
    )


def make_text_embedder(config: "SecurityKbSearch"):
    from haystack.utils import ComponentDevice
    from haystack_integrations.components.embedders.sentence_transformers import (
        SentenceTransformersTextEmbedder,
    )

    return SentenceTransformersTextEmbedder(
        model=config.embedding_model,
        revision=config.embedding_revision,
        # bge is asymmetric: the instruction belongs on the query side only, and must match what
        # the store was built with.
        prefix=config.query_prefix,
        normalize_embeddings=True,
        device=ComponentDevice.from_str(config.device),
        local_files_only=config.offline,
        progress_bar=False,
    )


def make_sparse_text_embedder(config: "SecurityKbSearch"):
    """The lexical leg's query encoder — BM25 as a sparse vector.

    No `revision` to pin: fastembed resolves `Qdrant/bm25` to a tokenizer and an IDF table it
    versions itself, so the pinned `fastembed-haystack` dependency is what fixes it. See
    `pins.MODEL_PINS`.
    """
    from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

    return FastembedSparseTextEmbedder(model=config.sparse_model, progress_bar=False)


def make_ranker(config: "SecurityKbSearch"):
    from haystack.utils import ComponentDevice
    from haystack_integrations.components.rankers.sentence_transformers import (
        SentenceTransformersSimilarityRanker,
    )

    # The ranker has no `revision` parameter (unlike the embedder), so the pin goes through to
    # the HuggingFace loaders via the *_kwargs passthroughs — which works on every release of
    # the integration.
    revision_kwargs: dict[str, Any] = {}
    if config.reranker_revision:
        pin = {"revision": config.reranker_revision}
        revision_kwargs = {"model_kwargs": pin, "tokenizer_kwargs": pin, "config_kwargs": pin}

    return SentenceTransformersSimilarityRanker(
        model=config.reranker_model,
        top_k=config.candidate_pool,
        # Sigmoid over the raw logits, so scores are comparable across queries when logging.
        scale_score=True,
        device=ComponentDevice.from_str(config.device),
        batch_size=16,
        **revision_kwargs,
    )


def make_llm(config: "SecurityKbSearch"):
    """The chat generator driving the sub-agent's loop.

    Low reasoning effort, matching the pack's own default: this loop picks tools and writes
    filters, it does not need to reason its way to the answer, and it runs inside another
    agent's tool call where latency compounds.
    """
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator

    return OpenAIResponsesChatGenerator(
        model=config.llm_model,
        timeout=config.llm_timeout,
        max_retries=3,
        generation_kwargs={
            "reasoning": {"effort": "low"},
            "max_output_tokens": config.max_output_tokens,
        },
    )


# ---------------------------------------------------------------------------
# Retrieval-pipeline components
# ---------------------------------------------------------------------------

@component
class SerializedTextEmbedder:
    """Holds `_MODEL_LOCK` across the wrapped embedder's `run`.

    A thin wrapper rather than a lock inside `SecurityKbSearch.run`: the embedder is reached from
    inside the pack's `PipelineTool`, several agent steps below anything this module calls, and
    locking the whole lookup would serialize two concurrent knowledge-base tool calls across
    their LLM turns as well.
    """

    def __init__(self, embedder: Any):
        self.embedder = embedder

    def warm_up(self) -> None:
        if hasattr(self.embedder, "warm_up"):
            self.embedder.warm_up()

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict:
        with _MODEL_LOCK:
            return {"embedding": self.embedder.run(text=text)["embedding"]}


@component
class SerializedSparseTextEmbedder:
    """`SerializedTextEmbedder`, for the BM25 sparse encoder."""

    def __init__(self, embedder: Any):
        self.embedder = embedder

    def warm_up(self) -> None:
        if hasattr(self.embedder, "warm_up"):
            self.embedder.warm_up()

    @component.output_types(sparse_embedding=SparseEmbedding)
    def run(self, text: str) -> dict:
        with _MODEL_LOCK:
            return {"sparse_embedding": self.embedder.run(text=text)["sparse_embedding"]}


@component
class SerializedRanker:
    """`SerializedTextEmbedder`, for the cross-encoder."""

    def __init__(self, ranker: Any):
        self.ranker = ranker

    def warm_up(self) -> None:
        if hasattr(self.ranker, "warm_up"):
            self.ranker.warm_up()

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: Optional[int] = None) -> dict:
        with _MODEL_LOCK:
            return {"documents": self.ranker.run(query=query, documents=documents, top_k=top_k)["documents"]}


@component
class KbResultShaper:
    """Final stage of the retrieval pipeline: fuse the two orderings, then diversify.

    Sits below the pack's tool boundary so `search_documents` returns a shaped result set without
    the sub-agent having to know any of this exists. See the module docstring for the measurements
    behind both steps.
    """

    def __init__(self, top_k: int = 5, max_per_source: int = 2, reranker_weight: float = 0.5):
        self.top_k = top_k
        self.max_per_source = max_per_source
        self.reranker_weight = reranker_weight

    @component.output_types(documents=list[Document])
    def run(
        self,
        retrieved: list[Document],
        reranked: Optional[list[Document]] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        top_k = top_k or self.top_k
        if reranked is None:
            ordered = retrieved
        else:
            ordered = fuse_by_rank([retrieved, reranked], weights=[1.0, self.reranker_weight])
        return {"documents": normalize_scores(cap_per_source(ordered, top_k, self.max_per_source))}


# ---------------------------------------------------------------------------
# The tool component
# ---------------------------------------------------------------------------

@component
class SecurityKbSearch:
    """Ask the curated security knowledge base a question; returns one rendered string.

    Never raises from `run()` or `warm_up()` — a missing store, an unavailable model or a failed
    sub-agent run becomes an actionable message in the tool result, so one misconfiguration
    cannot fail a whole benchmark task.
    """

    def __init__(
        self,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        qdrant_index: str = DEFAULT_QDRANT_INDEX,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_revision: Optional[str] = None,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        reranker_revision: Optional[str] = None,
        device: str = "cpu",
        top_k: int = 5,
        candidate_pool: int = 30,
        # At most this many results from any one dataset, so a volume-heavy source cannot fill
        # a whole result set. 0 disables the cap.
        max_per_source: int = 2,
        # Weight of the cross-encoder's ordering relative to hybrid retrieval's (1.0) when the
        # two are fused. Below 1.0 the reranker refines rather than dominates — see the module
        # docstring for why that matters on this corpus. 0 disables its influence entirely.
        reranker_weight: float = 0.5,
        max_output_chars: int = 12000,
        use_reranker: bool = True,
        offline: bool = False,
        # -- the sub-agent ---------------------------------------------------
        llm_model: str = DEFAULT_LLM_MODEL,
        llm_timeout: float = 120.0,
        max_output_tokens: int = 4000,
        # Small on purpose: this loop runs inside one of the outer agent's 40 steps. Six steps is
        # enough for inspect -> search -> narrow -> answer; the pack's BackupAnswerHook writes a
        # best-effort answer from what was gathered if the budget runs out, so a cut-off lookup
        # still returns something usable.
        max_agent_steps: int = 6,
        max_fetched_docs: int = 5,
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_index = qdrant_index
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.query_prefix = query_prefix
        self.sparse_model = sparse_model
        self.reranker_model = reranker_model
        self.reranker_revision = reranker_revision
        self.device = device
        self.top_k = top_k
        self.candidate_pool = candidate_pool
        self.max_per_source = max_per_source
        self.reranker_weight = reranker_weight
        self.max_output_chars = max_output_chars
        self.use_reranker = use_reranker
        self.offline = offline
        self.llm_model = llm_model
        self.llm_timeout = llm_timeout
        self.max_output_tokens = max_output_tokens
        self.max_agent_steps = max_agent_steps
        self.max_fetched_docs = max_fetched_docs

        self._pipeline: Optional[Any] = None
        self._store: Optional[Any] = None
        self._agent: Optional[Any] = None
        self._error: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def error(self) -> Optional[str]:
        """Why the tool is unavailable, or None. Set by `warm_up()`, which never raises."""
        return self._error

    @property
    def _retrieval_key(self) -> tuple:
        return (
            self.qdrant_url,
            self.qdrant_index,
            self.embedding_model,
            self.embedding_revision,
            self.query_prefix,
            self.sparse_model,
            self.reranker_model,
            self.reranker_revision,
            self.device,
            self.top_k,
            self.candidate_pool,
            self.max_per_source,
            self.reranker_weight,
            self.use_reranker,
            self.offline,
        )

    @property
    def _agent_key(self) -> tuple:
        return (
            *self._retrieval_key,
            self.llm_model,
            self.llm_timeout,
            self.max_output_tokens,
            self.max_agent_steps,
            self.max_fetched_docs,
        )

    def _fail(self, error: str) -> None:
        self._error = error
        logger.warning("search_security_kb unavailable: %s", error)

    def warm_up_retrieval(self) -> None:
        """Load the store and the two transformer models — everything except the sub-agent.

        Kept separate from `warm_up()` so the retrieval stack can be exercised and diagnosed
        without an LLM: `secagent kb query` and `secagent kb stats --check-models` are the
        pre-flight checks for an air-gapped deployment, and neither should need OPENAI_API_KEY.
        Records failures instead of raising, like `warm_up()`.
        """
        if self._pipeline is not None or self._error is not None:
            return
        key = self._retrieval_key
        with _CACHE_LOCK:
            cached = _PIPELINES.get(key)
        if cached is not None:
            self._pipeline, self._store = cached
            return
        try:
            store = self._open_store()
            pipeline = self._build_retrieval_pipeline(store)
        except KbStoreError as exc:
            self._fail(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._fail(
                f"Security knowledge-base search is unavailable: could not load the retrieval "
                f"models ({type(exc).__name__}: {exc}). The embedding model "
                f"{self.embedding_model!r} and reranker {self.reranker_model!r} are downloaded "
                f"from HuggingFace on first use; in an air-gapped environment pre-seed HF_HOME "
                f"and set KB_OFFLINE=1."
            )
            return
        with _CACHE_LOCK:
            _PIPELINES.setdefault(key, (pipeline, store))
            self._pipeline, self._store = _PIPELINES[key]

    def warm_up(self) -> None:
        """Load the store, the models and the knowledge-base agent. Never raises.

        Raising here would fail `Agent.warm_up()` (via `warm_up_tools`) and take down every
        other tool with it.
        """
        if self._agent is not None or self._error is not None:
            return
        self.warm_up_retrieval()
        if self._error is not None:
            return
        key = self._agent_key
        with _CACHE_LOCK:
            cached = _AGENTS.get(key)
        if cached is not None:
            self._agent = cached
            return
        try:
            agent = self._build_agent()
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._fail(
                f"Security knowledge-base search is unavailable: could not build the retrieval "
                f"agent ({type(exc).__name__}: {exc}). Its LLM ({self.llm_model}) needs "
                f"OPENAI_API_KEY."
            )
            return
        with _CACHE_LOCK:
            _AGENTS.setdefault(key, agent)
            self._agent = _AGENTS[key]

    def _open_store(self) -> Any:
        """Connect to the collection and refuse one built with a different embedding model."""
        from security_agent.knowledge_base.settings import load_kb_settings

        store = make_document_store(self)
        settings = load_kb_settings(
            qdrant_url=self.qdrant_url, qdrant_index=self.qdrant_index,
        )
        check_model(read_meta(store, settings), self.embedding_model)
        return store

    def _input_mapping(self) -> dict[str, list[str]]:
        """How the pack's `search_documents` tool inputs reach the retrieval pipeline.

        The keys are fixed by the pack: exactly `query` and `filters`. Both embedders take the
        query text; `filters` goes to the single retriever, which applies it to both vectors —
        one of the things that got simpler when Qdrant took over the lexical leg, since there is
        no longer a second retriever to keep in sync.
        """
        query_sockets = ["dense_embedder.text", "sparse_embedder.text"]
        if self.use_reranker:
            query_sockets.append("ranker.query")
        return {"query": query_sockets, "filters": ["retriever.filters"]}

    def _build_retrieval_pipeline(self, store: Any):
        """The hybrid retriever the pack turns into its `search_documents` tool.

        One retriever, not two legs plus a joiner: `QdrantHybridRetriever` sends both vectors in
        one request and Qdrant fuses them with RRF server-side.
        """
        from haystack import Pipeline
        from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

        pipeline = Pipeline()
        pipeline.add_component("dense_embedder", SerializedTextEmbedder(make_text_embedder(self)))
        pipeline.add_component(
            "sparse_embedder", SerializedSparseTextEmbedder(make_sparse_text_embedder(self))
        )
        pipeline.add_component(
            "retriever", QdrantHybridRetriever(document_store=store, top_k=self.candidate_pool)
        )
        pipeline.add_component(
            "shaper",
            KbResultShaper(
                top_k=self.top_k,
                max_per_source=self.max_per_source,
                reranker_weight=self.reranker_weight,
            ),
        )
        pipeline.connect("dense_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
        pipeline.connect("retriever.documents", "shaper.retrieved")

        if self.use_reranker:
            # Rank the whole candidate pool, not just top_k: the shaper fuses this ordering with
            # the retrieval ordering, so it needs both in full.
            pipeline.add_component("ranker", SerializedRanker(make_ranker(self)))
            pipeline.connect("retriever.documents", "ranker.documents")
            pipeline.connect("ranker.documents", "shaper.reranked")

        pipeline.warm_up()
        return pipeline

    def _build_agent(self) -> Any:
        """Hand the warmed store and retrieval pipeline to the agent pack."""
        from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent

        assert self._store is not None and self._pipeline is not None
        input_mapping = self._input_mapping()
        agent = create_advanced_rag_agent(
            document_store=self._store,
            retriever=self._pipeline,
            retrieval_pipeline_input_mapping=input_mapping,
            retrieval_pipeline_output_mapping={"shaper.documents": "documents"},
            llm=make_llm(self),
            backup_answer_llm=make_llm(self),
            system_prompt=prompts.KB_AGENT_SYSTEM_PROMPT,
            max_agent_steps=self.max_agent_steps,
            max_fetched_docs=self.max_fetched_docs,
            # A tool error the sub-agent can read and retry (e.g. a filter naming a field that
            # does not exist) beats an exception that costs the outer agent the whole lookup.
            raise_on_tool_invocation_failure=False,
        )
        # The retrieval pipeline is already warm; this warms the pack's own tools and the LLM.
        agent.warm_up()
        return agent

    # -- query -------------------------------------------------------------

    @component.output_types(results=str)
    def run(self, query: str) -> dict:
        """Ask the knowledge base a question.

        :param query: What to look up — a natural-language question about a weakness class, a
            technique or a CVE, or an exact identifier such as `CVE-2024-21626` or `CWE-787`.
        """
        self.warm_up()
        if self._error is not None:
            return {"results": self._error}

        query = (query or "").strip()
        if not query:
            return {"results": "No question given. Ask about a weakness, a technique or a CVE."}

        assert self._agent is not None
        try:
            result = self._agent.run(messages=[ChatMessage.from_user(query)])
        except Exception as exc:  # noqa: BLE001 - tool results must stay strings
            logger.exception("search_security_kb failed for query %r", query)
            return {
                "results": (
                    f"Security knowledge-base lookup failed ({type(exc).__name__}: {exc}). "
                    f"The knowledge base may need rebuilding: `secagent kb stats`."
                )
            }

        answer = (result["last_message"].text or "").strip()
        documents = result.get("documents") or []
        if not answer:
            return {
                "results": (
                    f"The knowledge-base lookup for {query!r} produced no answer. Try a broader "
                    f"phrasing (the weakness class rather than an exact function name)."
                )
            }
        return {
            "results": render.render_answer(
                answer,
                documents,
                query=query,
                max_output_chars=self.max_output_chars,
                steps=result.get("step_count"),
            )
        }

    def retrieve(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """Run the retrieval pipeline alone, with no LLM in the loop.

        This is what `secagent kb query` uses: it makes the retrieval stack testable and
        tunable — and diagnosable in an air-gapped or key-less environment — separately from the
        sub-agent that consumes it.
        """
        self.warm_up_retrieval()
        if self._error is not None:
            raise KbStoreError(self._error)
        assert self._pipeline is not None

        top_k = top_k or self.top_k
        pool = max(top_k * 6, self.candidate_pool)
        inputs: dict[str, dict[str, Any]] = {
            "dense_embedder": {"text": query},
            "sparse_embedder": {"text": query},
            "retriever": {"filters": filters, "top_k": pool},
            "shaper": {"top_k": top_k},
        }
        if self.use_reranker:
            inputs["ranker"] = {"query": query, "top_k": pool}
        return self._pipeline.run(inputs)["shaper"]["documents"]

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return default_to_dict(
            self,
            qdrant_url=self.qdrant_url,
            qdrant_index=self.qdrant_index,
            embedding_model=self.embedding_model,
            embedding_revision=self.embedding_revision,
            query_prefix=self.query_prefix,
            sparse_model=self.sparse_model,
            reranker_model=self.reranker_model,
            reranker_revision=self.reranker_revision,
            device=self.device,
            top_k=self.top_k,
            candidate_pool=self.candidate_pool,
            max_per_source=self.max_per_source,
            reranker_weight=self.reranker_weight,
            max_output_chars=self.max_output_chars,
            use_reranker=self.use_reranker,
            offline=self.offline,
            llm_model=self.llm_model,
            llm_timeout=self.llm_timeout,
            max_output_tokens=self.max_output_tokens,
            max_agent_steps=self.max_agent_steps,
            max_fetched_docs=self.max_fetched_docs,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SecurityKbSearch":
        return default_from_dict(cls, data)
