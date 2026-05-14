from __future__ import annotations
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from config.schema import RAGSchema
from config.settings import SystemConfig
from controllers.intrygue import IntrygueController, IntrygueState
from controllers.stop_rag import StopRagController
from engine.scheduler import PriorityNonStallScheduler, SequencePriority
from engine.vllm_wrapper import VLLMWrapper, SequenceState
from orchestration.data_plane import ContextPayload, DataPlane
from orchestration.nats_client import NATSControlPlane, RetrievalRequest
from retrieval.cache import RedisChunkCache
from retrieval.embedder import DocumentEmbedder
from retrieval.vector_store import TieredVectorStore


@dataclass
class IterativeRAGRequest:
    query: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class IterativeRAGResponse:
    request_id: str
    session_id: str
    generated_text: str
    retrieval_count: int
    total_latency_ms: float
    retrieval_latencies_ms: list[float]
    trace_id: str


class IterativeRAGPipeline:
    def __init__(
        self,
        schema: RAGSchema,
        config: SystemConfig,
        engine: VLLMWrapper,
        embedder: DocumentEmbedder,
        vector_store: TieredVectorStore,
        intrygue: IntrygueController,
        stop_rag: StopRagController,
        control_plane: NATSControlPlane,
        data_plane: DataPlane,
    ):
        self._schema = schema
        self._config = config
        self._engine = engine
        self._embedder = embedder
        self._vector_store = vector_store
        self._intrygue = intrygue
        self._stop_rag = stop_rag
        self._control_plane = control_plane
        self._data_plane = data_plane
        self._scheduler = PriorityNonStallScheduler(
            max_concurrent=config.hardware.decode_batch_size
        )
        self._subscriber_tasks: list[asyncio.Task] = []

    async def _initial_retrieval(self, query: str, session_id: str) -> list[str]:
        embedding = await asyncio.get_running_loop().run_in_executor(
            None, self._embedder.encode_query, query
        )
        results = await self._vector_store.search(
            query_embedding=embedding,
            session_id=session_id,
            top_k=self._schema.passages_per_retrieval,
            cold_timeout_ms=self._schema.cold_search_timeout_ms,
            cold_soft_timeout_ms=self._schema.cold_search_soft_timeout_ms,
            ef_full=self._schema.cold_search_hnsw_ef,
            ef_soft=self._schema.cold_search_hnsw_ef_soft,
        )
        return [r.text for r in results if r.text]

    async def _handle_retrieval_request(self, req: RetrievalRequest) -> None:
        t0 = time.perf_counter()
        embedding = await asyncio.get_running_loop().run_in_executor(
            None, self._embedder.encode_query, req.query_text
        )
        results = await self._vector_store.search(
            query_embedding=embedding,
            session_id=req.session_id,
            top_k=self._schema.passages_per_retrieval,
            cold_timeout_ms=self._schema.cold_search_timeout_ms,
            cold_soft_timeout_ms=self._schema.cold_search_soft_timeout_ms,
            ef_full=self._schema.cold_search_hnsw_ef,
            ef_soft=self._schema.cold_search_hnsw_ef_soft,
        )
        passages = [r.text for r in results if r.text]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        await self._data_plane.publish_context(
            reply_subject=req.reply_subject or self._control_plane.response_subject,
            payload=ContextPayload(
                sequence_id=req.sequence_id,
                retrieval_round=req.retrieval_round,
                passages=passages,
                retrieval_latency_ms=latency_ms,
            ),
        )

    async def generate(self, request: IterativeRAGRequest) -> IterativeRAGResponse:
        t_start = time.perf_counter()

        initial_passages = await self._initial_retrieval(request.query, request.session_id)
        passages_per_round: list[list[str]] = [initial_passages]
        completed_chunks: list[str] = []

        retrieval_count = 0
        retrieval_latencies: list[float] = []
        entropy_history: list[float] = []
        context_tokens = sum(len(p.split()) for p in initial_passages)
        intrygue_state = IntrygueState()
        seq_id = self._engine.new_request_id()

        context_token_ids: list[int] = self._engine.encode_text(
            " ".join(initial_passages)
        )

        await self._scheduler.acquire(seq_id, SequencePriority.ACTIVE)
        try:
            while True:
                prompt_token_ids = self._engine.build_prompt_token_ids(
                    query=request.query,
                    passages_per_round=passages_per_round,
                    completed_chunks=completed_chunks,
                )

                round_req_id = f"{seq_id}:{retrieval_count}"
                seq_state = SequenceState(
                    request_id=round_req_id,
                    session_id=request.session_id,
                    trace_id=request.trace_id,
                    context_token_ids=list(context_token_ids),
                )

                retrieval_triggered = False
                accumulated_delta = ""

                async for tok in self._engine.generate_tokens(
                    prompt_token_ids, seq_state
                ):
                    accumulated_delta += tok.delta_text

                    if tok.is_final:
                        break

                    if not tok.logprobs:
                        continue

                    e = self._intrygue.compute_entropy(tok.logprobs)
                    entropy_history.append(e)
                    if len(entropy_history) > 64:
                        entropy_history = entropy_history[-64:]

                    if not tok.is_last_in_batch:
                        continue

                    triggered, intrygue_state = self._intrygue.should_retrieve(
                        logprobs=tok.logprobs,
                        generated_ids=seq_state.generated_token_ids,
                        context_ids=seq_state.context_token_ids,
                        current_token_index=seq_state.token_index,
                        state=intrygue_state,
                    )

                    if not triggered:
                        continue

                    mean_entropy = sum(entropy_history) / len(entropy_history)
                    if self._stop_rag.should_stop(
                        retrieval_count=retrieval_count,
                        mean_entropy=mean_entropy,
                        context_tokens=context_tokens,
                        max_context_tokens=self._schema.max_context_tokens,
                    ):
                        continue

                    await self._engine.abort(round_req_id)
                    retrieval_triggered = True
                    break

                completed_chunks.append(accumulated_delta)

                if not retrieval_triggered:
                    break

                retrieval_count += 1
                t_ret = time.perf_counter()

                query_for_retrieval = (
                    f"{request.query} {accumulated_delta[-300:]}"
                )
                retrieval_req = RetrievalRequest(
                    sequence_id=seq_id,
                    retrieval_round=retrieval_count,
                    query_text=query_for_retrieval,
                    session_id=request.session_id,
                    trace_id=request.trace_id,
                    reply_subject=self._control_plane.response_subject,
                )

                await self._control_plane.publish_retrieval_request(retrieval_req)

                await self._scheduler.suspend(seq_id)
                try:
                    ctx = await self._data_plane.wait_for_context(
                        sequence_id=seq_id,
                        retrieval_round=retrieval_count,
                        timeout=self._schema.retrieval_inflight_timeout_s,
                    )
                    passages = ctx.passages
                    retrieval_latencies.append((time.perf_counter() - t_ret) * 1000.0)
                except asyncio.TimeoutError:
                    passages = []
                    retrieval_latencies.append(
                        self._schema.retrieval_inflight_timeout_s * 1000.0
                    )
                finally:
                    await self._scheduler.resume(seq_id)

                passages_per_round.append(passages)
                context_tokens += sum(len(p.split()) for p in passages)
                if passages:
                    context_token_ids = context_token_ids + self._engine.encode_text(
                        " ".join(passages)
                    )
                intrygue_state = self._intrygue.on_context_injected(
                    current_token_index=seq_state.token_index,
                    state=intrygue_state,
                )
        finally:
            await self._scheduler.release(seq_id)

        generated_text = "".join(completed_chunks)

        return IterativeRAGResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            generated_text=generated_text,
            retrieval_count=retrieval_count,
            total_latency_ms=(time.perf_counter() - t_start) * 1000.0,
            retrieval_latencies_ms=retrieval_latencies,
            trace_id=request.trace_id,
        )

    async def ingest(
        self,
        texts: list[str],
        sources: list[str],
        session_id: Optional[str] = None,
    ) -> int:
        from retrieval.chunker import SemanticChunker

        chunker = SemanticChunker(
            chunk_size=self._schema.chunk_size,
            overlap=self._schema.chunk_overlap,
        )
        chunks = []
        for text, source in zip(texts, sources):
            chunks.extend(chunker.chunk_document(text, source))

        if not chunks:
            return 0

        chunk_texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        chunk_sources = [c.source for c in chunks]

        embeddings = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._embedder.encode(chunk_texts)
        )
        await self._vector_store.index_documents(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            texts=chunk_texts,
            sources=chunk_sources,
            session_id=session_id,
        )
        return len(chunks)

    async def _start_subscribers(self) -> None:
        for i in range(self._schema.nats_subscriber_concurrency):
            t = asyncio.create_task(
                self._control_plane.subscribe_retrieval_requests(
                    self._handle_retrieval_request
                ),
                name=f"nats_subscriber_{i}",
            )
            self._subscriber_tasks.append(t)

    @classmethod
    async def create(cls, schema: RAGSchema, config: SystemConfig) -> "IterativeRAGPipeline":
        cache = RedisChunkCache(config.infra)
        await cache.initialize()

        embedder = DocumentEmbedder(schema, device="cpu")

        vector_store = TieredVectorStore(schema, config.infra, cache)
        await vector_store.initialize()

        engine = VLLMWrapper(schema, config)
        await engine.initialize()

        intrygue = IntrygueController(
            entropy_threshold=schema.intrygue_entropy_threshold,
            copy_threshold=schema.intrygue_copy_threshold,
            window=schema.intrygue_window,
            ngram_size=schema.intrygue_ngram,
            suppression_window=schema.intrygue_suppression_window,
            warmup_tokens=schema.intrygue_warmup_tokens,
        )

        stop_rag = StopRagController(
            max_retrievals=schema.stop_rag_max_retrievals,
            lambda_cost=schema.stop_rag_lambda,
            q_table_path=schema.stop_rag_q_table_path,
        )

        control_plane = NATSControlPlane(config.infra)
        await control_plane.initialize()

        data_plane = DataPlane(config.infra, config.deployment, control_plane)
        await data_plane.initialize()

        pipeline = cls(
            schema=schema,
            config=config,
            engine=engine,
            embedder=embedder,
            vector_store=vector_store,
            intrygue=intrygue,
            stop_rag=stop_rag,
            control_plane=control_plane,
            data_plane=data_plane,
        )
        await pipeline._start_subscribers()
        return pipeline

    async def close(self) -> None:
        for t in self._subscriber_tasks:
            t.cancel()
        for t in self._subscriber_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._subscriber_tasks.clear()
        await self._engine.shutdown()
        await self._vector_store.close()
        await self._data_plane.close()
        await self._control_plane.close()
