from __future__ import annotations
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config.schema import RAGSchema
from config.settings import SystemConfig
from pipeline.iterative_rag import IterativeRAGPipeline, IterativeRAGRequest


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192)
    session_id: Optional[str] = Field(default=None)


class QueryResponse(BaseModel):
    request_id: str
    session_id: str
    generated_text: str
    retrieval_count: int
    total_latency_ms: float
    retrieval_latencies_ms: list[float]
    trace_id: str


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    sources: list[str]
    session_id: Optional[str] = Field(default=None)

    def model_post_init(self, __context) -> None:
        if len(self.texts) != len(self.sources):
            raise ValueError("texts and sources must have equal length")


class IngestResponse(BaseModel):
    indexed_chunks: int
    status: str


_pipeline: Optional[IterativeRAGPipeline] = None


def _build_config() -> tuple[RAGSchema, SystemConfig]:
    profile = os.getenv("DEPLOY_PROFILE", "rtx4090_local")
    schema = RAGSchema()

    if profile == "rtx4090_local":
        cfg = SystemConfig.for_rtx4090_local()
    elif profile == "a6000_local":
        cfg = SystemConfig.for_a6000_local()
    elif profile == "a6000":
        host = os.environ["REMOTE_HOST"]
        cfg = SystemConfig.for_a6000_ssh(host)
    elif profile == "a100":
        host = os.environ["REMOTE_HOST"]
        cfg = SystemConfig.for_a100_ssh(host)
    elif profile == "a100_80g":
        host = os.environ["REMOTE_HOST"]
        cfg = SystemConfig.for_a100_80g_ssh(host)
    elif profile == "h100":
        host = os.environ["REMOTE_HOST"]
        cfg = SystemConfig.for_h100_ssh(host)
    elif profile == "io_net":
        host = os.environ["REMOTE_HOST"]
        world = int(os.getenv("WORLD_SIZE", "4"))
        cfg = SystemConfig.for_io_net_cluster(host, world)
    else:
        raise ValueError(f"Unknown DEPLOY_PROFILE: {profile}")

    return schema, cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    schema, config = _build_config()
    _pipeline = await IterativeRAGPipeline.create(schema, config)
    yield
    if _pipeline:
        await _pipeline.close()


app = FastAPI(
    title="Iterative RAG API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "pipeline_ready": _pipeline is not None}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    rag_req = IterativeRAGRequest(
        query=req.query,
        session_id=req.session_id or str(uuid.uuid4()),
    )
    resp = await _pipeline.generate(rag_req)

    return QueryResponse(
        request_id=resp.request_id,
        session_id=resp.session_id,
        generated_text=resp.generated_text,
        retrieval_count=resp.retrieval_count,
        total_latency_ms=resp.total_latency_ms,
        retrieval_latencies_ms=resp.retrieval_latencies_ms,
        trace_id=resp.trace_id,
    )


@app.post("/stream")
async def stream(req: QueryRequest) -> StreamingResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    rag_req = IterativeRAGRequest(
        query=req.query,
        session_id=req.session_id or str(uuid.uuid4()),
    )

    async def _sse() -> AsyncIterator[bytes]:
        async for ev in _pipeline.generate_stream(rag_req):
            payload = json.dumps(asdict(ev), separators=(",", ":"))
            yield f"event: {ev.kind}\ndata: {payload}\n\n".encode()

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    n = await _pipeline.ingest(
        texts=req.texts,
        sources=req.sources,
        session_id=req.session_id,
    )
    return IngestResponse(indexed_chunks=n, status="ok")


@app.delete("/session/{session_id}")
async def evict_session(session_id: str) -> dict:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    _pipeline._vector_store.evict_session(session_id)
    return {"evicted": session_id}
