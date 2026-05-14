from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RAGSchema:
    generative_llm_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    generative_llm_params: int = 7_000_000_000

    encoder_id: str = "sentence-transformers/all-mpnet-base-v2"
    encoder_params: int = 120_000_000
    encoder_dim: int = 768

    hot_tier_max_vectors: int = 100_000
    cold_tier_max_vectors: int = 1_000_000_000

    retrieval_freq_mean: float = 3.0
    queries_per_retrieval: int = 1
    passages_per_retrieval: int = 5
    tokens_per_passage: int = 20
    tokens_per_injection: int = 100

    rewriter_collocated: bool = False
    rewriter_id: Optional[str] = None

    prompt_length: int = 512
    decode_length: int = 256
    chunk_size: int = 512
    chunk_overlap: int = 50

    intrygue_entropy_threshold: float = 2.5
    intrygue_copy_threshold: float = 0.3
    intrygue_window: int = 8
    intrygue_ngram: int = 3
    intrygue_warmup_tokens: int = 4
    intrygue_suppression_window: int = 32

    stop_rag_max_retrievals: int = 8
    stop_rag_lambda: float = 0.1
    stop_rag_q_table_path: Optional[str] = None

    spec_decoding_suppression_window: int = 32
    spec_decoding_enabled: bool = False
    spec_decoding_num_tokens: int = 5
    spec_decoding_ngram_max: int = 4
    spec_decoding_ngram_min: int = 2

    logprobs_k: int = 32

    apc_enabled: bool = True
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    max_context_tokens: int = 8192

    nats_subscriber_concurrency: int = 4
    retrieval_inflight_timeout_s: float = 30.0
    cold_search_timeout_ms: int = 100
    cold_search_soft_timeout_ms: int = 50
    cold_search_hnsw_ef: int = 64
    cold_search_hnsw_ef_soft: int = 16
