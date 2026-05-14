from __future__ import annotations
import os
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.logprobs import FlatLogprobs
from vllm.outputs import CompletionOutput

from config.schema import RAGSchema
from config.settings import SystemConfig

_SYSTEM_PROMPT = (
    "You are a precise multi-hop reasoning assistant. "
    "Use only the provided evidence to answer. "
    "Think through each step carefully before writing your final answer."
)


@dataclass
class SequenceState:
    request_id: str
    session_id: str
    trace_id: str
    generated_token_ids: list[int] = field(default_factory=list)
    context_token_ids: list[int] = field(default_factory=list)
    retrieval_round: int = 0
    token_index: int = 0
    _prev_n_tokens: int = field(default=0, repr=False)
    _prev_text_len: int = field(default=0, repr=False)


@dataclass
class TokenOutput:
    token_id: int
    delta_text: str
    logprobs: dict[int, float]
    is_final: bool
    is_last_in_batch: bool
    request_id: str


def _logprobs_at(sample_logprobs, i: int) -> dict[int, float]:
    if not sample_logprobs:
        return {}
    if isinstance(sample_logprobs, FlatLogprobs):
        if i >= len(sample_logprobs.start_indices):
            return {}
        s = sample_logprobs.start_indices[i]
        e = sample_logprobs.end_indices[i]
        return {
            tid: lp
            for tid, lp in zip(
                sample_logprobs.token_ids[s:e],
                sample_logprobs.logprobs[s:e],
            )
        }
    if i >= len(sample_logprobs):
        return {}
    entry = sample_logprobs[i]
    if not entry:
        return {}
    return {t: lp.logprob for t, lp in entry.items()}


class VLLMWrapper:
    def __init__(self, schema: RAGSchema, config: SystemConfig):
        self._schema = schema
        self._config = config
        self._engine: Optional[AsyncLLMEngine] = None
        self._tokenizer = AutoTokenizer.from_pretrained(
            schema.generative_llm_id,
            trust_remote_code=True,
            use_fast=True,
        )

    def _resolve_model_path(self) -> str:
        model_id = self._schema.generative_llm_id
        model_dir = self._config.model_dir
        if model_dir:
            local_path = os.path.join(model_dir, model_id.split("/")[-1])
            if os.path.isdir(local_path):
                return local_path
        return model_id

    def _engine_args(self) -> AsyncEngineArgs:
        hw = self._config.hardware
        kwargs: dict = dict(
            model=self._resolve_model_path(),
            dtype="auto",
            gpu_memory_utilization=self._schema.gpu_memory_utilization,
            tensor_parallel_size=hw.tensor_parallel_size,
            enable_prefix_caching=self._schema.apc_enabled,
            max_model_len=self._schema.max_model_len,
            trust_remote_code=True,
            enable_log_requests=False,
            max_num_batched_tokens=self._schema.max_model_len,
            max_num_seqs=hw.decode_batch_size,
            max_logprobs=self._schema.logprobs_k,
        )
        if hw.quantization == "bitsandbytes":
            kwargs["quantization"] = "bitsandbytes"
            kwargs["load_format"] = "bitsandbytes"
        elif hw.quantization in ("awq", "gptq"):
            kwargs["quantization"] = hw.quantization

        if self._schema.spec_decoding_enabled:
            kwargs["speculative_config"] = {
                "method": "ngram",
                "num_speculative_tokens": self._schema.spec_decoding_num_tokens,
                "prompt_lookup_max": self._schema.spec_decoding_ngram_max,
                "prompt_lookup_min": self._schema.spec_decoding_ngram_min,
            }
        return AsyncEngineArgs(**kwargs)

    async def initialize(self) -> None:
        self._engine = AsyncLLMEngine.from_engine_args(self._engine_args())

    def _sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=0.0,
            max_tokens=self._schema.decode_length,
            logprobs=self._schema.logprobs_k,
            skip_special_tokens=True,
        )

    def encode_text(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def build_prompt_token_ids(
        self,
        query: str,
        passages_per_round: list[list[str]],
        completed_chunks: list[str],
    ) -> list[int]:
        prompt = self.build_prompt(
            query=query,
            passages_per_round=passages_per_round,
            completed_chunks=completed_chunks,
        )
        return self._tokenizer.encode(prompt, add_special_tokens=False)

    def build_prompt(
        self,
        query: str,
        passages_per_round: list[list[str]],
        completed_chunks: list[str],
    ) -> str:
        if len(passages_per_round) != len(completed_chunks) + 1:
            raise ValueError(
                "passages_per_round must have exactly one more entry than completed_chunks: "
                f"got {len(passages_per_round)} vs {len(completed_chunks)}"
            )

        initial_passages = passages_per_round[0]
        initial_block = "\n".join(
            f"  [0.{j+1}] {p.strip()}" for j, p in enumerate(initial_passages)
        ) or "(no passages retrieved)"

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"[Initial Context]\n{initial_block}\n\nQuestion: {query}",
            },
        ]

        for i, chunk_text in enumerate(completed_chunks):
            messages.append({"role": "assistant", "content": chunk_text})
            new_round_idx = i + 1
            new_passages = passages_per_round[new_round_idx]
            evidence_block = "\n".join(
                f"  [{new_round_idx}.{j+1}] {p.strip()}"
                for j, p in enumerate(new_passages)
            ) or "(no new passages retrieved)"
            messages.append({
                "role": "user",
                "content": f"[Retrieved Evidence Round {new_round_idx}]\n{evidence_block}",
            })

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def generate_tokens(
        self,
        prompt,
        state: SequenceState,
    ) -> AsyncIterator[TokenOutput]:
        params = self._sampling_params()
        state._prev_n_tokens = 0
        state.generated_token_ids.clear()

        if isinstance(prompt, list):
            engine_prompt = {"prompt_token_ids": prompt}
        else:
            engine_prompt = prompt

        async for output in self._engine.generate(
            prompt=engine_prompt,
            sampling_params=params,
            request_id=state.request_id,
        ):
            if not output.outputs:
                continue
            comp: CompletionOutput = output.outputs[0]
            if not comp.token_ids:
                continue

            start = state._prev_n_tokens
            stop = len(comp.token_ids)
            if stop <= start:
                continue

            full_text = comp.text
            prev_text_len = state._prev_text_len
            new_text = full_text[prev_text_len:]
            state._prev_text_len = len(full_text)
            state._prev_n_tokens = stop

            for i in range(start, stop):
                tid = comp.token_ids[i]
                state.generated_token_ids.append(tid)
                state.token_index += 1

                is_last_in_batch = (i == stop - 1)
                delta_text = new_text if is_last_in_batch else ""
                is_final = output.finished and is_last_in_batch

                yield TokenOutput(
                    token_id=tid,
                    delta_text=delta_text,
                    logprobs=_logprobs_at(comp.logprobs, i),
                    is_final=is_final,
                    is_last_in_batch=is_last_in_batch,
                    request_id=state.request_id,
                )

            if output.finished:
                break

    async def abort(self, request_id: str) -> None:
        await self._engine.abort(request_id)

    def new_request_id(self) -> str:
        return str(uuid.uuid4())

    async def shutdown(self) -> None:
        pass
