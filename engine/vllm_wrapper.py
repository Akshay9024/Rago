from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
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
    spec_suppressed_until: int = 0
    _prev_n_tokens: int = field(default=0, repr=False)


@dataclass
class TokenOutput:
    token_id: int
    delta_text: str
    logprobs: dict[int, float]
    is_final: bool
    request_id: str


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

    def _engine_args(self) -> AsyncEngineArgs:
        hw = self._config.hardware
        kwargs: dict = dict(
            model=self._schema.generative_llm_id,
            dtype="auto",
            gpu_memory_utilization=self._schema.gpu_memory_utilization,
            tensor_parallel_size=hw.tensor_parallel_size,
            enable_prefix_caching=self._schema.apc_enabled,
            max_model_len=self._schema.max_model_len,
            trust_remote_code=True,
            disable_log_requests=True,
            max_num_batched_tokens=self._schema.max_model_len,
            max_num_seqs=hw.decode_batch_size,
        )
        if hw.quantization == "bitsandbytes":
            kwargs["quantization"] = "bitsandbytes"
            kwargs["load_format"] = "bitsandbytes"
        elif hw.quantization in ("awq", "gptq"):
            kwargs["quantization"] = hw.quantization
        return AsyncEngineArgs(**kwargs)

    async def initialize(self) -> None:
        self._engine = AsyncLLMEngine.from_engine_args(self._engine_args())

    def _sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=0.0,
            max_tokens=self._schema.decode_length,
            logprobs=32,
            skip_special_tokens=True,
        )

    def encode_text(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def build_prompt(
        self,
        query: str,
        initial_context: str,
        retrieved_rounds: list[list[str]],
        generated_so_far: str,
    ) -> str:
        ctx_block = f"[Initial Context]\n{initial_context}"
        for ri, passages in enumerate(retrieved_rounds, start=1):
            joined = "\n".join(f"  [{ri}.{j+1}] {p.strip()}" for j, p in enumerate(passages))
            ctx_block += f"\n\n[Retrieved Evidence Round {ri}]\n{joined}"

        user_content = f"{ctx_block}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        if generated_so_far:
            messages.append({"role": "assistant", "content": generated_so_far})
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def generate_tokens(
        self,
        prompt: str,
        state: SequenceState,
    ) -> AsyncIterator[TokenOutput]:
        params = self._sampling_params()
        state._prev_n_tokens = 0
        state.generated_token_ids.clear()

        async for output in self._engine.generate(
            prompt=prompt,
            sampling_params=params,
            request_id=state.request_id,
        ):
            if not output.outputs:
                continue
            comp: CompletionOutput = output.outputs[0]
            if not comp.token_ids:
                continue

            n_new = len(comp.token_ids) - state._prev_n_tokens
            if n_new <= 0:
                continue

            new_token_id = comp.token_ids[-1]
            state.generated_token_ids.append(new_token_id)
            state.token_index += 1

            full_text = comp.text
            prev_len = getattr(state, "_prev_text_len", 0)
            delta = full_text[prev_len:]
            state._prev_text_len = len(full_text)  # type: ignore[attr-defined]
            state._prev_n_tokens = len(comp.token_ids)

            logprobs: dict[int, float] = {}
            if comp.logprobs and len(comp.logprobs) >= len(comp.token_ids):
                entry = comp.logprobs[len(comp.token_ids) - 1]
                if entry:
                    logprobs = {tid: lp.logprob for tid, lp in entry.items()}

            yield TokenOutput(
                token_id=new_token_id,
                delta_text=delta,
                logprobs=logprobs,
                is_final=output.finished,
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
