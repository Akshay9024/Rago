from __future__ import annotations
from typing import Union
from transformers import AutoTokenizer


class TokenizerWrapper:
    def __init__(self, model_id: str):
        self._tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
        )
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special)

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        return self._tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size
