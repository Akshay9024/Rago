from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from config.schema import RAGSchema


class DocumentEmbedder:
    def __init__(self, schema: RAGSchema, device: str = "cpu"):
        self._model = SentenceTransformer(
            schema.encoder_id,
            device=device,
            trust_remote_code=True,
        )
        self._dim = schema.encoder_dim
        self._device = device

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        normalize: bool = True,
    ) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]

    def encode_batch(self, queries: list[str]) -> np.ndarray:
        return self.encode(queries)

    @property
    def dim(self) -> int:
        return self._dim
