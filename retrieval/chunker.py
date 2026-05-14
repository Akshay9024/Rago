from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str
    start_char: int
    end_char: int


class SemanticChunker:
    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def _sentence_split(self, text: str) -> list[str]:
        parts = self._SENTENCE_END.split(text.strip())
        return [p.strip() for p in parts if p.strip()]

    def chunk_document(self, text: str, source: str) -> list[Chunk]:
        sentences = self._sentence_split(text)
        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_tokens = 0
        char_offset = 0

        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)

            if buf and buf_tokens + word_count > self._chunk_size:
                chunk_text = " ".join(buf)
                chunk_id = hashlib.sha256(f"{source}:{chunk_text[:64]}".encode()).hexdigest()[:16]
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                ))
                char_offset += len(chunk_text) + 1

                overlap_words = buf[-self._overlap:] if self._overlap > 0 else []
                buf = overlap_words
                buf_tokens = len(overlap_words)

            buf.extend(words)
            buf_tokens += word_count

        if buf:
            chunk_text = " ".join(buf)
            chunk_id = hashlib.sha256(f"{source}:{chunk_text[:64]}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_id=chunk_id,
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
            ))

        return chunks

    def chunk_corpus(self, documents: list[tuple[str, str]]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for text, source in documents:
            all_chunks.extend(self.chunk_document(text, source))
        return all_chunks
