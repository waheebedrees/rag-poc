import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import tiktoken

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    content: str
    index: int
    token_count: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextSplitter:

    def __init__(self, chunk_size_tokens: int = 500, overlap_tokens: int = 100):
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be less than "
                f"chunk_size_tokens ({chunk_size_tokens})"
            )
        self.chunk_size = chunk_size_tokens
        self.overlap_token = overlap_tokens
        self.separators = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", None]

        self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_token(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def split(self, text: str, base_metadata: dict | None = None) -> list[Chunk]:
        if not text.strip():
            return []
        raw_chunks = self._recursive_split(text, self.separators)
        merged = self._apply_overlap(raw_chunks)
        return self._to_chunks(merged, base_metadata or {}, text)

    def _recursive_split(self, text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text] if text.strip() else []

        sep = seps[0]
        rest = seps[1:]
        if sep is None:
            return self._hard_split_by_tokens(text)
                                          
        parts = [p for p in text.split(sep) if p.strip()]


        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if self.count_token(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if self.count_token(part) > self.chunk_size:
                    # extend flattens the recursive result
                    chunks.extend(self._recursive_split(part, rest))
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current)
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1 or self.overlap_token <= 0:
            return chunks

        result: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prefix = self._get_overlap_tail(result[-1])
            merged = (prefix + " " + chunks[i]
                      ).strip() if prefix else chunks[i]
            if self.count_token(merged) <= self.chunk_size:
                result.append(merged)
            else:
                result.append(chunks[i])
        return result

    def _get_overlap_tail(self, text: str) -> str:
        tokens = self.encoder.encode(text)
        if len(tokens) <= self.overlap_token:
            return text
        return self.encoder.decode(tokens[-self.overlap_token:])

    def _to_chunks(
        self, raw: list[str], base_meta: dict, source: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        cursor = 0
        for idx, text in enumerate(raw):
            start = source.find(text, cursor)
            if start == -1:
                start = cursor
            end = start + len(text)
            chunks.append(
                Chunk(
                    content=text,
                    index=idx,
                    token_count=self.count_token(text),
                    char_start=start,
                    char_end=end,
                    metadata={**base_meta, "chunk_index": idx},
                )
            )
            cursor = start + max(1, len(text) - 1)
        return chunks


    def _hard_split_by_tokens(self, text: str) -> list[str]:
        """
        Last-resort split for text with no separators at all.

        Encodes to tokens and slices at chunk_size boundaries, then decodes
        each slice back to text. Used only when all semantic separators
        have been exhausted.
        """
        tokens = self.encoder.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text] if text.strip() else []

        out: list[str] = []
        for i in range(0, len(tokens), self.chunk_size):
            piece = self.encoder.decode(tokens[i:i + self.chunk_size])
            if piece.strip():
                out.append(piece)
        return out
