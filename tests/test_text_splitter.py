
from __future__ import annotations

import pytest

from app.utils.text_splitter import Chunk, TextSplitter



def _token_count(splitter: TextSplitter, text: str) -> int:
    """Use the splitter's own tokenizer so tests stay in sync with the impl."""
    return splitter.count_token(text)


def _make_long_text(sentence: str, repeats: int) -> str:
    return (sentence.strip() + " ") * repeats



class TestChunkDataclass:
    def test_fields(self):
        c = Chunk(
            content="hello",
            index=0,
            token_count=1,
            char_start=0,
            char_end=5,
            metadata={"k": "v"},
        )
        assert c.content == "hello"
        assert c.index == 0
        assert c.token_count == 1
        assert c.char_start == 0
        assert c.char_end == 5
        assert c.metadata == {"k": "v"}

    def test_metadata_defaults_to_empty_dict(self):
        c = Chunk(content="x", index=0, token_count=1,
                  char_start=0, char_end=1)
        assert c.metadata == {}

    def test_default_metadata_is_not_shared(self):
        """Mutation safety — each Chunk gets its own dict."""
        c1 = Chunk(content="a", index=0, token_count=1,
                   char_start=0, char_end=1)
        c2 = Chunk(content="b", index=1, token_count=1,
                   char_start=1, char_end=2)
        c1.metadata["x"] = 1
        assert c2.metadata == {}



class TestConstructor:
    def test_defaults(self):
        s = TextSplitter()
        assert s.chunk_size == 500
        assert s.overlap_token == 100

    def test_custom_sizes(self):
        s = TextSplitter(chunk_size_tokens=100, overlap_tokens=20)
        assert s.chunk_size == 100
        assert s.overlap_token == 20

    def test_overlap_equal_to_capacity_raises(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size_tokens=50, overlap_tokens=50)

    def test_overlap_greater_than_capacity_raises(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size_tokens=50, overlap_tokens=60)



class TestEdgeCases:
    def test_empty_string_returns_no_chunks(self):
        s = TextSplitter(chunk_size_tokens=50, overlap_tokens=10)
        assert s.split("") == []

    def test_whitespace_only_returns_no_chunks(self):
        s = TextSplitter(chunk_size_tokens=50, overlap_tokens=10)
        assert s.split("   ") == []
        assert s.split("\n\n\n") == []
        assert s.split("\t\t") == []

    def test_single_word_fits_in_one_chunk(self):
        s = TextSplitter(chunk_size_tokens=50, overlap_tokens=10)
        chunks = s.split("Hello")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello"
        assert chunks[0].index == 0

    def test_short_text_is_one_chunk(self):
        s = TextSplitter(chunk_size_tokens=500, overlap_tokens=50)
        text = "Short text that fits easily."
        chunks = s.split(text)
        assert len(chunks) == 1
        assert chunks[0].content == text



class TestCapacity:
    def test_no_chunk_exceeds_capacity(self):
        s = TextSplitter(chunk_size_tokens=50, overlap_tokens=10)
        text = "This is a sentence. " * 200
        chunks = s.split(text)

        assert len(chunks) > 0
        for c in chunks:
            assert c.token_count <= 50, (
                f"chunk {c.index} has {c.token_count} tokens, exceeds 50"
            )

    def test_no_chunk_exceeds_capacity_with_no_separators(self):
        """Words separated only by spaces — stresses the last-resort separator."""
        s = TextSplitter(chunk_size_tokens=30, overlap_tokens=5)
        text = " ".join(f"w{i}" for i in range(500))
        chunks = s.split(text)
        for c in chunks:
            assert c.token_count <= 30

    def test_long_single_token_falls_back_gracefully(self):
        """
        A single 'word' longer than capacity cannot be split semantically.
        The splitter should not crash and should still produce chunks.
        """
        s = TextSplitter(chunk_size_tokens=20, overlap_tokens=5)
        text = "x" * 2000  # no separators at all
        chunks = s.split(text)
        assert len(chunks) >= 1
        # Reassembling chunks should contain all the original chars
        joined = "".join(c.content for c in chunks)
        assert joined.count("x") >= 2000 - 100  # allow overlap slack



class TestOverlap:
    def test_consecutive_chunks_share_content(self):
        s = TextSplitter(chunk_size_tokens=50, overlap_tokens=10)
        text = "Alpha bravo charlie delta echo foxtrot golf hotel india. " * 40
        chunks = s.split(text)

        assert len(chunks) >= 2, "test needs at least two chunks"

        # Grab the last few words of chunk 0 and look for them in chunk 1
        tail_words = chunks[0].content.split()[-5:]
        head_words = chunks[1].content.split()[:20]
        # At least one of the tail words should appear in the head of the next chunk
        assert any(w in head_words for w in tail_words), (
            "consecutive chunks do not share overlap content"
        )

    def test_overlap_does_not_overwrite_previous_chunk(self):
        """
        Regression test: an old bug did `result[-1] = merged`, which silently
        threw away most of the previous chunk's content.
        """
        s = TextSplitter(chunk_size_tokens=100, overlap_tokens=20)
        text = "word " * 500
        chunks = s.split(text)

        # Total characters across all chunks should be >= the source length
        # (overlap means it can only be bigger, never smaller)
        total_chars = sum(len(c.content) for c in chunks)
        assert total_chars >= len(text.strip()) - 5, (
            "content was lost during overlap merging"
        )



class TestOrdering:
    def test_indices_are_sequential(self):
        s = TextSplitter(chunk_size_tokens=30, overlap_tokens=5)
        text = "one two three four five six seven eight nine ten. " * 30
        chunks = s.split(text)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_content_preserves_source_order(self):
        s = TextSplitter(chunk_size_tokens=40, overlap_tokens=5)
        text = "APPLE BANANA CHERRY DATE ELDERBERRY FIG GRAPE. " * 20
        chunks = s.split(text)

        # The first occurrence of APPLE should appear before BANANA
        # in the concatenated chunk stream
        concat = " ".join(c.content for c in chunks)
        assert concat.find("APPLE") < concat.find("BANANA")
        assert concat.find("BANANA") < concat.find("CHERRY")



class TestCharOffsets:
    def test_offsets_slice_back_to_original(self):
        """
        Regression test: an old bug computed char_start/char_end by
        accumulating len(text), which drifted once overlap was applied.
        """
        s = TextSplitter(chunk_size_tokens=30, overlap_tokens=5)
        text = "The quick brown fox jumps over the lazy dog. " * 30
        chunks = s.split(text)

        for c in chunks:
            sliced = text[c.char_start:c.char_end]
            # The slice may differ from content by whitespace trimming,
            # but the content must be findable starting at char_start
            assert c.content.strip() in sliced or sliced.strip() == c.content.strip(), (
                f"chunk {c.index} offsets do not match content\n"
                f"  content: {c.content[:60]!r}\n"
                f"  slice:   {sliced[:60]!r}"
            )

    def test_offsets_are_monotonic(self):
        s = TextSplitter(chunk_size_tokens=40, overlap_tokens=5)
        text = "Sentence number one. Sentence number two. " * 40
        chunks = s.split(text)
        for prev, curr in zip(chunks, chunks[1:]):
            assert curr.char_start >= prev.char_start

    def test_offsets_handle_multibyte_characters(self):
        """UTF-8 multibyte chars must not break offset arithmetic."""
        s = TextSplitter(chunk_size_tokens=20, overlap_tokens=3)
        text = "café résumé naïve coöperate " * 30
        chunks = s.split(text)

        for c in chunks:
            sliced = text[c.char_start:c.char_end]
            # Content chars should all appear in the slice
            for ch in c.content.strip()[:5]:
                if ch.isalpha():
                    assert ch in sliced



class TestMetadata:
    def test_base_metadata_flows_through(self):
        s = TextSplitter(chunk_size_tokens=30, overlap_tokens=5)
        chunks = s.split(
            "Hello world. " * 40,
            base_metadata={"document_id": "abc-123"},
        )
        for c in chunks:
            assert c.metadata["document_id"] == "abc-123"

    def test_chunk_index_is_added_to_metadata(self):
        s = TextSplitter(chunk_size_tokens=30, overlap_tokens=5)
        chunks = s.split("Hello world. " * 40)
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i

    def test_chunk_index_overrides_base_metadata_on_collision(self):
        """Implementation-defined: internal chunk_index wins over base."""
        s = TextSplitter(chunk_size_tokens=30, overlap_tokens=5)
        chunks = s.split(
            "Hello world. " * 40,
            base_metadata={"chunk_index": 999},
        )
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i

    def test_none_metadata_is_allowed(self):
        s = TextSplitter(chunk_size_tokens=30, overlap_tokens=5)
        chunks = s.split("Hello world. " * 40, base_metadata=None)
        assert len(chunks) > 0
        for c in chunks:
            assert "chunk_index" in c.metadata



class TestTokenCounting:
    def test_count_token_matches_chunk_token_count(self):
        s = TextSplitter(chunk_size_tokens=50, overlap_tokens=10)
        chunks = s.split("Repeat after me. " * 100)
        for c in chunks:
            assert c.token_count == s.count_token(c.content)

    def test_count_token_is_monotonic(self):
        s = TextSplitter()
        short = s.count_token("hello")
        longer = s.count_token("hello world how are you")
        assert longer > short

    def test_count_token_empty_string(self):
        s = TextSplitter()
        assert s.count_token("") == 0



class TestSeparators:
    def test_paragraph_break_is_preferred(self):
        """
        When text has paragraph breaks, chunks should respect them
        rather than cutting mid-sentence.
        """
        s = TextSplitter(chunk_size_tokens=100, overlap_tokens=10)
        paragraphs = [
            "First paragraph. " * 5,
            "Second paragraph. " * 5,
            "Third paragraph. " * 5,
        ]
        text = "\n\n\n".join(paragraphs)
        chunks = s.split(text)

        # No chunk should start mid-sentence within a paragraph —
        # the first chunk should start with "First", etc.
        assert chunks[0].content.startswith("First")

    def test_splits_on_sentence_boundary_when_possible(self):
        s = TextSplitter(chunk_size_tokens=25, overlap_tokens=3)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 10
        chunks = s.split(text)

        # Chunks should preferentially end at sentence boundaries
        # (i.e. with a period or after one), not mid-word
        for c in chunks[:-1]:  # last chunk can end anywhere
            # Allow chunks ending in a period or with whitespace after one
            ends_cleanly = c.content.rstrip().endswith((".", "!", "?"))
            # Or the boundary was forced by a very long sentence
            assert ends_cleanly or c.token_count >= s.chunk_size - 5
