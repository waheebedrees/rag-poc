import pytest
from app.services.bm25_search import BM25SearchService
from app.services.vector_service import get_vector_service




class TestRRF:
    """
    RRF now produces TWO scores per chunk:
      - rrf_score: rank-based fusion score, normalized to [0, 1], always
                   present. Top-ranked result gets 1.0.
      - score:     original vector cosine similarity, or None for
                   BM25-only hits.
    """

    def _svc(self):
        return BM25SearchService.__new__(BM25SearchService)

    def test_rrf_only_vector(self):
        svc = self._svc()
        v = [{"chunk_id": "x", "content": "x", "score": 0.85}]
        fused = svc._rrf(v, [])

        assert len(fused) == 1
        assert fused[0]["chunk_id"] == "x"
        # rank-based score: single result at rank 1 → normalized to 1.0
        assert fused[0]["rrf_score"] == 1.0
        # cosine score preserved from the vector result
        assert fused[0]["score"] == 0.85

    def test_rrf_only_bm25(self):
        """BM25-only hits keep score=None so they can bypass the cosine filter."""
        svc = self._svc()
        b = [{"chunk_id": "z", "content": "z"}]
        fused = svc._rrf([], b)

        assert len(fused) == 1
        assert fused[0]["chunk_id"] == "z"
        assert fused[0]["rrf_score"] == 1.0
        assert fused[0]["score"] is None

    def test_rrf_scores_normalized_to_one(self):
        """rrf_score is always in [0, 1] with the top result at 1.0."""
        svc = self._svc()
        v = [
            {"chunk_id": "x", "score": 0.9},
            {"chunk_id": "y", "score": 0.5},
        ]
        b = [{"chunk_id": "z"}]
        fused = svc._rrf(v, b)

        rrf_scores = [c["rrf_score"] for c in fused]
        assert max(rrf_scores) == 1.0
        assert all(0.0 <= s <= 1.0 for s in rrf_scores)

    def test_rrf_boosts_items_in_both_lists(self):
        """A chunk appearing in both vector and BM25 lists outranks ones
        that appear in only one."""
        svc = self._svc()
        vector = [
            {"chunk_id": "a", "score": 0.9},
            {"chunk_id": "b", "score": 0.8},
        ]
        bm25 = [
            {"chunk_id": "b"},
            {"chunk_id": "c"},
        ]
        fused = svc._rrf(vector, bm25)

        # b is in both lists → should rank first
        assert fused[0]["chunk_id"] == "b"
        # b's cosine score (from vector) is preserved
        assert fused[0]["score"] == 0.8

    def test_rrf_empty_inputs(self):
        svc = self._svc()
        assert svc._rrf([], []) == []

    def test_rrf_preserves_vector_cosine(self):
        """The cosine score reported by _rrf is the raw vector score, not
        the rank-based normalization."""
        svc = self._svc()
        v = [
            {"chunk_id": "a", "score": 0.62},
            {"chunk_id": "b", "score": 0.41},
        ]
        fused = svc._rrf(v, [])

        by_id = {c["chunk_id"]: c for c in fused}
        assert by_id["a"]["score"] == 0.62
        assert by_id["b"]["score"] == 0.41


    def test_rrf_only_vector_no_cosine(self):
        """
        A vector result without a 'score' key defaults to 0.0 (unknown similarity
        → will fail the cosine threshold). Only BM25-only hits get score=None.
        """
        svc = self._svc()
        v = [{"chunk_id": "x", "content": "x"}]   
        fused = svc._rrf(v, [])

        assert fused[0]["rrf_score"] == 1.0
        assert fused[0]["score"] == 0.0         

