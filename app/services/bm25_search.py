from uuid import UUID
import asyncio
import re
from typing import Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.vector_service import get_vector_service
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
FETCH_MULTIPLIER = 5      # fetch 5x what we'll return from each source
RRF_K = 60                # standard RRF constant


class BM25SearchService:
    """
    Hybrid retrieval: Qdrant vector search + PostgreSQL full-text search,
    fused with Reciprocal Rank Fusion.

    Requires document_chunk.content to be stored in the DB (it is).
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.vectors = get_vector_service()

    async def _bm25_search(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[dict]:
        if not query.strip():
            return []

        doc_filter = ""
        params: dict = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
        }
        if document_ids:
            doc_filter = "AND c.document_id = ANY(:doc_ids)"
            params["doc_ids"] = document_ids

        sql = text(f"""
            SELECT
                c.id::text                          AS chunk_id,
                c.document_id::text                 AS document_id,
                c.chunk_index                       AS chunk_index,
                c.content                           AS content,
                c.metadata_                         AS metadata,
                ts_rank_cd(
                    to_tsvector('english', c.content),
                    plainto_tsquery('english', :query),
                    32
                )                                   AS score
            FROM
                document_chunk c
                JOIN documents d ON d.id = c.document_id
            WHERE
                d.owner_id = :user_id
                {doc_filter}
                AND to_tsvector('english', c.content)
                    @@ plainto_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :top_k
        """)

        try:
            result = await self.db.execute(sql, params)
            rows = result.mappings().all()
        except Exception:
            logger.exception("BM25 search failed")
            return []

        if not rows:
            return []

        max_score = max(r["score"] for r in rows) or 1.0
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata": row["metadata"] or {},
                "score": float(row["score"]) / max_score,
            }
            for row in rows
        ]

    async def search(
        self,
        query_embedding: list[float],
        query_text: str,
        user_id: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[dict]:
        fetch_k = top_k * FETCH_MULTIPLIER

        vector_results, bm25_results = await asyncio.gather(
            self.vectors.search(
                query_embedding=query_embedding,
                user_id=user_id,
                document_ids=document_ids,
                top_k=fetch_k,
                score_threshold=0.0,   # no pre-filter — RRF handles ranking
            ),
            self._bm25_search(
                query=query_text,
                user_id=user_id,
                document_ids=document_ids,
                top_k=fetch_k,
            ),
        )

        logger.debug(
            "hybrid search: vector=%d bm25=%d",
            len(vector_results), len(bm25_results),
        )

        fused = self._rrf(vector_results, bm25_results)

        # Relevance filter:
        #  - vector hits must clear SIMILARITY_THRESHOLD (real cosine score)
        #  - BM25-only hits (score is None) are kept — they matched on keywords
        fused = [
            c for c in fused
            if c["score"] is None or c["score"] >= settings.SIMILARITY_THRESHOLD
        ]

        return fused[:top_k]
    
    def _rrf(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = RRF_K,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion of vector + BM25 results.

        Sets TWO scores per chunk:
        - rrf_score: rank-based fusion score (0-1). Use for ORDERING.
        - score:     original cosine similarity from Qdrant (0-1),
                    or None for BM25-only hits. Use for THRESHOLDING
                    and for reporting to API consumers.
        """
        scores: dict[str, float] = {}
        chunks: dict[str, dict] = {}
        vector_scores: dict[str, float] = {}   # ← NEW

        for rank, chunk in enumerate(vector_results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + VECTOR_WEIGHT / (k + rank)
            chunks.setdefault(cid, chunk)
            vector_scores[cid] = float(chunk.get("score", 0.0))   # ← capture

        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + BM25_WEIGHT / (k + rank)
            chunks.setdefault(cid, chunk)

        if not scores:
            return []

        max_score = max(scores.values())

        results = []
        for cid, raw_score in scores.items():
            chunk = dict(chunks[cid])
            chunk["rrf_score"] = raw_score / max_score          # for sorting
            # for relevance (None for BM25-only)
            chunk["score"] = vector_scores.get(cid)
            results.append(chunk)

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results
