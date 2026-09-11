from functools import lru_cache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    UpdateStatus,
    PayloadSchemaType,
)
from qdrant_client import AsyncQdrantClient
from typing import Any, Optional
import uuid


from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


class VectorService:
    def __init__(self):
        self.qdrant_client = None
        self._collection = settings.QDRANT_COLLECTION_NAME
        self._dims = settings.VECTOR_DIM       
        self.qdrant_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            check_compatibility=False,      

            port=settings.QDRANT_PORT,
            timeout=50,
        )
        self._collection = settings.QDRANT_COLLECTION_NAME
        self._dims = settings.VECTOR_DIM

    async def ensure_collection(self):
        collections = await self.qdrant_client.get_collections()
        names = {c.name for c in collections.collections}
        if self._collection in names:
            return

        await self.qdrant_client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._dims, distance=Distance.COSINE
            ),
        )
        for field in ("document_id", "user_id"):
            await self.qdrant_client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info("Created Qdrant collection: %s", self._collection)

    async def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        document_id: str,
        user_id: str,
    ) -> list[str]:
        points = []
        vector_ids = []
        for chunk , emd in zip(chunks, embeddings):
            vid = str(uuid.uuid4())
            vector_ids.append(vid)
            payload = {
                'document_id': document_id,
                'user_id':user_id,
                'chunk_id':str(chunk['id']),
                'content': chunk['content'],
                'chunk_index': chunk['chunk_index'],
                "metadata": chunk.get("metadata", {}),
            }
            points.append(PointStruct(id=vid, vector=emd, payload=payload))
    
        for i in range(0, len(points), 100):
            batch = points[i:i+100]
            result = await self.qdrant_client.upsert(
                collection_name=self._collection,
                points=batch,
                wait=True
            )
            if result.status != UpdateStatus.COMPLETED:
                raise RuntimeError(f"Qdrant upsert failed: {result.status}")

        return vector_ids    
        
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    async def search(
        self,
        query_embedding: list[float],
        user_id: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        if document_ids:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=document_ids),
                )
            )
            
        results = await self.qdrant_client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            query_filter=Filter(must=conditions),
            
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {
                "vector_id": str(hit.id),
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "document_id": hit.payload.get("document_id"),
                "chunk_id": hit.payload.get("chunk_id"),
                "chunk_index": hit.payload.get("chunk_index"),
                "metadata": hit.payload.get("metadata", {}),
            }
            for hit in results.points
        ]
        
    async def delete_by_document(self, document_id: str):
        await self.qdrant_client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )


@lru_cache()
def get_vector_service() -> VectorService:
    return VectorService()