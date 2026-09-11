
from typing import Any, Optional, List
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import func

from app.config import settings
from app.services.errors import ConversationNotFoundError
from app.models import Conversation, Document, Message, MessageRole

class ConversationService:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_conversation(
        self, conversation_id: UUID, user_id: UUID
    ) -> Optional[Conversation]:
        result = await self.db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create_conversation(self, user_id: UUID, conversation_id: Optional[UUID]):
        if conversation_id:
            conv = await self.get_conversation(conversation_id, user_id)
            if not conv:
                raise ConversationNotFoundError("Conversation not found")     # ← was ValueError
            return conv

            return conv
        conv = Conversation(user_id=user_id)
        self.db.add(conv)
        await self.db.flush()
        return conv

    async def delete_conversation(
        self, conversation_id: UUID, user_id: UUID
    ) -> bool:
        conv = await self.get_conversation(conversation_id, user_id)
        if not conv:
            return False
        await self.db.delete(conv)
        await self.db.commit()
        return True

    async def list_conversations(
        self, user_id: UUID, page: int = 1, page_size: int = 20
    ) -> List[dict[str, Any]]:

        result = await self.db.execute(
            select(
                Conversation,
                func.count(Message.id).label("message_count"),
            )
            .outerjoin(
                Message,
                Message.conversation_id == Conversation.id 
            )
            .where(Conversation.user_id == user_id)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = result.all()
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "message_count": count,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
            }
            for conv, count in rows
        ]
    
    async def get_recent_history(
        self, conversation_id: UUID, user_id: UUID,
    ) -> list[dict[str, str]]:
        """Bounded fetch — only retrieves the N most recent messages."""
        result = await self.db.execute(
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Message.role != MessageRole.SYSTEM,
            )
            .order_by(Message.created_at.desc())
            .limit(settings.MAX_HISTORY_MESSAGES)
        )
        messages = list(reversed(result.scalars().all()))
        return [
            {"role": m.role.value, "content": m.content} for m in messages
        ]

    async def build_sources(self, chunks: list[dict]) -> list[dict[str, Any]]:
        if not chunks:
            return []

        doc_ids = list({c["document_id"] for c in chunks})
        result = await self.db.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
        docs = {str(d.id): d for d in result.scalars().all()}

        sources = []
        for c in chunks:
            doc = docs.get(c["document_id"])
            score = c.get("score")
            sources.append({
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "document_title": doc.title if doc else "Unknown",
                "content_preview": c["content"][:300],
                "similarity_score": round(score, 4) if score is not None else None,
            })
        return sources
