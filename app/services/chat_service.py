import asyncio
import re
from typing import Any, Optional, List, Dict
from uuid import UUID
import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.models import Message, MessageRole
from app.services.llm.llm_service import get_llm_service, LLMMessage
from app.services.vector_service import get_vector_service
from app.security.filters import SecurityFilter, Severity
from app.services.conversation_service import ConversationService
from app.utils.logger import get_logger
from app.services.embeddings import get_embedding_service
from app.prompts import SYSTEM_PROMPT
from app.services.bm25_search import BM25SearchService
from app.services.errors import ChatError, ConversationNotFoundError



logger = get_logger(__name__)


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedder = get_embedding_service()
        self.vectors = get_vector_service()
        self.llm = get_llm_service()
        self.conv_service = ConversationService(db=self.db)
        self.search = BM25SearchService(self.db)
        self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        self.security_filter = SecurityFilter()


    def _build_context(self, chunks: list[dict]) -> str:
        """Build context string respecting a token budget."""
        if not chunks:
            return ""

        parts: list[str] = []
        total_tokens = 0

        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = len(self.encoder.encode(chunk["content"]))
            if total_tokens + chunk_tokens > settings.MAX_CONTEXT_TOKENS:
                break
            score = chunk.get("score")
            score_str = f"{score:.2f}" if score is not None else "bm25"
            parts.append(
                f"[Source {i}] (score: {score_str})\n{chunk['content']}"
            )
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(parts)
        
    def _generate_title(self, message: str) -> str:
        if not message.strip():
            return 'no title'
        return message[:50]

    def deduplicate(self, chunks: list[dict], threshold: float = 0.70) -> list[dict]:
        """Drop chunks whose text is near-identical to a higher-ranked chunk."""
        import re

        def tokens(text: str) -> set[str]:
            return set(re.findall(r"\w+", text.lower()))

        seen: list[set[str]] = []
        unique: list[dict] = []

        for chunk in chunks:
            toks = tokens(chunk["content"])
            if not toks:
                continue
            if any(
                len(toks & prev) / max(len(toks | prev), 1) >= threshold
                for prev in seen
            ):
                continue
            seen.append(toks)
            unique.append(chunk)

        return unique

    def _build_query_text(self, message: str, history: list[dict], max_turns: int = 4) -> str:
        if not history:
            return message

        _FOLLOWUP_PRONOUNS = re.compile(
            r"\b(it|its|that|this|those|these|they|them|their|he|him|his|she|her)\b",
            re.IGNORECASE,
        )

        # Standalone questions are self-contained; don't dilute them.
        is_short = len(message.split()) <= 8
        has_pronoun = bool(_FOLLOWUP_PRONOUNS.search(message))
        if not (is_short and has_pronoun):
            return message

        recent = history[-max_turns:]
        prefix = " ".join(f"{h['role']}: {h['content']}" for h in recent)
        return f"{prefix}\n\nCurrent question: {message}"

    def _build_messages(self, user_query, context, history):
        system_parts = [SYSTEM_PROMPT]
        if context:
            system_parts.append(f"Document context:\n\n{context}")
        else:
            system_parts.append(
                "No relevant context was found. If the user's question requires "
                "document knowledge, say so rather than guessing."
            )
        system_content = "\n\n---\n\n".join(system_parts)

        msgs = [LLMMessage(role="system", content=system_content)]
        for h in history:
            msgs.append(LLMMessage(role=h["role"], content=h["content"]))
        msgs.append(LLMMessage(role="user", content=user_query))
        return msgs

    async def chat(
        self,
        user_id: UUID,
        message: str,
        conversation_id: Optional[UUID] = None,
        document_ids: Optional[list[UUID]] = None,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        model = model or settings.DEFAULT_LLM_MODEL
        provider = provider or settings.DEFAULT_LLM_PROVIDER

        security = self.security_filter.check_query(message)
        if not security.is_safe:
            logger.warning(
                "chat.query_blocked rule=%s preview=%s",
                security.rule,
                security.redacted[:200],
            )
            raise ChatError("Query blocked by security filter")

        try:
            conversation = await self.conv_service.get_or_create_conversation(
                user_id, conversation_id
            )
        except ConversationNotFoundError as e:
            raise ChatError(str(e)) from e
        
    
        # Fetch prior turns BEFORE persisting the new user message
        history = await self.conv_service.get_recent_history(conversation.id, user_id)
        user_msg = Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=message,
        )
        self.db.add(user_msg)

        if not conversation.title:
            conversation.title = self._generate_title(message)

        await self.db.commit()    

        #  everything below can fail without losing the question 
        query_text = self._build_query_text(message, history)
        query_embedding = await self.embedder.embed_one(query_text)

        doc_id_strs = [str(d) for d in document_ids] if document_ids else None

        chunks = await self.search.search(
            query_embedding=query_embedding,
            query_text=query_text,
            user_id=str(user_id),
            document_ids=doc_id_strs,
            top_k=settings.TOP_K_RESULTS,
        )

        # filter chunks for injection
        chunk_results = self.security_filter.check_chunks(
            [c["content"] for c in chunks]
        )
        safe_chunks = [
            c for c, r in zip(chunks, chunk_results)
            if r.severity is not Severity.BLOCK
        ]

        context_chunks = self.deduplicate(safe_chunks)
        context_text = self._build_context(context_chunks)
        source = await self.conv_service.build_sources(context_chunks)
        llm_messages = self._build_messages(message, context_text, history)

        try:
            llm_result = await self.llm.complete(
                messages=llm_messages,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
        except Exception:
            logger.exception(
                "LLM call failed for conversation %s (user message already saved)",
                conversation.id,
            )
            
            raise       # caller sees 500; user question is on disk

        assistant_msg = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=llm_result.response.content,
            source=source,
            tokens_used=llm_result.response.total_tokens,
            model_used=f"{llm_result.provider_used}/{llm_result.model_used}",
        )
        self.db.add(assistant_msg)
        await self.db.commit()

        return {
            "conversation_id": conversation.id,
            "conversation_title": conversation.title,
            "message": assistant_msg,
        }
