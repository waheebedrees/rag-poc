from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from uuid import UUID

from app.database import get_db
from app.api.deps import get_current_user
from app.models import User
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationListItem,
    ConversationDetailResponse,
)
from app.services.chat_service import ChatService
from app.api.limiter import limiter

router = APIRouter(prefix="/chat", tags=["Chat"])

from app.services.chat_service import ChatService, ChatError

@router.post("/", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Send a message and get an AI response grounded in your documents.

    - Provide `conversation_id` to continue an existing conversation
    - Provide `document_ids` to limit retrieval to specific documents
    """
    svc = ChatService(db)
    try:
        result = await svc.chat(
            user_id=current_user.id,
            message=body.message,
            conversation_id=body.conversation_id,
            document_ids=body.document_ids,
            temperature=body.temperature,
            model=body.model,
            provider=body.provider,
        )
    except ChatError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ChatResponse(
        conversation_id=result["conversation_id"],
        message=result["message"],
        conversation_title=result.get("conversation_title"),
    )


@router.get("/conversations", response_model=list[ConversationListItem])
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = ChatService(db)
    return await svc.conv_service.list_conversations(current_user.id, page, page_size)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = ChatService(db)
    conv = await svc.conv_service.get_conversation(conversation_id, current_user.id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        messages=conv.messages,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = ChatService(db)
    if not await svc.conv_service.delete_conversation(conversation_id, current_user.id):
        raise HTTPException(status_code=404, detail="Conversation not found")
