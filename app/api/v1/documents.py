from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form,
    HTTPException, Query, Request, UploadFile, status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from uuid import UUID

from app.database import get_db
from app.api.deps import get_current_user
from app.models import User
from app.schemas import DocumentResponse, DocumentListResponse
from app.services.document_service import DocumentService
from app.api.limiter import limiter

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=202)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a document for processing.

    Processing runs in the background:
    text extraction → chunking → embedding → vector store.

    Poll `GET /documents/{id}` to check status.
    """
    svc = DocumentService(db)
    try:
        document, disk_path = await svc.upload(file, current_user.id, title)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    
    background_tasks.add_task(
        svc.process,
        document_id=document.id,
        user_id=current_user.id,  
        disk_path=disk_path,
        file_type=document.file_type,

    )

    return document


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        svc = DocumentService(db)
        docs, total = await svc.list_for_user(current_user.id, page, page_size)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    
    return DocumentListResponse(
        documents=docs, total=total, page=page, page_size=page_size
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    svc = DocumentService(db)
    doc = await svc.get(document_id, current_user.id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):  
    svc = DocumentService(db)
    if not await svc.delete(document_id, current_user.id):
        raise HTTPException(status_code=404, detail="Document not found")
