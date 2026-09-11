

import hashlib
import os
from typing import Optional
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession


from app.config import settings
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.embeddings import get_embedding_service
from app.services.vector_service import get_vector_service
from app.utils.text_extractor import extract_text
from app.utils.text_splitter import TextSplitter
from app.utils.file_utils import safe_filename, upload_path, remove_file
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.splitter = TextSplitter(
            chunk_size_tokens=settings.CHUNK_SIZE_TOKENS,
            overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        )
        self.embedder = get_embedding_service()
        self.vectors = get_vector_service()

    async def upload(
        self,
        file: UploadFile,
        user_id: UUID,
        title: Optional[str] = None,
    ) -> Document:
        content = await file.read()

        if len(content) > settings.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File exceeds {settings.MAX_FILE_SIZE_BYTES // 1024 // 1024}MB limit"
            )

        ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(f"Extension '.{ext}' is not supported")

        content_hash = hashlib.sha256(content).hexdigest()
        # Duplicate check
        dup = await self.db.execute(
            select(Document).where(
                Document.owner_id == user_id,
                Document.content_hash == content_hash,
                Document.status == DocumentStatus.COMPLETED,
            )
        )
        existing = dup.scalar_one_or_none()
        if existing:
            raise ValueError(
                f"Already uploaded as '{existing.title}' (id={existing.id})"
            )
            
        document = Document(
            owner_id=user_id,
            title=title or (file.filename or "Untitled"),
            filename=file.filename or "Untitled",
            content_hash=content_hash,
            file_size=len(content),
        
            file_type=ext,
            status=DocumentStatus.PENDING,
        )
        self.db.add(document)
        await self.db.commit() 
        await self.db.refresh(document)

        # Save file to disk for background processing
        disk_name = safe_filename(file.filename or "upload", str(user_id))
        disk_path = upload_path(disk_name)
        with open(disk_path, "wb") as f:
            f.write(content)
        return document, disk_path

    async def process(self, document_id: UUID, user_id: UUID, disk_path: str, file_type: str):

        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            try:
                                
                doc = await session.get(Document, document_id)
                if doc is None:
                    logger.error("process: document %s not found", document_id)
                    return
                if doc.status not in (DocumentStatus.PENDING, DocumentStatus.FAILED):
                    logger.warning("process: document %s already %s", document_id, doc.status)
                    return
                if doc.status == DocumentStatus.COMPLETED:
                    logger.info("document already processed")
                    return 
                
                doc.status = DocumentStatus.PROCESSING
                await session.commit()
                with open(disk_path, "rb") as f:
                    file_bytes = f.read()

                extracted = await extract_text(file_bytes, file_type)
                text = extracted["text"]
                meta = extracted["metadata"]

                if not text.strip():
                    raise ValueError("No text content could be extracted")

                chunks = self.splitter.split(text, {"document_id": str(document_id)})
                if not chunks:
                    raise ValueError("Document produced no chunks")

                # Generate embeddings
                embeddings = await self.embedder.embed_batch([c.content for c in chunks])

                # Save chunks to Postgres
                db_chunks = []
                for chunk in chunks:
                    db = DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk.index,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        metadata_=chunk.metadata,
                    )
                    session.add(db)
                    db_chunks.append(db)
                await session.flush()

                # Upsert to Qdrant
                chunk_dicts = [
                    {
                        "id": str(dc.id),
                        "chunk_index": dc.chunk_index,
                        "content": dc.content,
                        "token_count": dc.token_count,
                        "metadata": dc.metadata_,
                    }
                    for dc in db_chunks
                ]
                vector_ids = await self.vectors.upsert_chunks(
                    chunks=chunk_dicts,
                    embeddings=embeddings,
                    document_id=str(document_id),
                    user_id=str(user_id),
                )
                for dc, vid in zip(db_chunks, vector_ids):
                    dc.vector_id = vid

                await session.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(
                        status=DocumentStatus.COMPLETED,
                        chunk_count=len(chunks),
                        page_count=meta.get("page_count", 1),
                        word_count=meta.get("word_count", 0),
                        metadata_=meta,
                    )
                )
                await session.commit()
                logger.info("Document %s processed: %d chunks",document_id, len(chunks))
                
            except Exception as e:
                logger.error("Processing failed for %s: %s",
                             document_id, e, exc_info=True)
                await session.rollback()
                try:
                    await session.execute(
                        update(Document)
                        .where(Document.id == document_id)
                        .values(
                            status=DocumentStatus.FAILED,
                            error_message=str(e)[:2000],
                        )
                    )
                    await session.commit()
                except Exception:
                    logger.error(
                        "Could not even mark document as failed", exc_info=True)
            finally:
                remove_file(disk_path)

    async def get(self, document_id: UUID, user_id: UUID) -> Optional[Document]:
        result = await self.db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.owner_id == user_id,
            )
        )
        
        return result.scalar_one_or_none()

    async def list_for_user(
        self, user_id: UUID, page: int = 1, page_size: int = 20
    ) -> tuple[list[Document], int]:
        from sqlalchemy import func

        count_q = await self.db.execute(
            select(func.count(Document.id)).where(Document.owner_id == user_id)
        )
        total = count_q.scalar() or 0

        result = await self.db.execute(
            select(Document)
            .where(Document.owner_id == user_id)
            .order_by(Document.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    async def delete(self, document_id: UUID, user_id: UUID) -> bool:
        doc = await self.get(document_id, user_id)
        if not doc:
            return False
        await self.vectors.delete_by_document(str(document_id))
        await self.db.delete(doc)
        await self.db.commit()
        return True


    async def process2(self, document_id: UUID, disk_path: str, file_type: str, user_id: UUID):
        from app.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            try:
                doc = await session.scalar(
                    select(Document).where(
                        Document.id == document_id,
                        Document.owner_id == user_id,
                    )
                )
                if doc is None:
                    logger.error("process: document %s not found for user %s",
                                document_id, user_id)
                    return
                if doc.status not in (DocumentStatus.PENDING, DocumentStatus.FAILED):
                    logger.warning("process: %s already %s",
                                document_id, doc.status)
                    return

                doc.status = DocumentStatus.PROCESSING
                await session.commit()

                with open(disk_path, "rb") as f:
                    file_bytes = f.read()

                extracted = await extract_text(file_bytes, file_type)
                text = extracted["text"]
                meta = extracted["metadata"]

                if not text.strip():
                    raise ValueError("No text content could be extracted")

                chunks = self.splitter.split(
                    text, {"document_id": str(document_id)})
                if not chunks:
                    raise ValueError("Document produced no chunks")

                embeddings = await self.embedder.embed_batch([c.content for c in chunks])

                db_chunks = [
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=c.index,
                        content=c.content,
                        token_count=c.token_count,
                        metadata_=c.metadata,
                    )
                    for c in chunks
                ]
                session.add_all(db_chunks)
                await session.flush()

                chunk_dicts = [
                    {
                        "id": str(dc.id),
                        "chunk_index": dc.chunk_index,
                        "content": dc.content,
                        "token_count": dc.token_count,
                        "metadata": dc.metadata_,
                    }
                    for dc in db_chunks
                ]
                vector_ids = await self.vectors.upsert_chunks(
                    chunks=chunk_dicts,
                    embeddings=embeddings,
                    document_id=str(document_id),
                    user_id=str(user_id),
                )
                for dc, vid in zip(db_chunks, vector_ids):
                    dc.vector_id = vid

                doc.status = DocumentStatus.COMPLETED
                doc.chunk_count = len(chunks)
                doc.page_count = meta.get("page_count", 1)
                doc.word_count = meta.get("word_count", 0)
                doc.metadata_ = meta
                await session.commit()

                logger.info("Document %s processed: %d chunks",
                            document_id, len(chunks))

            except Exception as e:
                logger.error("Processing failed for %s: %s",
                            document_id, e, exc_info=True)
                await session.rollback()
                try:
                    await session.execute(
                        update(Document)
                        .where(Document.id == document_id)
                        .values(
                            status=DocumentStatus.FAILED,
                            error_message=str(e)[:2000],
                        )
                    )
                    await session.commit()
                except Exception:
                    logger.error(
                        "Could not even mark document as failed", exc_info=True)
            finally:
                remove_file(disk_path)
