

from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import String, Column, DateTime, Boolean, Enum as SQLEnum, ForeignKey, Integer, Text, Table, Float, Index, text 
from sqlalchemy.orm import relationship
import uuid
from  datetime import datetime, timezone
from enum import Enum

from app.database import Base

def utc_now():
    return datetime.now(timezone.utc)


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable= False, index=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password =  Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime(timezone=True),
                        default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True),
                        default=utc_now, onupdate=utc_now, nullable=False)
    
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"
    

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    title = Column(String(500),  nullable=False)
    filename = Column(String(500),  nullable=False)
    file_path = Column(String(1000),  nullable=True)
    file_size = Column(Integer,  nullable=False)
    file_type = Column(String(50),  nullable=False)
    content_hash = Column(String(64), nullable=True, index=True)  # SHA-256

    status  = Column(
        SQLEnum(DocumentStatus),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True
    )
    
    chunk_count = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    metadata_ = Column(JSONB, default=dict)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True),
                        default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now,
                        onupdate=utc_now, nullable=False)



    __table_args__ = (
        Index("ix_documents_owner_hash", "owner_id", "content_hash"),
        Index("ix_documents_owner_created", "owner_id", "created_at"),
        Index("ix_documents_owner_status", "owner_id", "status"),
    )

    owner = relationship("User", back_populates='documents')
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document {self.title}>"


class DocumentChunk(Base):
    __tablename__ = 'document_chunk'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)

    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=True)
    token_count = Column(Integer, default=0)

    vector_id = Column(String(255), nullable=True) 
    metadata_ = Column(JSONB, default=dict)

    created_at = Column(DateTime(timezone=True),
                        default=utc_now, nullable=False)
    document = relationship("Document", back_populates="chunks")


    __table_args__ = (
        Index(
            "ix_document_chunk_content_fts",
            text("to_tsvector('english', content)"),
            postgresql_using="gin",
        ),
    )


conversation_documents = Table(
    'conversation_documents',
    Base.metadata,
    Column('conversation_id', UUID(as_uuid=True),
           ForeignKey('conversations.id', ondelete='CASCADE'), primary_key=True),
    Column('document_id', UUID(as_uuid=True),
           ForeignKey('documents.id', ondelete='CASCADE'), primary_key=True),
)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(500), nullable=True)
    
    created_at = Column(DateTime(timezone=True),
                        default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now,
                        onupdate=utc_now, nullable=False)

    document_ids = relationship('Document', secondary=conversation_documents)

    user = relationship('User', back_populates='conversations')
    messages = relationship(
        'Message',
        back_populates='conversation',
        cascade="all, delete-orphan"
        
    )
    
    
class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)


    role = Column(SQLEnum(MessageRole), nullable=False )
    content = Column(Text, nullable=False)
    source = Column(JSONB, default=[])
    model_used = Column(String(100), nullable=True)
    tokens_used = Column(Integer, default=0)
    confidence_score = Column(Float, nullable=True)
    metadata_ = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    conversation = relationship('Conversation', back_populates='messages')


    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )
