import re
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Tuple, Dict
from datetime import datetime
from uuid  import UUID


from app.models import DocumentStatus, MessageRole

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=80)   
    password: str = Field(..., min_length=8, max_length=128)
    
    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain an uppercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain a digit")
        return v

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        if len(v) < 2:
            raise ValueError("Username must be at least 2 characters")
        return v.strip()


class UserLogin(BaseModel):
    email: EmailStr
    password: str 
    

class UserResponse(BaseModel):
    id: UUID
    email: str
    username: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token : str 
    refresh_token: str 
    token_type: str = 'bearer'
    expires_in: int
    
    
class TokenRefresh(BaseModel):
    refresh_token : str 
    
    
class DocumentResponse(BaseModel):
    id: UUID
    title: str
    filename: str
    file_size: int
    file_type: str
    status: DocumentStatus
    chunk_count: int
    page_count: int
    word_count: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int



class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id : Optional[UUID] = None
    document_ids: Optional[List[UUID]] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    provider: Optional[str] = None
    model: Optional[str] = None
    stream: bool = False


class CompletionResponse(BaseModel):
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    was_fallback: bool

class SourceReference(BaseModel):
    chunk_id : str 
    document_id: str 
    document_title: str 
    content_preview: str 
    similarity_score: float | None = None  


class MessageResponse(BaseModel):
    id: UUID
    role: MessageRole
    content: str 
    source : List[SourceReference]
    tokens_used: int 
    model_used: Optional[str]
    created_at : datetime
    
    model_config = {"from_attributes": True}
    

class ChatResponse(BaseModel):
    conversation_id: UUID
    message: MessageResponse
    conversation_title: str 
    
    
class ConversationListItem(BaseModel):
    id : UUID
    title: Optional[str]
    message_count: int 
    created_at: datetime 
    updated_at : datetime
    
    model_config = {'from_attributes': True}
    
class ConversationDetailResponse(BaseModel):
    id : UUID
    title: Optional[str]
    messages: List[MessageResponse]
    created_at: datetime
    updated_at: datetime

    model_config = {'from_attributes': True}