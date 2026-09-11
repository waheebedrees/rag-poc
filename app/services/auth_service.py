
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from typing import Optional, Dict, Any

from app.config import settings
from app.utils.logger import get_logger
from app.models import User
from app.schemas import UserCreate, TokenResponse
from app.security.jwt import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)


logger = get_logger(__name__)


class AuthService:
    def __init__(self, db:AsyncSession):
        self.db = db 
        
    async def register(self, data: UserCreate) -> User:
        existing = await self.db.execute(
            select(User).where(
                (User.email == data.email) | (User.username == data.username)
            )
        )
        conflict = existing.scalar_one_or_none()
        if conflict:
            if conflict.email == data.email:
                raise ValueError("Email Already registered")
            raise ValueError("Username already taken")
        
        user = User(
            email = data.email,
            username= data.username,
            hashed_password=hash_password(data.password)
        )
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)

        logger.info("User registered: %s", user.email)
        return user 
    
    async def refresh(self, token:str) -> TokenResponse:
        payload = decode_token(token)
        if payload.get('type') != 'refresh':
            raise ValueError("Invalid token type")
        result = await self.db.execute(
            select(User).where(User.id == payload['sub'], User.is_active.is_(True))
        )
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found")
        return self._make_tokens(user)
    
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    def _make_tokens(self, user: User) -> TokenResponse:
        return TokenResponse(
            access_token=create_access_token(str(user.id), user.email),
            refresh_token=create_refresh_token(str(user.id), user.email),
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def authenticate(self, email: str, password: str) -> TokenResponse:
        result = await self.db.execute(
            select(User).where(User.email == email, User.is_active.is_(True))
        )
        user = result.scalar_one_or_none()
        if not user or not verify_password(password, user.hashed_password):
            raise ValueError("Invalid email or password")
        return self._make_tokens(user)
