from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.database import get_db
from app.models import User
from app.services.auth_service import AuthService
from app.security.jwt import decode_token
from app.services.redis_client import TokenBlacklist, get_blacklist


_bearer = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    blacklist: TokenBlacklist = Depends(get_blacklist),
    db: AsyncSession = Depends(get_db),
) -> User:
    token = credentials.credentials

    _401 = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if await blacklist.contains(token):
        raise _401

    try:
        payload = decode_token(token)
    except ValueError:
        raise _401

    if payload.get("type") != "access" or payload.get("sub") is None:
        raise _401

    svc = AuthService(db)
    user = await svc.get_by_id(UUID(payload["sub"]))
    if not user:
        raise _401

    return user


