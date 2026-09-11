from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, Request, status
import time

from app.database import get_db
from app.api.deps import get_current_user, _bearer
from app.security.jwt import decode_token
from app.services.redis_client import get_blacklist, TokenBlacklist
from app.api.limiter import limiter
from app.models import User
from app.schemas import UserCreate, UserLogin, UserResponse, TokenResponse, TokenRefresh
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post(
    '/register',
    response_model=UserResponse,
    status_code=201
)
@limiter.limit('5/minute')
async def register(
    request: Request,
    body: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    svc = AuthService(db)
    try:
        user = await svc.register(body)
        return user
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(
    request: Request,
    body: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    svc = AuthService(db)
    try:
        return await svc.authenticate(body.email, body.password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))



@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: TokenRefresh,
    db: AsyncSession = Depends(get_db),
):
    svc = AuthService(db)
    try:
        return await svc.refresh(body.refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))



@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    blacklist: TokenBlacklist = Depends(get_blacklist),
):
    try:
        payload = decode_token(credentials.credentials)
    except ValueError:
        return None
    ttl = max(0, int(payload.get("exp", 0) - time.time()))
    if ttl > 0:
        await blacklist.add(credentials.credentials, ttl)
    return None

@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user
