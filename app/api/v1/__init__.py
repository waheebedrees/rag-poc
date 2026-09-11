from fastapi import APIRouter
from app.api.v1 import auth, documents, chat
from app.api.limiter import limiter

router = APIRouter(prefix="/api/v1")
router.include_router(auth.router)
router.include_router(documents.router)
router.include_router(chat.router)
