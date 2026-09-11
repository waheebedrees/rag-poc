
import os
os.environ.setdefault("SECRET_KEY", "test-secret-key-at-least-32-chars-long")
os.environ.setdefault("ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "7")

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://test:test@localhost:5432/test_knowledge_db",
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("QDRANT_COLLECTION_NAME", "rag_collection")
os.environ.setdefault("VECTOR_DIM", "384")
os.environ.setdefault(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-fake")
os.environ.setdefault("GOOGLE_API_KEY", "fake")

os.environ.setdefault("DEFAULT_LLM_PROVIDER", "openai")
os.environ.setdefault("DEFAULT_LLM_MODEL", "gpt-4o-mini")
os.environ.setdefault("ALLOWED_ORIGINS", '["http://localhost:3000"]')


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.pool import NullPool

from app.main import app
from app.database import Base, get_db
from app.config import settings
from app.services.redis_client import TokenBlacklist, get_blacklist



@pytest.fixture(autouse=True, scope="session")
def _assert_test_settings():
    assert settings.SECRET_KEY.startswith("test-"), (
        f"Tests are using a non-test SECRET_KEY: {settings.SECRET_KEY[:20]}... "
        "Check that env vars in conftest.py run before `from app...` imports."
    )
    assert "test_knowledge_db" in settings.DATABASE_URL, (
        f"Tests are pointing at the wrong DB: {settings.DATABASE_URL}"
    )


@pytest.fixture
def mock_blacklist():
    """In-memory TokenBlacklist stand-in with real add/contains semantics."""
    class _FakeBlacklist:
        def __init__(self):
            self._store: set[str] = set()

        async def add(self, token: str, ttl: int) -> None:
            if ttl > 0:
                self._store.add(token)

        async def contains(self, token: str) -> bool:
            return token in self._store

    return _FakeBlacklist()


@pytest.fixture(autouse=True, scope="session")
def _disable_rate_limits():
    from app.api.limiter import limiter
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture
def mock_llm(monkeypatch):
    """
    Prevent tests from making real LLM calls.
    Replaces LLMService.complete with a fake that returns a canned response.
    """
    from unittest.mock import MagicMock
    from app.services.llm.llm_service import LLMService

    async def fake_complete(self, messages, **kwargs):
        result = MagicMock()
        result.response.content = "[mocked LLM response]"
        result.response.was_truncated = False
        result.response.was_filtered = False
        result.provider_used = "mock"
        result.model_used = "mock-model"
        result.was_fallback = False
        return result

    monkeypatch.setattr(LLMService, "complete", fake_complete)

@pytest_asyncio.fixture(scope="session")
async def engine():
    eng = create_async_engine(
        settings.DATABASE_URL,       
        poolclass=NullPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def db(engine):
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db, mock_blacklist):
    async def _override_db():
        yield db

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_blacklist] = lambda: mock_blacklist

    transport = ASGITransport(
        app=app, raise_app_exceptions=False)  
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_blacklist, None)
        
@pytest_asyncio.fixture
async def authed(client):
    """Register + login, return client with Authorization header set."""
    await client.post("/api/v1/auth/register", json={
        "email": "test@test.com",
        "username": "testuser",
        "password": "TestPass1",
    })
    resp = await client.post("/api/v1/auth/login", json={
        "email": "test@test.com",
        "password": "TestPass1",
    })
    token = resp.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client
