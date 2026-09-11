from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from collections.abc import AsyncGenerator

from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)

class Base(DeclarativeBase):
    pass





def resolve_async_url(url: str) -> str:
    parsed = make_url(url)
    driver_map = {
        "postgresql": "postgresql+asyncpg",
        "postgres": "postgresql+asyncpg",
        "postgresql+psycopg2": "postgresql+asyncpg",
    }
    driver = parsed.drivername
    if driver in driver_map:
        return str(parsed.set(drivername=driver_map[driver]))
    if "+asyncpg" in driver:
        return url
    raise ValueError(f"Unsupported database driver: {driver}")

engine = create_async_engine(
    resolve_async_url(settings.DATABASE_URL),
    pool_size= settings.DB_POOL_SIZE,
    max_overflow= settings.DB_MAX_OVERFLOW,
    echo= settings.DEBUG,
    pool_pre_ping = True,
    pool_recycle= 3600,    
)


AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception :
            await session.rollback()
            raise

async def init_db():
    async with engine.begin() as conn:

        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully")


