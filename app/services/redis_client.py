from app.config import settings

import redis.asyncio as aioredis

redis_client = aioredis.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True,
)


class TokenBlacklist:
    def __init__(self, client):
        self.redis = client
        self.prefix = "blacklist:"

    async def add(self, token: str, expire_in_seconds: int):
        if expire_in_seconds > 0:
            await self.redis.setex(
                f"{self.prefix}{token}", expire_in_seconds, "1"
            )

    async def contains(self, token: str) -> bool:
        return bool(await self.redis.exists(f"{self.prefix}{token}"))


_blacklist: TokenBlacklist | None = None


def get_blacklist() -> TokenBlacklist:     
    """FastAPI dependency – returns the singleton blacklist."""
    if _blacklist is None:
        raise RuntimeError("Blacklist not initialized")
    return _blacklist


def init_blacklist(redis: aioredis.Redis) -> None:
    global _blacklist
    _blacklist = TokenBlacklist(redis)
