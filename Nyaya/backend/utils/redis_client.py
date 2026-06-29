"""Redis async client singleton."""
import logging
from typing import Optional
import redis.asyncio as aioredis
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
_redis: Optional[aioredis.Redis] = None


async def get_redis_client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        settings = get_settings().redis
        _redis = aioredis.from_url(
            settings.url,
            max_connections=settings.max_connections,
            socket_timeout=settings.socket_timeout,
            decode_responses=True,
        )
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.close()
        _redis = None
