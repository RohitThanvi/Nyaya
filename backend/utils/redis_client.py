"""Redis async client singleton."""
import asyncio
import logging
from typing import Optional
import redis.asyncio as aioredis
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
_redis: Optional[aioredis.Redis] = None
_redis_lock = asyncio.Lock()


async def get_redis_client() -> aioredis.Redis:
    global _redis
    # Fast path: client exists and is healthy
    if _redis is not None:
        try:
            await _redis.ping()
            return _redis
        except Exception:
            logger.warning("Redis ping failed — reconnecting")
            _redis = None

    async with _redis_lock:
        if _redis is None:
            settings = get_settings().redis
            _redis = aioredis.from_url(
                settings.url,
                max_connections=settings.max_connections,
                socket_timeout=settings.socket_timeout,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            logger.info("Redis client created")
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
