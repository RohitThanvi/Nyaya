"""
Embedding service v2 — async embed_batch, sync embed_batch_sync for Celery,
correct _get_model export for main.py warmup, Redis caching.
"""
import asyncio
import hashlib
import json
import logging
from typing import List, Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


def _get_model():
    """
    Module-level function exported for main.py warmup.
    Returns the loaded SentenceTransformer model.
    """
    return EmbeddingService._load_model_once()


class EmbeddingService:
    _model = None
    _model_name: Optional[str] = None

    def __init__(self):
        self._cfg = get_settings().embedding
        self._redis = None

    @classmethod
    def _load_model_once(cls):
        cfg = get_settings().embedding
        if cls._model is None or cls._model_name != cfg.model:
            try:
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(
                    cfg.model,
                    device=cfg.device,
                    cache_folder=cfg.cache_dir,
                )
                cls._model_name = cfg.model
                logger.info(f"Embedding model loaded: {cfg.model} on {cfg.device}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return cls._model

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                cfg = get_settings().redis
                self._redis = aioredis.from_url(cfg.url, decode_responses=False)
            except Exception:
                pass
        return self._redis

    def _cache_key(self, text: str) -> str:
        h = hashlib.md5(text.encode()).hexdigest()
        return f"emb:{self._cfg.model.replace('/', ':')}:{h}"

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query string. Uses Redis cache."""
        cache_key = self._cache_key(text)
        redis = self._get_redis()

        if redis:
            try:
                cached = await redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        model = self._load_model_once()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(
                text,
                normalize_embeddings=self._cfg.normalize,
                show_progress_bar=False,
            ).tolist(),
        )

        if redis:
            try:
                ttl = get_settings().redis.ttl_embedding
                await redis.setex(cache_key, ttl, json.dumps(embedding))
            except Exception:
                pass

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Used by ingestion pipeline (async context)."""
        if not texts:
            return []
        model = self._load_model_once()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=self._cfg.batch_size,
                normalize_embeddings=self._cfg.normalize,
                show_progress_bar=False,
            ).tolist(),
        )
        return embeddings

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding for Celery workers."""
        if not texts:
            return []
        model = self._load_model_once()
        return model.encode(
            texts,
            batch_size=self._cfg.ingest_batch_size,
            normalize_embeddings=self._cfg.normalize,
            show_progress_bar=False,
        ).tolist()
