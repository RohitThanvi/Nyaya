"""
Embedding service using sentence-transformers.
Supports batched encoding, caching, and device-aware inference.
"""
import asyncio
import hashlib
import json
import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

_model_instance = None


def _get_model():
    """Lazy-load the embedding model (singleton)."""
    global _model_instance
    if _model_instance is None:
        import os
        from sentence_transformers import SentenceTransformer
        from backend.config.settings import get_settings
        settings = get_settings()
        emb = settings.embedding
        logger.info(f"Loading embedding model: {emb.model}")

        # Use HF_HOME if set (docker-compose mount), otherwise fall back to cache_dir.
        # Always clear offline flags so the model can download if not cached.
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        hf_home = os.environ.get("HF_HOME")
        cache = os.path.join(hf_home, "hub") if hf_home else emb.cache_dir

        _model_instance = SentenceTransformer(
            emb.model,
            device=emb.device,
            cache_folder=cache,
        )
        logger.info("Embedding model loaded")
    return _model_instance


class EmbeddingService:
    """
    Production embedding service.
    - BGE-large (1024-dim) by default.
    - Redis caching to avoid re-embedding identical text.
    - Async-compatible via executor offload.
    - BGE requires prepending "Represent this sentence: " for queries.
    """

    BGE_QUERY_PREFIX = "Represent this sentence: "
    BGE_PASSAGE_PREFIX = ""  # passages need no prefix for BGE

    def __init__(self, redis_client=None):
        self._redis = redis_client
        from backend.config.settings import get_settings
        self._settings = get_settings().embedding

    def _cache_key(self, text: str, is_query: bool) -> str:
        h = hashlib.sha256(f"{is_query}:{text}".encode()).hexdigest()
        return f"emb:{h}"

    def _encode_sync(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """CPU/GPU encoding — runs in thread pool."""
        model = _get_model()
        if is_query:
            texts = [self.BGE_QUERY_PREFIX + t for t in texts]

        embeddings = model.encode(
            texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=self._settings.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single search query with caching."""
        if self._redis:
            cache_key = self._cache_key(query, is_query=True)
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._encode_sync, [query], True
        )
        result = embeddings[0].tolist()

        if self._redis:
            try:
                await self._redis.setex(
                    cache_key,
                    self._settings.ttl_embedding if hasattr(self._settings, 'ttl_embedding') else 86400,
                    json.dumps(result)
                )
            except Exception:
                pass

        return result

    async def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of passages (no query prefix)."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._encode_sync, texts, False
        )
        return embeddings.tolist()

    async def embed_passages_batched(
        self, texts: List[str], batch_size: int = 64
    ) -> List[List[float]]:
        """
        Large-scale batched embedding for ingestion.
        Yields control between batches to avoid blocking.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            loop = asyncio.get_event_loop()
            batch_embs = await loop.run_in_executor(
                None, self._encode_sync, batch, False
            )
            all_embeddings.extend(batch_embs.tolist())
            if i + batch_size < len(texts):
                await asyncio.sleep(0)  # yield control
        return all_embeddings

    async def initialize(self) -> None:
        """Pre-warm the embedding model. Safe to call multiple times."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_model)
        logger.info("EmbeddingService: model ready")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts as passages (no query prefix). Alias for embed_passages_batched."""
        return await self.embed_passages_batched(texts)
