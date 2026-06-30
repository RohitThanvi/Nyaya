"""
Embedding service v3 — defensive dimension validation, no silent empty vectors.

Root-cause fix for "expected dim: 1024, got 0":
  sentence-transformers>=3.0.0 has no ceiling pin, so different installs can
  resolve different major versions with different default encode() kwargs
  (e.g. truncate_dim, precision). If the model returns malformed/empty output
  for any reason, we now raise immediately with the exact shape we got,
  instead of letting a [[], [], ...] silently reach Qdrant three calls later.
"""
import asyncio
import hashlib
import json
import logging
import threading
from typing import List, Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


def _get_model():
    """Module-level function exported for main.py warmup."""
    return EmbeddingService._load_model_once()


class EmbeddingDimensionError(RuntimeError):
    """Raised when the embedding model returns vectors of unexpected dimension."""
    pass


class EmbeddingService:
    _model = None
    _model_name: Optional[str] = None
    _verified_dim: Optional[int] = None
    _load_lock = threading.Lock()

    def __init__(self):
        self._cfg = get_settings().embedding
        self._expected_dim = get_settings().qdrant.vector_size
        self._redis = None

    @classmethod
    def _load_model_once(cls):
        # Fast path: model already loaded, no lock needed.
        if cls._model is not None:
            return cls._model

        with cls._load_lock:
            # Re-check inside the lock — another thread may have finished
            # loading while we were waiting. Without this lock, concurrent
            # first calls (e.g. parallel document ingestion workers) could
            # each pass the None-check before either finishes loading,
            # double- or triple-loading the model into memory/VRAM.
            cfg = get_settings().embedding
            if cls._model is None or cls._model_name != cfg.model:
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(
                    cfg.model,
                    device=cfg.device,
                    cache_folder=cfg.cache_dir,
                )
                cls._model_name = cfg.model
                cls._verified_dim = None  # force re-verification on reload
                logger.info(f"Embedding model loaded: {cfg.model} on {cfg.device}")

                # Verify the model actually produces non-empty, correctly-sized
                # vectors immediately after loading — fail loud here, not 3 calls later.
                probe = cls._model.encode(
                    ["dimension verification probe"],
                    normalize_embeddings=cfg.normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                probe_dim = probe.shape[-1] if hasattr(probe, "shape") else len(probe[0]) if probe else 0
                cls._verified_dim = probe_dim
                logger.info(f"Embedding model verified: produces {probe_dim}-dim vectors")

                expected = get_settings().qdrant.vector_size
                if probe_dim != expected:
                    raise EmbeddingDimensionError(
                        f"Model '{cfg.model}' produced {probe_dim}-dim vectors but "
                        f"QDRANT_VECTOR_SIZE is configured as {expected}. "
                        f"This usually means: (1) the installed sentence-transformers "
                        f"version changed the model's default output behavior — pin "
                        f"sentence-transformers==3.0.1 in requirements.txt and rebuild, or "
                        f"(2) the model's config_sentence_transformers.json defines a "
                        f"truncate_dim that doesn't match. Got {probe_dim}, expected {expected}."
                    )
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

    def _validate_vectors(self, vectors: List[List[float]], source: str) -> None:
        """Raise immediately if any vector is empty or wrong dimension."""
        if not vectors:
            return
        for i, v in enumerate(vectors):
            if not v or len(v) == 0:
                raise EmbeddingDimensionError(
                    f"{source}: vector at index {i} is EMPTY (dim=0). "
                    f"Model='{self._cfg.model}', expected_dim={self._expected_dim}. "
                    f"Check sentence-transformers version compatibility — "
                    f"see EmbeddingService._load_model_once() probe check."
                )
            if len(v) != self._expected_dim:
                raise EmbeddingDimensionError(
                    f"{source}: vector at index {i} has dim={len(v)}, "
                    f"expected {self._expected_dim}. Model='{self._cfg.model}'."
                )

    async def embed_query(self, text: str) -> List[float]:
        cache_key = self._cache_key(text)
        redis = self._get_redis()

        if redis:
            try:
                cached = await redis.get(cache_key)
                if cached:
                    vec = json.loads(cached)
                    if vec and len(vec) == self._expected_dim:
                        return vec
                    logger.warning(f"Cached embedding has wrong dim, ignoring cache: {cache_key}")
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
                convert_to_numpy=True,
            ).tolist(),
        )

        self._validate_vectors([embedding], "embed_query")

        if redis:
            try:
                ttl = get_settings().redis.ttl_embedding
                await redis.setex(cache_key, ttl, json.dumps(embedding))
            except Exception:
                pass

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Validates dimensions before returning."""
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
                convert_to_numpy=True,
            ).tolist(),
        )
        self._validate_vectors(embeddings, "embed_batch")
        return embeddings

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding for Celery workers."""
        if not texts:
            return []
        model = self._load_model_once()
        embeddings = model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=self._cfg.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
        self._validate_vectors(embeddings, "embed_batch_sync")
        return embeddings
