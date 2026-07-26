"""
Embedding service v4 — multi-GPU, fp16, per-worker GPU pinning.

Architecture for 4× RTX 6000 Ada (48GB each):
- Each Celery embed worker runs with CUDA_VISIBLE_DEVICES=N so it sees
  exactly one GPU as "cuda:0". The EmbeddingService loads the model once
  per process onto that GPU and stays there for the worker's lifetime.
- batch_size=512 keeps each GPU fully saturated (bge-large-en-v1.5 is
  ~1.3GB at fp16, leaving 46GB headroom — 512 × 512-token sequences fit
  comfortably in a single forward pass).
- fp16=True halves VRAM footprint, doubles throughput on Ampere/Ada.
- Combined throughput: ~4000-8000 chunks/sec across all 4 GPUs, meaning
  a corpus of 50M chunks (1M documents × 50 chunks) finishes in ~2-3 hours
  instead of days on CPU.
"""
import asyncio
import hashlib
import json
import logging
import os
import threading
from typing import List, Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


def _get_model():
    """Module-level function exported for main.py warmup."""
    return EmbeddingService._load_model_once()


class EmbeddingDimensionError(RuntimeError):
    pass


class EmbeddingService:
    # Class-level singleton per process. Each Celery worker is a separate
    # process with its own CUDA_VISIBLE_DEVICES, so each worker holds its
    # own model instance on its own GPU — no cross-GPU sharing needed.
    _model = None
    _model_name: Optional[str] = None
    _verified_dim: Optional[int] = None
    _load_lock = threading.Lock()
    _async_load_lock: Optional[asyncio.Lock] = None
    _gpu_semaphore: Optional[asyncio.Semaphore] = None

    def __init__(self):
        self._cfg = get_settings().embedding
        self._expected_dim = get_settings().qdrant.vector_size
        self._redis = None

    @classmethod
    def _get_gpu_semaphore(cls) -> asyncio.Semaphore:
        # Created lazily (not at class-definition time) so it's always
        # bound to whichever event loop is actually running.
        if cls._gpu_semaphore is None:
            limit = get_settings().embedding.max_concurrent_gpu_calls
            cls._gpu_semaphore = asyncio.Semaphore(limit)
        return cls._gpu_semaphore

    @classmethod
    async def _load_model_once_async(cls):
        """
        Async-safe entry point for embed_query/embed_batch. Fast path (model
        already loaded) is a plain attribute check — no lock, no executor,
        negligible cost. Only the first-ever call in a process's lifetime
        pays for the actual load, and it does so in a thread so the event
        loop keeps serving every other in-flight request meanwhile.
        """
        if cls._model is not None:
            return cls._model
        if cls._async_load_lock is None:
            cls._async_load_lock = asyncio.Lock()
        async with cls._async_load_lock:
            if cls._model is not None:
                return cls._model
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, cls._load_model_once)

    @classmethod
    def _load_model_once(cls):
        if cls._model is not None:
            return cls._model

        with cls._load_lock:
            if cls._model is not None:
                return cls._model

            cfg = get_settings().embedding

            # Resolve device: workers set CUDA_VISIBLE_DEVICES=N before
            # starting, so "cuda" always maps to their assigned GPU.
            # Fallback to CPU if CUDA isn't available (dev/test environments).
            import torch
            if cfg.device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "CUDA not available — falling back to CPU. "
                    "Set EMBEDDING_DEVICE=cpu in .env to suppress this warning."
                )
                device = "cpu"
            else:
                device = cfg.device

            if device == "cuda":
                gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
                logger.info(
                    f"Loading embedding model on GPU {gpu_id} "
                    f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})"
                )

            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                cfg.model,
                device=device,
                cache_folder=cfg.cache_dir,
            )

            # fp16: halves VRAM usage, doubles throughput on Ada/Ampere.
            # Only on CUDA — CPU fp16 is slower than fp32.
            if cfg.fp16 and device == "cuda":
                import torch
                model = model.half()
                logger.info("Embedding model converted to fp16")

            # Warm up: single forward pass to JIT-compile CUDA kernels so
            # the first real batch doesn't pay the compilation latency.
            probe = model.encode(
                ["dimension verification probe"],
                normalize_embeddings=cfg.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            probe_dim = probe.shape[-1] if hasattr(probe, "shape") else len(probe[0]) if len(probe) else 0
            expected = get_settings().qdrant.vector_size
            if probe_dim != expected:
                raise EmbeddingDimensionError(
                    f"Model '{cfg.model}' produced {probe_dim}-dim vectors but "
                    f"QDRANT_VECTOR_SIZE={expected}. Pin sentence-transformers==3.0.1 "
                    f"in requirements.txt or check the model's truncate_dim config."
                )

            cls._model = model
            cls._model_name = cfg.model
            cls._verified_dim = probe_dim
            logger.info(f"Embedding model ready: {cfg.model} on {device}, dim={probe_dim}")
            return cls._model

    # ── Redis embedding cache ─────────────────────────────────────────────────

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    get_settings().redis.url, decode_responses=False
                )
            except Exception:
                pass
        return self._redis

    def _cache_key(self, text: str) -> str:
        h = hashlib.md5(text.encode()).hexdigest()
        return f"emb:{self._cfg.model.replace('/', ':')}:{h}"

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_vectors(self, vectors: List[List[float]], source: str) -> None:
        for i, v in enumerate(vectors):
            if not v or len(v) == 0:
                raise EmbeddingDimensionError(
                    f"{source}: vector[{i}] is EMPTY. "
                    f"Model='{self._cfg.model}', expected_dim={self._expected_dim}."
                )
            if len(v) != self._expected_dim:
                raise EmbeddingDimensionError(
                    f"{source}: vector[{i}] has dim={len(v)}, "
                    f"expected {self._expected_dim}."
                )

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query with Redis caching."""
        cache_key = self._cache_key(text)
        redis = self._get_redis()

        if redis:
            try:
                cached = await redis.get(cache_key)
                if cached:
                    vec = json.loads(cached)
                    if vec and len(vec) == self._expected_dim:
                        return vec
            except Exception:
                pass

        model = await self._load_model_once_async()
        loop = asyncio.get_event_loop()
        async with self._get_gpu_semaphore():
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
                await redis.setex(
                    cache_key,
                    get_settings().redis.ttl_embedding,
                    json.dumps(embedding)
                )
            except Exception:
                pass

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Async batch embed — runs GPU inference in a thread executor."""
        if not texts:
            return []
        model = await self._load_model_once_async()
        loop = asyncio.get_event_loop()
        async with self._get_gpu_semaphore():
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
        """Synchronous batch embed for Celery GPU workers.

        Each Celery worker process has CUDA_VISIBLE_DEVICES set to its
        assigned GPU, so this always runs on the right device.
        batch_size=512 keeps the GPU fully saturated for bge-large-en-v1.5
        on a 48GB card.
        """
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
