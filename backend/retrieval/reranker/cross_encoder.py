"""
Cross-encoder reranker v3 — GPU fp16, thread-safe singleton, async inference.

Upgrades for 4× RTX 6000 Ada:
- Model upgraded to MiniLM-L-12-v2 (12 layers vs 6, better accuracy,
  same VRAM footprint at fp16 on 48GB card)
- fp16 inference: 2x throughput, half VRAM
- Thread-safe singleton with double-check lock (matches EmbeddingService)
- Runs predict() in asyncio executor so it doesn't block the event loop
  during heavy batch inference (was a direct synchronous call inside an
  async method — blocked FastAPI's event loop for the full reranker pass)
- top_k raised 10→20: GPU speed makes checking more candidates free
"""
import asyncio
import logging
import math
import threading
import time
from typing import List, Optional

from backend.config.settings import get_settings
from backend.models.domain import RetrievedChunk, RetrievalPath

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    _model = None
    _loaded = False
    _load_lock = threading.Lock()
    _async_load_lock: Optional[asyncio.Lock] = None
    _gpu_semaphore: Optional[asyncio.Semaphore] = None

    def __init__(self):
        self._cfg = get_settings().reranker

    @classmethod
    def _get_gpu_semaphore(cls) -> asyncio.Semaphore:
        if cls._gpu_semaphore is None:
            limit = get_settings().reranker.max_concurrent_gpu_calls
            cls._gpu_semaphore = asyncio.Semaphore(limit)
        return cls._gpu_semaphore

    @classmethod
    async def _load_model_once_async(cls):
        """Async-safe entry point — see EmbeddingService._load_model_once_async
        for the full rationale: offloads the blocking CrossEncoder(...) construction
        (+ warmup pass) to a thread so the event loop isn't frozen for every other
        in-flight request while it happens."""
        if cls._loaded:
            return cls._model
        if cls._async_load_lock is None:
            cls._async_load_lock = asyncio.Lock()
        async with cls._async_load_lock:
            if cls._loaded:
                return cls._model
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, cls._load_model_once)

    @classmethod
    def _load_model_once(cls):
        if cls._loaded:
            return cls._model

        with cls._load_lock:
            if cls._loaded:
                return cls._model
            try:
                import torch
                from sentence_transformers import CrossEncoder

                device = get_settings().reranker.device
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("Reranker: CUDA not available, falling back to CPU")
                    device = "cpu"

                cfg = get_settings().reranker
                model = CrossEncoder(
                    cfg.model,
                    device=device,
                    cache_folder=cfg.cache_dir,
                )

                # fp16 on GPU: 2x throughput, half VRAM for MiniLM-L-12-v2
                if cfg.fp16 and device == "cuda":
                    model.model = model.model.half()
                    logger.info(f"Reranker converted to fp16 on {device}")

                # Warmup pass
                _ = model.predict([("warmup query", "warmup document")],
                                  batch_size=1, show_progress_bar=False)

                cls._model = model
                logger.info(f"Reranker loaded: {cfg.model} on {device}")
            except Exception as e:
                logger.warning(f"Reranker unavailable ({e}) — using hybrid score passthrough")
            finally:
                cls._loaded = True

        return cls._model

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        cfg = get_settings().reranker
        top_k = top_k or cfg.top_k
        model = await self._load_model_once_async()

        # Exact-match results bypass reranking — pin to top
        exact = [r for r in candidates if r.retrieval_source == RetrievalPath.EXACT_LOOKUP.value]
        to_rerank = [r for r in candidates if r.retrieval_source != RetrievalPath.EXACT_LOOKUP.value]

        if model is None or not to_rerank:
            for rc in to_rerank:
                rc.rerank_score = rc.hybrid_score
                rc.final_score = rc.hybrid_score
            for rc in exact:
                rc.rerank_score = 1.0
                rc.final_score = 1.0
            combined = exact + sorted(to_rerank, key=lambda x: x.final_score, reverse=True)
            return combined[:top_k]

        # Truncate to 512 tokens max — CrossEncoder tokenizer handles this
        # internally but explicit truncation avoids model-specific surprises
        pairs = [(query, rc.chunk.content[:512]) for rc in to_rerank]

        try:
            t0 = time.perf_counter()
            # Run in executor: predict() is synchronous and can take 50-200ms
            # on a GPU batch. Blocking the asyncio event loop for that long
            # stalls ALL concurrent requests during the reranker pass.
            loop = asyncio.get_event_loop()
            async with self._get_gpu_semaphore():
                raw_scores: List[float] = await loop.run_in_executor(
                    None,
                    lambda: model.predict(
                        pairs,
                        batch_size=cfg.batch_size,
                        show_progress_bar=False,
                    ),
                )
            logger.debug(
                f"Reranker: {len(pairs)} pairs in "
                f"{(time.perf_counter() - t0) * 1000:.0f}ms "
                f"(batch_size={cfg.batch_size})"
            )
        except Exception as e:
            logger.error(f"Reranker inference failed: {e}")
            for rc in to_rerank:
                rc.rerank_score = rc.hybrid_score
                rc.final_score = rc.hybrid_score
            combined = exact + sorted(to_rerank, key=lambda x: x.final_score, reverse=True)
            return combined[:top_k]

        # Normalise hybrid scores within this batch before blending
        hybrid_vals = [rc.hybrid_score for rc in to_rerank]
        h_lo, h_hi = min(hybrid_vals), max(hybrid_vals)
        h_spread = h_hi - h_lo

        def _norm_h(v: float) -> float:
            return 1.0 if h_spread == 0 else (v - h_lo) / h_spread

        for rc, raw in zip(to_rerank, raw_scores):
            rc.rerank_score = sigmoid(float(raw))
            rc.final_score = 0.7 * rc.rerank_score + 0.3 * _norm_h(rc.hybrid_score)

        for rc in exact:
            rc.rerank_score = 1.0
            rc.final_score = 1.0

        combined = exact + sorted(to_rerank, key=lambda x: x.final_score, reverse=True)
        return combined[:top_k]
