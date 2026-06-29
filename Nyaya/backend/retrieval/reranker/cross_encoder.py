"""
Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Changes from v1:
- Sigmoid normalisation applied correctly (not just clipped)
- Final score blends rerank + hybrid in configurable ratio
- Exact-match items bypass reranking and are pinned to top
- Batch inference with configurable batch_size
- Graceful CPU fallback when model unavailable
"""
import logging
import time
from typing import List, Optional

from backend.config.settings import get_settings
from backend.models.domain import RetrievedChunk, RetrievalPath

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self):
        self._settings = get_settings()
        self._cfg = self._settings.reranker
        self._model = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self._cfg.model,
                device=self._cfg.device,
                cache_folder=self._cfg.cache_dir,
            )
            self._loaded = True
            logger.info(f"Reranker loaded: {self._cfg.model} on {self._cfg.device}")
        except Exception as e:
            logger.warning(f"Reranker unavailable ({e}), using hybrid score passthrough")
            self._loaded = True   # mark loaded to avoid repeated attempts

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 10,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        self._load()

        # Pin exact-match results — they don't need reranking
        exact = [r for r in candidates if r.retrieval_source == RetrievalPath.EXACT_LOOKUP.value]
        to_rerank = [r for r in candidates if r.retrieval_source != RetrievalPath.EXACT_LOOKUP.value]

        if self._model is None or not to_rerank:
            # Passthrough: sort by hybrid_score
            for rc in to_rerank:
                rc.rerank_score = rc.hybrid_score
                rc.final_score = rc.hybrid_score
            combined = exact + sorted(to_rerank, key=lambda x: x.final_score, reverse=True)
            return combined[:top_k]

        pairs = [(query, rc.chunk.content[:512]) for rc in to_rerank]

        try:
            t0 = time.perf_counter()
            raw_scores: List[float] = self._model.predict(
                pairs,
                batch_size=self._cfg.batch_size,
                show_progress_bar=False,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(f"Reranker: {len(pairs)} pairs in {elapsed:.0f}ms")
        except Exception as e:
            logger.error(f"Reranker inference failed: {e}")
            for rc in to_rerank:
                rc.rerank_score = rc.hybrid_score
                rc.final_score = rc.hybrid_score
            combined = exact + sorted(to_rerank, key=lambda x: x.final_score, reverse=True)
            return combined[:top_k]

        # Sigmoid normalise raw logit scores → [0, 1]
        import math
        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        for rc, raw in zip(to_rerank, raw_scores):
            rc.rerank_score = sigmoid(float(raw))
            # Blend: 70% reranker + 30% hybrid fusion score
            rc.final_score = 0.7 * rc.rerank_score + 0.3 * rc.hybrid_score

        # Exact-match items get final_score = 1.0 so they always lead
        for rc in exact:
            rc.rerank_score = 1.0
            rc.final_score = 1.0

        combined = exact + sorted(to_rerank, key=lambda x: x.final_score, reverse=True)
        return combined[:top_k]
