"""
Cross-encoder reranker using ms-marco-MiniLM.
Takes top-k candidates from hybrid retrieval and re-scores them
with a full cross-attention model (query + passage together).
"""
import asyncio
import logging
from typing import List, Tuple

import numpy as np

from backend.config.settings import get_settings
from backend.models.domain import RetrievedChunk

logger = logging.getLogger(__name__)

_reranker_instance = None


def _get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        import os
        from sentence_transformers import CrossEncoder
        settings = get_settings().reranker
        logger.info(f"Loading reranker: {settings.model}")
        # CrossEncoder doesn't accept cache_folder — set HF_HOME instead
        if settings.cache_dir:
            os.environ.setdefault("HF_HOME", settings.cache_dir)
        _reranker_instance = CrossEncoder(
            settings.model,
            device=settings.device,
            max_length=512,
        )
        logger.info("Reranker loaded")
    return _reranker_instance


class Reranker:
    """
    Cross-encoder reranker.

    Why cross-encoder over bi-encoder for reranking:
    - Cross-encoders process query+passage jointly → better relevance signal
    - Slower than bi-encoder but only runs on top-k candidates (~20-40)
    - ms-marco-MiniLM-L-6-v2: 6-layer model, good speed/accuracy tradeoff

    Output: re-sorted list with rerank_score populated.
    """

    def __init__(self):
        self._settings = get_settings().reranker

    def _rerank_sync(
        self, query: str, passages: List[str]
    ) -> List[float]:
        """CPU/GPU scoring — runs in thread pool."""
        reranker = _get_reranker()
        pairs = [[query, p] for p in passages]
        scores = reranker.predict(
            pairs,
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
        )
        # Normalize to [0, 1] using sigmoid
        scores = 1 / (1 + np.exp(-scores))
        return scores.tolist()

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int | None = None,
    ) -> List[RetrievedChunk]:
        """
        Rerank candidates using cross-encoder scores.
        Returns top_k results sorted by rerank_score descending.
        """
        if not candidates:
            return []

        k = top_k or self._settings.top_k

        passages = [c.chunk.content for c in candidates]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, self._rerank_sync, query, passages
        )

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)
            # Final score blends hybrid score and rerank score
            chunk.final_score = 0.3 * chunk.hybrid_score + 0.7 * chunk.rerank_score

        reranked = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        result = reranked[:k]

        logger.debug(
            f"Reranked {len(candidates)} → top {len(result)}. "
            f"Top score: {result[0].final_score:.4f}" if result else "No results"
        )
        return result
