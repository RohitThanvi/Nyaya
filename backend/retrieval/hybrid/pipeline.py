"""
Hybrid retrieval pipeline.

Merges BM25 and ANN results using Reciprocal Rank Fusion (RRF).
RRF is robust to score-scale differences between BM25 and cosine similarity.

Formula: RRF(d) = Σ 1 / (k + rank(d))  where k=60 (standard)

Then applies cross-encoder reranking on merged candidates.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from backend.config.settings import get_settings
from backend.models.domain import (
    CourtType, DocumentType, LawCategory,
    QueryUnderstanding, RetrievedChunk
)
from backend.retrieval.bm25.retriever import BM25Retriever
from backend.retrieval.reranker.cross_encoder import Reranker
from backend.retrieval.vector.retriever import VectorRetriever
from backend.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard RRF constant


class HybridRetriever:
    """
    Full hybrid retrieval pipeline:
    1. Parallel BM25 + ANN retrieval
    2. RRF score fusion
    3. Cross-encoder reranking
    4. Context compression

    Weights are configurable and can be tuned against golden set.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        embedding_service: EmbeddingService,
        reranker: Reranker,
    ):
        self._bm25 = bm25_retriever
        self._vector = vector_retriever
        self._embedder = embedding_service
        self._reranker = reranker
        self._settings = get_settings().retrieval

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (RRF_K + rank)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[RetrievedChunk],
        vector_results: List[RetrievedChunk],
        bm25_weight: float,
        vector_weight: float,
    ) -> List[RetrievedChunk]:
        """
        Fuse BM25 and vector results using weighted RRF.

        Uses chunk_id as the deduplication key.
        Chunks appearing in both lists get boosted scores.
        """
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # BM25 contributions
        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk.chunk.chunk_id
            rrf = self._rrf_score(rank) * bm25_weight
            scores[cid] = scores.get(cid, 0.0) + rrf
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        # Vector contributions
        for rank, chunk in enumerate(vector_results, start=1):
            cid = chunk.chunk.chunk_id
            rrf = self._rrf_score(rank) * vector_weight
            scores[cid] = scores.get(cid, 0.0) + rrf
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        # Merge scores into chunks
        fused = []
        for cid, hybrid_score in scores.items():
            c = chunk_map[cid]
            c.hybrid_score = hybrid_score
            c.retrieval_source = "hybrid" if (
                any(r.chunk.chunk_id == cid for r in bm25_results) and
                any(r.chunk.chunk_id == cid for r in vector_results)
            ) else c.retrieval_source
            fused.append(c)

        fused.sort(key=lambda x: x.hybrid_score, reverse=True)
        return fused

    async def retrieve(
        self,
        query: str,
        query_understanding: Optional[QueryUnderstanding] = None,
        top_k_final: int = 10,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """
        Full hybrid retrieval with timing instrumentation.

        Returns (reranked_chunks, latency_breakdown_ms)
        """
        timings: Dict[str, float] = {}
        settings = self._settings

        bw = bm25_weight or settings.bm25_weight
        vw = vector_weight or settings.vector_weight

        # Extract filters from query understanding
        law_filter = None
        court_filter = None
        year_from = None
        year_to = None
        document_type = None
        section_refs = None

        if query_understanding:
            law_filter = query_understanding.law_filter
            court_filter = query_understanding.court_filter
            if query_understanding.year_range:
                year_from = query_understanding.year_range.get("from")
                year_to = query_understanding.year_range.get("to")
            section_refs = query_understanding.section_refs or None

        # Step 1: Embed query (async, cached)
        t0 = time.perf_counter()
        query_vector = await self._embedder.embed_query(query)
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        # Step 2: Parallel BM25 + ANN retrieval
        t1 = time.perf_counter()
        bm25_coro = self._bm25.search(
            query=query,
            top_k=settings.bm25_top_k,
            law_filter=law_filter,
            court_filter=court_filter,
            year_from=year_from,
            year_to=year_to,
            section_refs=section_refs,
        )
        vector_coro = self._vector.search(
            query_vector=query_vector,
            top_k=settings.vector_top_k,
            law_filter=law_filter,
            court_filter=court_filter,
            year_from=year_from,
            year_to=year_to,
            score_threshold=settings.min_score_threshold,
        )

        bm25_results, vector_results = await asyncio.gather(bm25_coro, vector_coro)
        timings["retrieval_ms"] = (time.perf_counter() - t1) * 1000

        logger.debug(
            f"Retrieved: BM25={len(bm25_results)}, Vector={len(vector_results)}"
        )

        # Step 3: RRF fusion
        t2 = time.perf_counter()
        fused = self._reciprocal_rank_fusion(bm25_results, vector_results, bw, vw)
        # Take top hybrid_top_k before reranking
        fused = fused[: settings.hybrid_top_k]
        timings["fusion_ms"] = (time.perf_counter() - t2) * 1000

        if not fused:
            return [], timings

        # Step 4: Cross-encoder reranking
        t3 = time.perf_counter()
        reranked = await self._reranker.rerank(
            query=query,
            candidates=fused,
            top_k=top_k_final,
        )
        timings["rerank_ms"] = (time.perf_counter() - t3) * 1000

        timings["total_ms"] = sum(timings.values())
        logger.info(
            f"Hybrid retrieval complete: {len(reranked)} results, "
            f"{timings['total_ms']:.0f}ms total"
        )
        return reranked, timings

    async def retrieve_multi_query(
        self,
        queries: List[str],
        query_understanding: Optional[QueryUnderstanding] = None,
        top_k_final: int = 10,
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """
        Multi-query retrieval for better recall.
        Runs retrieval for each expanded query, then merges with RRF.
        Used when query_understanding produces multiple expanded queries.
        """
        if not queries:
            return [], {}

        all_results: List[List[RetrievedChunk]] = []
        combined_timings: Dict[str, float] = {}

        for q in queries[:3]:  # cap at 3 expanded queries
            results, timings = await self.retrieve(
                query=q,
                query_understanding=query_understanding,
                top_k_final=settings.hybrid_top_k,
            )
            all_results.append(results)
            for k, v in timings.items():
                combined_timings[k] = combined_timings.get(k, 0) + v

        if len(all_results) == 1:
            return all_results[0][:top_k_final], combined_timings

        # Cross-query RRF fusion
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}
        for result_list in all_results:
            for rank, chunk in enumerate(result_list, start=1):
                cid = chunk.chunk.chunk_id
                scores[cid] = scores.get(cid, 0) + self._rrf_score(rank)
                if cid not in chunk_map:
                    chunk_map[cid] = chunk

        merged = sorted(chunk_map.values(), key=lambda c: scores[c.chunk.chunk_id], reverse=True)
        for c in merged:
            c.hybrid_score = scores[c.chunk.chunk_id]

        # Final rerank on primary query
        final = await self._reranker.rerank(
            query=queries[0],
            candidates=merged[:40],
            top_k=top_k_final,
        )
        return final, combined_timings


def compress_context(chunks: List[RetrievedChunk], max_tokens: int = 3000) -> str:
    """
    Context window compression.
    Selects the most relevant parts of each chunk to fit within token budget.
    Uses a simple character-based approximation (1 token ≈ 4 chars).
    """
    char_budget = max_tokens * 4
    used = 0
    parts = []

    for chunk in chunks:
        content = chunk.chunk.content
        meta = chunk.chunk.metadata

        # Build citation header
        header_parts = []
        if meta.citation:
            header_parts.append(meta.citation)
        if chunk.chunk.section_ref:
            header_parts.append(f"§{chunk.chunk.section_ref}")
        if meta.law:
            header_parts.append(meta.law.value)

        header = f"[{' | '.join(header_parts)}]" if header_parts else "[Source]"
        header_chars = len(header) + 2

        available = char_budget - used - header_chars
        if available <= 100:
            break

        # Truncate content if needed
        if len(content) > available:
            content = content[:available - 3] + "..."

        parts.append(f"{header}\n{content}")
        used += len(header) + len(content) + 2

    return "\n\n".join(parts)
