"""
Hybrid retrieval pipeline — three-path architecture.

Path 1: Exact lookup   — always runs when section_refs detected
Path 2: BM25 (lexical) — always runs
Path 3: ANN (vector)   — conditional: only fires when:
            a) exact lookup returned 0 results AND
            b) no section numbers in query AND
            c) query word count > VECTOR_MIN_QUERY_WORDS

Fusion: Reciprocal Rank Fusion (k=60) with path-weighted scores.
        Exact-match items receive a fixed top-rank boost before fusion.

Context compression: semantic-aware, not naive truncation.
                     Preserves structural chunk types (FACTS, RATIO, FINAL_ORDER)
                     before filling remaining budget with high-score PASSAGE chunks.
"""
import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from backend.config.settings import get_settings
from backend.models.domain import (
    ChunkType, CourtType, DocumentType, LawCategory,
    QueryUnderstanding, RetrievedChunk, RetrievalPath,
)
from backend.retrieval.bm25.retriever import BM25Retriever
from backend.retrieval.reranker.cross_encoder import Reranker
from backend.retrieval.vector.retriever import VectorRetriever
from backend.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

RRF_K = 60


# ─────────────────────────────────────────────────────────────────────────────
# RRF helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rrf(rank: int, weight: float = 1.0) -> float:
    return weight / (RRF_K + rank)


def _reciprocal_rank_fusion(
    result_lists: List[Tuple[List[RetrievedChunk], float]],
) -> List[RetrievedChunk]:
    """
    Weighted RRF over N result lists.
    result_lists: [(chunks, weight), ...]
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedChunk] = {}

    for chunks, weight in result_lists:
        for rank, rc in enumerate(chunks, start=1):
            cid = rc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + _rrf(rank, weight)
            if cid not in chunk_map:
                chunk_map[cid] = rc

    fused = []
    for cid, score in scores.items():
        rc = chunk_map[cid]
        rc.hybrid_score = score
        fused.append(rc)

    fused.sort(key=lambda x: x.hybrid_score, reverse=True)
    return fused


def _query_word_count(query: str) -> int:
    return len(re.findall(r"\b\w+\b", query))


def _has_section_numbers(query: str) -> bool:
    return bool(re.search(
        r"\b(?:section|sec\.?|s\.)\s*\d+|\bBNS\s+\d+|\bBNSS\s+\d+|\bBSA\s+\d+|\bIPC\s+\d+",
        query, re.IGNORECASE
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Context Compression
# ─────────────────────────────────────────────────────────────────────────────

# Structural chunk types that carry the highest information density for
# summarisation and question answering — preserved before filler passages.
_PRIORITY_TYPES = {
    ChunkType.RATIO,
    ChunkType.FINAL_ORDER,
    ChunkType.FINDINGS,
    ChunkType.FACTS,
    ChunkType.ISSUES,
    ChunkType.SECTION,
    ChunkType.PUNISHMENT,
}


def compress_context(chunks: List[RetrievedChunk], max_tokens: int = 3000) -> str:
    """
    Semantic-aware context compression.

    Priority ordering:
      1. Structural chunks (RATIO, FINAL_ORDER, FINDINGS, FACTS, ISSUES, SECTION, PUNISHMENT)
         — sorted by final_score
      2. Remaining PASSAGE / other chunks — sorted by final_score

    Each chunk gets a citation header: [citation | §section | law | page N]
    Token budget: 1 token ≈ 4 chars (conservative estimate for legal text).
    Never truncates mid-sentence — breaks at last full stop within budget.
    """
    char_budget = max_tokens * 4
    used = 0
    parts = []

    priority = sorted(
        [rc for rc in chunks if rc.chunk.chunk_type in _PRIORITY_TYPES],
        key=lambda x: x.final_score, reverse=True,
    )
    remainder = sorted(
        [rc for rc in chunks if rc.chunk.chunk_type not in _PRIORITY_TYPES],
        key=lambda x: x.final_score, reverse=True,
    )

    for rc in priority + remainder:
        chunk = rc.chunk
        meta = chunk.metadata

        header_parts = []
        if meta.citation:
            header_parts.append(meta.citation)
        if chunk.section_ref:
            header_parts.append(f"§{chunk.section_ref}")
        if meta.law:
            header_parts.append(meta.law.value)
        if chunk.page_number:
            header_parts.append(f"p.{chunk.page_number}")
        if meta.court_name:
            header_parts.append(meta.court_name)

        header = f"[{' | '.join(header_parts)}]" if header_parts else "[Source]"
        header_len = len(header) + 1   # +1 for newline

        available = char_budget - used - header_len
        if available <= 80:
            break

        content = chunk.content
        if len(content) > available:
            # Truncate at sentence boundary
            truncated = content[:available]
            last_stop = max(
                truncated.rfind(". "),
                truncated.rfind(".\n"),
            )
            content = (truncated[:last_stop + 1] if last_stop > 0 else truncated) + " [...]"

        parts.append(f"{header}\n{content}")
        used += header_len + len(content) + 2   # +2 for double newline separator

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# HybridRetriever
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Full three-path retrieval pipeline.
    Vector path is conditional — only fires for semantic queries at scale.
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

    def _should_use_vector(
        self, query: str, exact_count: int, bm25_count: int,
        qu: Optional[QueryUnderstanding]
    ) -> bool:
        """
        Decide whether to fire the ANN vector path.

        Rule changes vs original:
        - word threshold 6→3: short precise legal queries ('anticipatory bail',
          'preventive detention', 'section 482 quashing') need semantic search
          most — they're underrepresented in BM25 because they appear as noun
          phrases with synonyms across millions of documents. The original
          threshold of 6 words silently skipped ANN for virtually every
          common legal query.
        - Always fire vector when BM25 returned < 3 results regardless of
          query length — thin BM25 results are a signal that lexical matching
          missed something and semantic fallback is needed.
        """
        if exact_count >= 3:
            return False   # strong exact matches — vector adds noise, not signal
        if bm25_count < 3:
            return True    # BM25 missed almost everything — always try vector
        if _has_section_numbers(query) and exact_count > 0:
            return False   # section query with exact hit — lexical is sufficient
        if _query_word_count(query) < self._settings.vector_min_query_words:
            return False
        return True

    async def retrieve(
        self,
        query: str,
        query_understanding: Optional[QueryUnderstanding] = None,
        top_k_final: int = 10,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """
        Full retrieval pipeline with timing.
        Returns (reranked_chunks, latency_breakdown_ms).
        """
        timings: Dict[str, float] = {}
        s = self._settings
        bw = bm25_weight or s.bm25_weight
        vw = vector_weight or s.vector_weight

        qu = query_understanding
        law_filter    = qu.law_filter    if qu else None
        court_filter  = qu.court_filter  if qu else None
        year_from     = qu.year_range.get("from") if qu and qu.year_range else None
        year_to       = qu.year_range.get("to")   if qu and qu.year_range else None
        section_refs  = qu.section_refs  if qu else []

        # ── Path 1 + 2: BM25 (includes exact lookup internally) ──────────
        t0 = time.perf_counter()
        bm25_results = await self._bm25.search(
            query=query,
            top_k=s.bm25_top_k,
            law_filter=law_filter,
            court_filter=court_filter,
            year_from=year_from,
            year_to=year_to,
            section_refs=section_refs or [],
        )
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        exact_count = sum(
            1 for r in bm25_results
            if r.retrieval_source == RetrievalPath.EXACT_LOOKUP.value
        )

        # ── Path 3: Vector (conditional) ─────────────────────────────────
        vector_results: List[RetrievedChunk] = []
        bm25_count = len([r for r in bm25_results
                          if r.retrieval_source != RetrievalPath.EXACT_LOOKUP.value])
        if self._should_use_vector(query, exact_count, bm25_count, qu):
            t0 = time.perf_counter()
            query_vector = await self._embedder.embed_query(query)
            vector_results = await self._vector.search(
                query_vector=query_vector,
                top_k=s.vector_top_k,
                law_filter=law_filter,
                court_filter=court_filter,
                year_from=year_from,
                year_to=year_to,
                score_threshold=s.min_score_threshold,
            )
            timings["vector_ms"] = (time.perf_counter() - t0) * 1000
        else:
            timings["vector_ms"] = 0.0

        logger.debug(
            f"Retrieved: exact={exact_count}, bm25={len(bm25_results)}, "
            f"vector={len(vector_results)}"
        )

        # ── RRF Fusion ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        # Exact results get bm25_weight=2.0 — their top ranks dominate fusion
        exact_results = [r for r in bm25_results if r.retrieval_source == RetrievalPath.EXACT_LOOKUP.value]
        lexical_results = [r for r in bm25_results if r.retrieval_source != RetrievalPath.EXACT_LOOKUP.value]

        result_lists = [(exact_results, 2.0), (lexical_results, bw)]
        if vector_results:
            result_lists.append((vector_results, vw))

        fused = _reciprocal_rank_fusion(result_lists)
        fused = fused[:s.hybrid_top_k]
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

        if not fused:
            return [], timings

        # ── Cross-encoder Reranking ───────────────────────────────────────
        t0 = time.perf_counter()
        reranked = await self._reranker.rerank(
            query=query,
            candidates=fused,
            top_k=top_k_final,
        )
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = sum(timings.values())

        logger.info(
            f"Hybrid retrieval: {len(reranked)} results in {timings['total_ms']:.0f}ms "
            f"(exact={exact_count}, bm25={len(lexical_results)}, vector={len(vector_results)})"
        )
        return reranked, timings

    async def retrieve_multi_query(
        self,
        queries: List[str],
        query_understanding: Optional[QueryUnderstanding] = None,
        top_k_final: int = 10,
        law_filter: Optional[List] = None,
        court_filter: Optional[List] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_type: Optional[Any] = None,
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """
        Multi-query retrieval with parallel execution and filter forwarding.

        Previously dropped law_filter/court_filter/year_from/year_to/document_type —
        _step_retrieve passed them via **retrieve_kwargs but the signature didn't
        accept them, so all search filters were silently ignored on multi-query paths.
        """
        if not queries:
            return [], {}

        capped = queries[:3]

        filter_kwargs = dict(
            query_understanding=query_understanding,
            top_k_final=top_k_final,
            law_filter=law_filter,
            court_filter=court_filter,
            year_from=year_from,
            year_to=year_to,
            document_type=document_type,
        )

        tasks = [
            self.retrieve(query=q, **filter_kwargs)
            for q in capped
        ]
        results_and_timings = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: List[List[RetrievedChunk]] = []
        combined_timings: Dict[str, float] = {}

        for item in results_and_timings:
            if isinstance(item, Exception):
                logger.warning(f"Multi-query sub-retrieve failed: {item}")
                continue
            results, timings = item
            all_results.append(results)
            for k, v in timings.items():
                combined_timings[k] = combined_timings.get(k, 0) + v

        if not all_results:
            return [], combined_timings

        if len(all_results) == 1:
            return all_results[0][:top_k_final], combined_timings

        # Cross-query RRF then rerank
        fused = _reciprocal_rank_fusion([(res, 1.0) for res in all_results])
        final = await self._reranker.rerank(
            query=queries[0],
            candidates=fused[:40],
            top_k=top_k_final,
        )
        return final, combined_timings

    async def retrieve_for_document(
        self,
        document_id: str,
        query: str,
        top_k: int = 15,
    ) -> List[RetrievedChunk]:
        """
        Scoped retrieval within a single document.
        Uses websearch_to_tsquery (not plainto_tsquery — supports phrase/OR/NOT).
        Fetches real document metadata so context headers show correct law/court.
        """
        from sqlalchemy import text as sql_text
        from backend.models.domain import (
            DocumentMetadata, DocumentType, LegalChunk, LawCategory,
        )

        clean = query
        for prefix in ["summarize the key", "summarize", "what is", "explain",
                        "from the case:", "from the document:", "in this case",
                        "who were", "what are the"]:
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):].strip(" :-")
        if len(clean.strip()) < 5:
            clean = query

        db = self._bm25._db

        # Fetch real document metadata for citation headers in context
        doc_row = (await db.execute(sql_text("""
            SELECT document_type, law, court, court_name, citation, year, source_url
            FROM documents WHERE document_id = :doc_id
        """), {"doc_id": document_id})).fetchone()

        if doc_row:
            try:
                doc_type = DocumentType(doc_row.document_type)
            except ValueError:
                doc_type = DocumentType.JUDGMENT
            try:
                law = LawCategory(doc_row.law) if doc_row.law else LawCategory.OTHER
            except ValueError:
                law = LawCategory.OTHER
            meta = DocumentMetadata(
                document_id=document_id,
                document_type=doc_type,
                law=law,
                court_name=doc_row.court_name,
                citation=doc_row.citation,
                year=doc_row.year,
                source_url=doc_row.source_url,
                language="en",
            )
        else:
            meta = DocumentMetadata(
                document_id=document_id,
                document_type=DocumentType.JUDGMENT,
                law=LawCategory.OTHER,
                language="en",
            )

        # websearch_to_tsquery handles phrase queries, OR, NOT
        result = (await db.execute(sql_text("""
            SELECT chunk_id, content, chunk_type, chunk_index,
                   section_ref, page_number,
                   ts_rank_cd(content_tsv,
                       websearch_to_tsquery('english', :q), 32) AS rank
            FROM chunks
            WHERE document_id = :doc_id
              AND content_tsv @@ websearch_to_tsquery('english', :q)
            ORDER BY rank DESC LIMIT :k
        """), {"doc_id": document_id, "q": clean, "k": top_k})).fetchall()

        if not result:
            stop = {"from","this","that","with","were","what","which",
                    "have","been","case","legal","court","suit","file"}
            keywords = [w.strip(".,?!") for w in clean.split()
                        if len(w) > 4 and w.lower() not in stop][:4]
            if keywords:
                like_clause = " OR ".join(f"content ILIKE :kw{i}" for i in range(len(keywords)))
                params: dict = {"doc_id": document_id, "k": top_k}
                for i, kw in enumerate(keywords):
                    params[f"kw{i}"] = f"%{kw}%"
                result = (await db.execute(sql_text(f"""
                    SELECT chunk_id, content, chunk_type, chunk_index,
                           section_ref, page_number, 0.5 AS rank
                    FROM chunks WHERE document_id = :doc_id AND ({like_clause})
                    ORDER BY chunk_index LIMIT :k
                """), params)).fetchall()

        if not result:
            result = (await db.execute(sql_text("""
                SELECT chunk_id, content, chunk_type, chunk_index,
                       section_ref, page_number, 1.0 AS rank
                FROM chunks WHERE document_id = :doc_id AND content_length > 150
                ORDER BY chunk_index LIMIT :k
            """), {"doc_id": document_id, "k": top_k})).fetchall()

        chunks = []
        for row in result:
            try:
                ct = ChunkType(row.chunk_type)
            except ValueError:
                ct = ChunkType.PASSAGE
            lc = LegalChunk(
                chunk_id=str(row.chunk_id),
                document_id=document_id,
                chunk_type=ct,
                content=row.content,
                content_length=len(row.content),
                chunk_index=row.chunk_index,
                page_number=getattr(row, "page_number", None),
                section_ref=getattr(row, "section_ref", None) or "",
                metadata=meta,
            )
            score = float(getattr(row, "rank", 0.5))
            chunks.append(RetrievedChunk(
                chunk=lc,
                bm25_score=score,
                vector_score=score,
                final_score=score,
                retrieval_source=RetrievalPath.DOCUMENT_FTS.value,
                retrieval_method=RetrievalPath.DOCUMENT_FTS.value,
            ))
        return chunks
