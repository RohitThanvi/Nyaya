"""
BM25 Retriever — three-tier lexical search.

Tier 1: Exact lookup — section_ref + law primary key match.
         Zero approximation. Used when query contains explicit section numbers.

Tier 2: PostgreSQL tsvector (websearch_to_tsquery, ts_rank_cd).
         Production-ready for corpora up to ~5M chunks.
         ts_rank_cd with normalization option 32 approximates BM25 length norm.

Tier 3: Elasticsearch (optional, enabled via ES_ENABLED=true).
         Used for TB-scale corpora. Falls back to PostgreSQL if ES unavailable.

Legal synonym expansion is applied before both Tier 2 and Tier 3:
    IPC -> BNS equivalents, s. -> section, v. -> versus, etc.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import get_settings
from backend.models.domain import (
    ChunkType, CourtType, DocumentMetadata, DocumentType,
    LawCategory, LegalChunk, RetrievedChunk, RetrievalPath,
)

logger = logging.getLogger(__name__)

# ── Legal synonym map (applied before BM25 query construction) ──────────────
LEGAL_SYNONYMS: Dict[str, List[str]] = {
    r"\bipc\b":            ["BNS", "Indian Penal Code"],
    r"\bcrpc\b":           ["BNSS", "Criminal Procedure Code"],
    r"\b420\b":            ["318"],        # IPC 420 → BNS 318 (cheating)
    r"\b302\b":            ["103"],        # IPC 302 → BNS 103 (murder)
    r"\b307\b":            ["109"],        # IPC 307 → BNS 109 (attempt to murder)
    r"\b376\b":            ["64"],         # IPC 376 → BNS 64 (rape)
    r"\b498a\b":           ["85"],         # IPC 498A → BNS 85 (cruelty by husband)
    r"\b482\s+crpc\b":    ["528 BNSS"],   # CrPC 482 → BNSS 528 (quashing)
    r"\b437\s+crpc\b":    ["480 BNSS"],   # CrPC 437 → BNSS 480 (bail)
    r"\b438\s+crpc\b":    ["482 BNSS"],   # CrPC 438 → BNSS 482 (anticipatory bail)
    r"\bs\.\s*(\d+)":     ["section \\1"],
    r"\bv\.\b":           ["versus"],
}


def _expand_legal_synonyms(query: str) -> str:
    """
    Expand IPC/CrPC references to BNS/BNSS equivalents for better recall.

    IMPORTANT: websearch_to_tsquery treats space-separated terms as AND.
    Appending synonyms as plain words turned "IPC 302" into "IPC AND 302
    AND BNS AND 103", which almost nothing matches. Synonyms must be OR'd
    in using websearch_to_tsquery's explicit "OR" keyword instead.
    """
    extra_terms: List[str] = []
    for pattern, replacements in LEGAL_SYNONYMS.items():
        if re.search(pattern, query, re.IGNORECASE):
            extra_terms.extend(replacements)
    if not extra_terms:
        return query.strip()
    # Original query stays one AND'd group; each synonym is its own OR branch.
    or_chain = " OR ".join([f'"{query}"'] + extra_terms)
    return or_chain.strip()


def _normalize_section(raw: str) -> str:
    """Normalize section reference to bare number+letter: '318(2)(a)' -> '318'."""
    m = re.match(r"(\d+[A-Za-z]?)", raw.strip())
    return m.group(1) if m else raw.strip()


def _normalize_scores(results: List["RetrievedChunk"]) -> None:
    """
    Min-max normalize bm25_score to [0, 1] in place, within a single tier's
    result list. PG's ts_rank_cd lives in a tiny ~0-0.3 range while ES BM25
    scores are unbounded (often 5-20+) — without this, merging tiers and
    sorting on raw bm25_score lets ES silently dominate every query whenever
    it's enabled, regardless of true relevance.
    """
    if not results:
        return
    scores = [r.bm25_score for r in results]
    lo, hi = min(scores), max(scores)
    spread = hi - lo
    for r in results:
        r.bm25_score = 1.0 if spread == 0 else (r.bm25_score - lo) / spread
        r.final_score = r.bm25_score


class BM25Retriever:
    """
    Three-tier lexical retrieval.
    Caller should use retrieve() which orchestrates all tiers.
    Individual tier methods are public for direct use in tests.
    """

    def __init__(self, db: AsyncSession):
        self._db = db
        self._settings = get_settings()
        self._ret = self._settings.retrieval
        self._es_enabled = self._settings.es.enabled
        self._es_client = None   # lazy-init when ES is enabled

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 50,
        law_filter: Optional[List[LawCategory]] = None,
        court_filter: Optional[List[CourtType]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_type: Optional[DocumentType] = None,
        section_refs: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Orchestrates three-tier retrieval and returns merged, deduplicated results.

        Tier 1 (exact) always runs when section_refs are present.
        Tier 2 (PG FTS) always runs.
        Tier 3 (ES) runs when ES_ENABLED=true.
        """
        results: List[RetrievedChunk] = []
        seen_ids: set = set()

        # Tier 1 — exact section lookup
        if section_refs:
            for sec in section_refs:
                tier1 = await self.exact_section_lookup(sec, law_filter)
                for r in tier1:
                    if r.chunk.chunk_id not in seen_ids:
                        seen_ids.add(r.chunk.chunk_id)
                        results.append(r)

        # Tier 2 — PostgreSQL FTS
        pg_results = await self._pg_fts_search(
            query=query,
            top_k=top_k,
            law_filter=law_filter,
            court_filter=court_filter,
            year_from=year_from,
            year_to=year_to,
            document_type=document_type,
        )
        _normalize_scores(pg_results)
        for r in pg_results:
            if r.chunk.chunk_id not in seen_ids:
                seen_ids.add(r.chunk.chunk_id)
                results.append(r)

        # Tier 3 — Elasticsearch (when enabled)
        if self._es_enabled:
            es_results = await self._es_search(
                query=query,
                top_k=top_k,
                law_filter=law_filter,
                court_filter=court_filter,
                year_from=year_from,
                year_to=year_to,
            )
            _normalize_scores(es_results)
            for r in es_results:
                if r.chunk.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk.chunk_id)
                    results.append(r)

        # Sort by BM25 score descending; exact-lookup items have score=1.0
        results.sort(key=lambda x: x.bm25_score, reverse=True)
        return results[:top_k]

    # ──────────────────────────────────────────────────────────────────────
    # Tier 1 — Exact section lookup
    # ──────────────────────────────────────────────────────────────────────

    async def exact_section_lookup(
        self,
        section_number: str,
        law_filter: Optional[List[LawCategory]] = None,
    ) -> List[RetrievedChunk]:
        """
        Direct primary-key-style lookup by section_ref.
        Zero approximation — returns exactly the chunks for that section.
        Used when user explicitly references a section number.
        """
        normalized = _normalize_section(section_number)
        params: Dict[str, Any] = {"section": normalized}
        law_clause = ""
        if law_filter:
            law_clause = "AND d.law = ANY(:laws)"
            params["laws"] = [l.value for l in law_filter]

        sql = text(f"""
            SELECT
                c.chunk_id, c.document_id, c.chunk_type, c.content,
                c.content_length, c.chunk_index, c.page_number,
                c.section_ref, c.subsection_ref,
                d.document_type, d.law, d.court, d.court_name,
                d.case_number, d.citation, d.year, d.date_decided,
                d.bench, d.parties, d.topic, d.keywords,
                d.source_url, d.is_landmark, d.language
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE
                (c.section_ref = :section
                 OR c.section_ref ILIKE :section_like
                 OR c.subsection_ref ILIKE :section_like)
                {law_clause}
            ORDER BY c.chunk_index
            LIMIT 10
        """)
        params["section_like"] = f"{normalized}%"

        try:
            result = await self._db.execute(sql, params)
            rows = result.fetchall()
        except Exception as e:
            logger.error(f"Exact section lookup failed: {e}")
            return []

        retrieved = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                bm25_score=1.0,        # perfect match — maximum score
                retrieval_source=RetrievalPath.EXACT_LOOKUP.value,
                retrieval_method=RetrievalPath.EXACT_LOOKUP.value,
                final_score=1.0,
            ))
        logger.debug(f"Exact lookup '{section_number}' → {len(retrieved)} chunks")
        return retrieved

    # ──────────────────────────────────────────────────────────────────────
    # Tier 2 — PostgreSQL FTS
    # ──────────────────────────────────────────────────────────────────────

    async def _pg_fts_search(
        self,
        query: str,
        top_k: int,
        law_filter: Optional[List[LawCategory]],
        court_filter: Optional[List[CourtType]],
        year_from: Optional[int],
        year_to: Optional[int],
        document_type: Optional[DocumentType],
    ) -> List[RetrievedChunk]:
        """
        PostgreSQL tsvector search with legal synonym expansion.
        ts_rank_cd normalization=32: divides by document length — approximates BM25.
        websearch_to_tsquery handles phrase queries, negation, and OR naturally.
        """
        expanded = _expand_legal_synonyms(query)
        clean = re.sub(r"[&|!<>():]", " ", expanded)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            return []

        where, params = self._build_pg_filter(
            law_filter, court_filter, year_from, year_to, document_type
        )
        params["query"] = clean
        params["top_k"] = top_k

        sql = text(f"""
            WITH ranked AS (
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.chunk_type,
                    c.content,
                    c.content_length,
                    c.chunk_index,
                    c.page_number,
                    c.section_ref,
                    c.subsection_ref,
                    d.document_type, d.law, d.court, d.court_name,
                    d.case_number, d.citation, d.year, d.date_decided,
                    d.bench, d.parties, d.topic, d.keywords,
                    d.source_url, d.is_landmark, d.language,
                    ts_rank_cd(
                        c.content_tsv,
                        websearch_to_tsquery('english', :query),
                        32
                    ) AS bm25_score
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE
                    {where}
                    AND c.content_tsv @@ websearch_to_tsquery('english', :query)
                ORDER BY bm25_score DESC
                LIMIT :top_k
            )
            SELECT * FROM ranked WHERE bm25_score > 0
        """)

        try:
            result = await self._db.execute(sql, params)
            rows = result.fetchall()
        except Exception as e:
            logger.error(f"PG FTS search failed: {e}")
            return []

        retrieved = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            score = float(getattr(row, "bm25_score", 0.0))
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                bm25_score=score,
                retrieval_source=RetrievalPath.BM25.value,
                retrieval_method=RetrievalPath.BM25.value,
                final_score=score,
            ))
        logger.debug(f"PG FTS '{clean[:60]}' → {len(retrieved)} chunks")
        return retrieved

    # ──────────────────────────────────────────────────────────────────────
    # Tier 3 — Elasticsearch (TB-scale)
    # ──────────────────────────────────────────────────────────────────────

    async def _es_search(
        self,
        query: str,
        top_k: int,
        law_filter: Optional[List[LawCategory]],
        court_filter: Optional[List[CourtType]],
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[RetrievedChunk]:
        """
        Elasticsearch BM25 search for TB-scale corpora.
        Uses multi_match on content + section_ref with BM25 similarity.
        Metadata filters are pushed down as ES filter clauses (cached, zero scoring cost).
        """
        try:
            client = self._get_es_client()
        except Exception as e:
            logger.warning(f"ES client unavailable, skipping tier 3: {e}")
            return []

        expanded = _expand_legal_synonyms(query)

        must_clauses = [{"multi_match": {
            "query": expanded,
            "fields": ["content^1.0", "section_ref^3.0", "citation^2.0"],
            "type": "best_fields",
            "tie_breaker": 0.3,
        }}]

        filter_clauses = []
        if law_filter:
            filter_clauses.append({"terms": {"law": [l.value for l in law_filter]}})
        if court_filter:
            filter_clauses.append({"terms": {"court": [c.value for c in court_filter]}})
        if year_from or year_to:
            range_filter: Dict[str, Any] = {}
            if year_from:
                range_filter["gte"] = year_from
            if year_to:
                range_filter["lte"] = year_to
            filter_clauses.append({"range": {"year": range_filter}})

        body: Dict[str, Any] = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
            "_source": True,
        }

        try:
            resp = await client.search(
                index=self._settings.es.index_name,
                body=body,
            )
        except Exception as e:
            logger.error(f"ES search failed: {e}")
            return []

        retrieved = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            score = float(hit["_score"])
            # Reconstruct LegalChunk from ES document
            metadata = DocumentMetadata(
                document_id=src.get("document_id", ""),
                document_type=DocumentType(src.get("document_type", "judgment")),
                law=LawCategory(src["law"]) if src.get("law") else None,
                court=CourtType(src["court"]) if src.get("court") else None,
                court_name=src.get("court_name"),
                case_number=src.get("case_number"),
                citation=src.get("citation"),
                year=src.get("year"),
                topic=src.get("topic"),
                keywords=src.get("keywords", []),
                source_url=src.get("source_url"),
                is_landmark=src.get("is_landmark", False),
                language=src.get("language", "en"),
            )
            chunk = LegalChunk(
                chunk_id=src.get("chunk_id", hit["_id"]),
                document_id=src.get("document_id", ""),
                chunk_type=ChunkType(src.get("chunk_type", "passage")),
                content=src.get("content", ""),
                content_length=src.get("content_length", 0),
                chunk_index=src.get("chunk_index", 0),
                page_number=src.get("page_number"),
                section_ref=src.get("section_ref"),
                subsection_ref=src.get("subsection_ref"),
                metadata=metadata,
            )
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                bm25_score=score,
                retrieval_source=RetrievalPath.BM25.value,
                retrieval_method="elasticsearch",
                final_score=score,
            ))
        logger.debug(f"ES '{query[:60]}' → {len(retrieved)} chunks")
        return retrieved

    def _get_es_client(self):
        """Lazy-init Elasticsearch async client."""
        if self._es_client is None:
            from elasticsearch import AsyncElasticsearch
            cfg = self._settings.es
            kwargs: Dict[str, Any] = {
                "hosts": [cfg.hosts],
                "request_timeout": cfg.timeout,
                "retry_on_timeout": True,
                "max_retries": cfg.max_retries,
            }
            if cfg.username and cfg.password:
                kwargs["basic_auth"] = (cfg.username, cfg.password)
            self._es_client = AsyncElasticsearch(**kwargs)
        return self._es_client

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_pg_filter(
        self,
        law_filter: Optional[List[LawCategory]],
        court_filter: Optional[List[CourtType]],
        year_from: Optional[int],
        year_to: Optional[int],
        document_type: Optional[DocumentType],
    ) -> Tuple[str, Dict[str, Any]]:
        conditions = ["c.content_tsv IS NOT NULL"]
        params: Dict[str, Any] = {}
        if law_filter:
            law_vals = [l.value for l in law_filter]
            # BOTH conditions are required:
            # - d.law = ANY(...): filters documents table (index scan on documents)
            # - c.law = ANY(...): allows Postgres to prune chunks partitions at
            #   query planning time. Without c.law in WHERE, the planner cannot
            #   eliminate partitions even though the JOIN result would be the same
            #   — it must scan ALL partitions then join + filter, negating the
            #   entire benefit of LIST(law) partitioning.
            conditions.append("d.law = ANY(:law_filter)")
            conditions.append("c.law = ANY(:law_filter)")
            params["law_filter"] = law_vals
        if court_filter:
            conditions.append("d.court = ANY(:court_filter)")
            params["court_filter"] = [c.value for c in court_filter]
        if year_from:
            conditions.append("d.year >= :year_from")
            params["year_from"] = year_from
        if year_to:
            conditions.append("d.year <= :year_to")
            params["year_to"] = year_to
        if document_type:
            conditions.append("d.document_type = :document_type")
            params["document_type"] = document_type.value
        return " AND ".join(conditions), params

    def _row_to_chunk(self, row) -> LegalChunk:
        """Reconstruct LegalChunk from a PostgreSQL row."""
        metadata = DocumentMetadata(
            document_id=str(row.document_id),
            document_type=DocumentType(row.document_type),
            law=LawCategory(row.law) if row.law else None,
            court=CourtType(row.court) if row.court else None,
            court_name=row.court_name,
            case_number=row.case_number,
            citation=row.citation,
            year=row.year,
            date_decided=getattr(row, "date_decided", None),
            bench=list(row.bench) if getattr(row, "bench", None) else None,
            parties=dict(row.parties) if getattr(row, "parties", None) else None,
            topic=row.topic,
            keywords=list(row.keywords) if getattr(row, "keywords", None) else [],
            source_url=row.source_url,
            is_landmark=getattr(row, "is_landmark", False) or False,
            language=getattr(row, "language", "en") or "en",
        )
        return LegalChunk(
            chunk_id=str(row.chunk_id),
            document_id=str(row.document_id),
            chunk_type=ChunkType(row.chunk_type) if row.chunk_type else ChunkType.PASSAGE,
            content=row.content,
            content_length=row.content_length,
            chunk_index=row.chunk_index,
            page_number=getattr(row, "page_number", None),
            section_ref=getattr(row, "section_ref", None),
            subsection_ref=getattr(row, "subsection_ref", None),
            metadata=metadata,
        )
