"""
BM25-equivalent lexical retrieval using PostgreSQL tsvector/tsquery.
Combines ts_rank_cd (BM25-like) with trigram similarity for robustness.
"""
import logging
import re
from typing import List, Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.domain import (
    DocumentMetadata, DocumentType, LawCategory, LegalChunk,
    RetrievedChunk, ChunkType, CourtType
)
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    PostgreSQL full-text search retriever.

    Uses:
    - tsvector for lexical matching (BM25-like scoring via ts_rank_cd)
    - pg_trgm for fuzzy/partial matching
    - Metadata filters pushed down to SQL

    ts_rank_cd uses term frequency with document length normalization,
    which approximates BM25 behavior.
    """

    def __init__(self, db: AsyncSession):
        self._db = db
        self._settings = get_settings().retrieval

    def _sanitize_query(self, query: str) -> str:
        """Clean user input to prevent tsquery injection."""
        # Remove special tsquery operators that user shouldn't inject
        query = re.sub(r"[&|!<>():]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query

    def _build_tsquery(self, query: str) -> str:
        """
        Convert natural language query to plainto_tsquery format.
        Uses websearch_to_tsquery for better handling of multi-word queries.
        """
        return query  # passed as parameterized input to websearch_to_tsquery

    def _build_filter_clause(
        self,
        law_filter: Optional[List[LawCategory]] = None,
        court_filter: Optional[List[CourtType]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_type: Optional[DocumentType] = None,
        section_refs: Optional[List[str]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Build parameterized WHERE clauses from filters."""
        conditions = ["c.content_tsv IS NOT NULL"]
        params: Dict[str, Any] = {}

        if law_filter:
            conditions.append("d.law = ANY(:law_filter)")
            params["law_filter"] = [l.value for l in law_filter]

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

        if section_refs:
            conditions.append("c.section_ref = ANY(:section_refs)")
            params["section_refs"] = section_refs

        where = " AND ".join(conditions)
        return where, params

    async def search(
        self,
        query: str,
        top_k: int = 20,
        law_filter: Optional[List[LawCategory]] = None,
        court_filter: Optional[List[CourtType]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_type: Optional[DocumentType] = None,
        section_refs: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute BM25 search with metadata filtering.

        Uses ts_rank_cd for ranking which normalizes by document length,
        approximating BM25's document length normalization.
        """
        clean_query = self._sanitize_query(query)
        if not clean_query:
            return []

        where_clause, params = self._build_filter_clause(
            law_filter, court_filter, year_from, year_to, document_type, section_refs
        )

        params["query"] = clean_query
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
                    c.qdrant_id,
                    d.document_type,
                    d.law,
                    d.court,
                    d.court_name,
                    d.case_number,
                    d.citation,
                    d.year,
                    d.date_decided,
                    d.bench,
                    d.parties,
                    d.topic,
                    d.keywords,
                    d.source_url,
                    d.is_landmark,
                    d.language,
                    -- ts_rank_cd: BM25-like normalization (option 32 = divide by length)
                    ts_rank_cd(
                        c.content_tsv,
                        websearch_to_tsquery('english', :query),
                        32
                    ) AS bm25_score,
                    -- headline for snippet extraction
                    ts_headline(
                        'english',
                        c.content,
                        websearch_to_tsquery('english', :query),
                        'MaxFragments=2,MaxWords=40,MinWords=10,StartSel=<mark>,StopSel=</mark>'
                    ) AS headline
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE
                    {where_clause}
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
            logger.error(f"BM25 search failed: {e}")
            return []

        retrieved: List[RetrievedChunk] = []
        for row in rows:
            metadata = DocumentMetadata(
                document_id=str(row.document_id),
                document_type=DocumentType(row.document_type),
                law=LawCategory(row.law) if row.law else None,
                court=CourtType(row.court) if row.court else None,
                court_name=row.court_name,
                case_number=row.case_number,
                citation=row.citation,
                year=row.year,
                date_decided=row.date_decided,
                bench=list(row.bench) if row.bench else None,
                parties=dict(row.parties) if row.parties else None,
                topic=row.topic,
                keywords=list(row.keywords) if row.keywords else [],
                source_url=row.source_url,
                is_landmark=row.is_landmark or False,
                language=row.language or "en",
            )
            chunk = LegalChunk(
                chunk_id=str(row.chunk_id),
                document_id=str(row.document_id),
                chunk_type=ChunkType(row.chunk_type) if row.chunk_type else ChunkType.PASSAGE,
                content=row.content,
                content_length=row.content_length,
                chunk_index=row.chunk_index,
                page_number=row.page_number,
                section_ref=row.section_ref,
                subsection_ref=row.subsection_ref,
                metadata=metadata,
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    bm25_score=float(row.bm25_score),
                    retrieval_source="bm25",
                )
            )

        logger.debug(f"BM25 returned {len(retrieved)} results for: {clean_query[:60]}")
        return retrieved

    async def section_lookup(self, section_number: str, law: LawCategory) -> List[RetrievedChunk]:
        """
        Direct section lookup — bypasses full-text search.
        Used when user asks specifically about e.g. 'BNS Section 302'.
        """
        sql = text("""
            SELECT
                c.chunk_id, c.document_id, c.chunk_type, c.content,
                c.content_length, c.chunk_index, c.page_number,
                c.section_ref, c.subsection_ref,
                d.document_type, d.law, d.court, d.court_name,
                d.citation, d.year, d.topic, d.keywords,
                d.source_url, d.is_landmark, d.language
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE
                (c.section_ref ILIKE :section_pattern OR c.subsection_ref ILIKE :section_pattern)
                AND d.law = :law
            ORDER BY c.chunk_index
            LIMIT 10
        """)
        params = {
            "section_pattern": f"%{section_number}%",
            "law": law.value,
        }
        try:
            result = await self._db.execute(sql, params)
            rows = result.fetchall()
        except Exception as e:
            logger.error(f"Section lookup failed: {e}")
            return []

        retrieved = []
        for row in rows:
            metadata = DocumentMetadata(
                document_id=str(row.document_id),
                document_type=DocumentType(row.document_type),
                law=LawCategory(row.law) if row.law else None,
                court=CourtType(row.court) if row.court else None,
                court_name=row.court_name,
                citation=row.citation,
                year=row.year,
                topic=row.topic,
                keywords=list(row.keywords) if row.keywords else [],
                source_url=row.source_url,
                is_landmark=row.is_landmark or False,
                language=row.language or "en",
            )
            chunk = LegalChunk(
                chunk_id=str(row.chunk_id),
                document_id=str(row.document_id),
                chunk_type=ChunkType(row.chunk_type) if row.chunk_type else ChunkType.SECTION,
                content=row.content,
                content_length=row.content_length,
                chunk_index=row.chunk_index,
                section_ref=row.section_ref,
                subsection_ref=row.subsection_ref,
                metadata=metadata,
            )
            retrieved.append(
                RetrievedChunk(chunk=chunk, bm25_score=1.0, retrieval_source="direct_lookup")
            )
        return retrieved
