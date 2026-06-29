"""
ChunkRepository — now properly wired and used by VerificationAgent.

All DB fallback queries for citation verification go through this class.
Returns source_url, page_number, and snippet on every result.
"""
import logging
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ChunkRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_by_section(
        self,
        section_num: str,
        laws: Optional[List[str]] = None,
        limit: int = 3,
    ) -> List[Dict]:
        """
        Find chunks matching a section number.
        Returns list of dicts with chunk_id, document_id, content,
        page_number, source_url, law, year, court.
        """
        params: Dict = {"section": section_num, "section_like": f"{section_num}%", "limit": limit}
        law_clause = ""
        if laws:
            law_clause = "AND d.law = ANY(:laws)"
            params["laws"] = laws

        result = await self._db.execute(text(f"""
            SELECT c.chunk_id, c.document_id, c.content, c.page_number,
                   c.section_ref, c.subsection_ref,
                   d.source_url, d.law, d.year, d.court, d.court_name, d.citation
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE (c.section_ref = :section OR c.subsection_ref ILIKE :section_like)
                  {law_clause}
            ORDER BY c.chunk_index
            LIMIT :limit
        """), params)
        rows = result.fetchall()
        return [dict(r._mapping) for r in rows]

    async def find_by_citation(
        self, normalised_citation: str, limit: int = 2
    ) -> List[Dict]:
        """
        Find document + representative chunk by normalised citation string.
        """
        result = await self._db.execute(text("""
            SELECT d.document_id, d.source_url, d.citation, d.year, d.court, d.court_name,
                   c.chunk_id, c.content, c.page_number
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.document_id
                               AND c.chunk_type = 'final_order'
            WHERE LOWER(REGEXP_REPLACE(d.citation, '[().,\\s]+', ' ', 'g')) = :norm_cit
            LIMIT :limit
        """), {"norm_cit": normalised_citation, "limit": limit})
        rows = result.fetchall()
        return [dict(r._mapping) for r in rows]

    async def get_section_snippet(
        self, document_id: str, section_ref: str
    ) -> Optional[str]:
        """Get a 150-char snippet for a specific section in a document."""
        result = await self._db.execute(text("""
            SELECT content FROM chunks
            WHERE document_id = :doc_id AND section_ref = :section
            ORDER BY chunk_index LIMIT 1
        """), {"doc_id": document_id, "section": section_ref})
        row = result.fetchone()
        return row.content[:150] if row else None

    async def get_source_url(self, document_id: str) -> Optional[str]:
        """Get source_url for a document."""
        result = await self._db.execute(text("""
            SELECT source_url FROM documents WHERE document_id = :doc_id
        """), {"doc_id": document_id})
        row = result.fetchone()
        return row.source_url if row else None
