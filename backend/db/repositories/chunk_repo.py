"""
Chunk repository — database access layer for chunks.
"""
import logging
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ChunkRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def section_exists(self, section_ref: str) -> bool:
        """Check if a section reference exists in the DB."""
        result = await self._db.execute(
            text("SELECT 1 FROM chunks WHERE section_ref ILIKE :ref LIMIT 1"),
            {"ref": f"%{section_ref}%"},
        )
        return result.fetchone() is not None

    async def get_chunks_by_document(self, document_id: str) -> List[dict]:
        result = await self._db.execute(
            text("SELECT * FROM chunks WHERE document_id = :doc_id ORDER BY chunk_index"),
            {"doc_id": document_id},
        )
        return [dict(r._mapping) for r in result.fetchall()]

    async def update_qdrant_id(self, chunk_id: str, qdrant_id: str) -> None:
        await self._db.execute(
            text("UPDATE chunks SET qdrant_id = :qid WHERE chunk_id = :cid"),
            {"qid": qdrant_id, "cid": chunk_id},
        )
