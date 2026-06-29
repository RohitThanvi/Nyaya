"""
Documents route v2 — adds judgment summarisation endpoint with hierarchical pipeline.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.pipeline import get_pipeline
from backend.db.session import get_db
from backend.models.domain import SummarizeRequest

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.get("/")
async def list_documents(
    law: str = None,
    court: str = None,
    year: int = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List documents with optional filters."""
    conditions = ["1=1"]
    params = {"limit": limit, "offset": offset}

    if law:
        conditions.append("d.law = :law")
        params["law"] = law
    if court:
        conditions.append("d.court = :court")
        params["court"] = court
    if year:
        conditions.append("d.year = :year")
        params["year"] = year

    where = " AND ".join(conditions)
    result = await db.execute(text(f"""
        SELECT document_id, document_type, law, court, court_name,
               case_number, citation, year, topic, source_url,
               is_landmark, pages, created_at
        FROM documents d
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """), params)
    rows = result.fetchall()
    return [dict(row._mapping) for row in rows]


@router.get("/{document_id}")
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single document with its metadata."""
    result = await db.execute(text("""
        SELECT document_id, document_type, law, court, court_name,
               case_number, citation, year, date_decided, bench, parties,
               topic, keywords, source_url, is_landmark, language, pages, created_at
        FROM documents
        WHERE document_id = :doc_id
    """), {"doc_id": document_id})
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return dict(row._mapping)


@router.post("/{document_id}/summarize")
async def summarize_document(
    document_id: str,
    pipeline: AgentPipeline = Depends(get_pipeline),
):
    """
    Full hierarchical summarisation of a judgment.
    Fetches ALL chunks, uses typed routing via summarize_chunks().
    No 40-chunk LIMIT, no 12,000-char truncation.
    """
    request = SummarizeRequest(document_id=document_id, summary_type="full")
    result = await pipeline.run_summarize(request)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    chunk_type: str = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """Get chunks for a document, optionally filtered by chunk_type."""
    params = {"doc_id": document_id, "limit": limit}
    type_clause = ""
    if chunk_type:
        type_clause = "AND chunk_type = :chunk_type"
        params["chunk_type"] = chunk_type

    result = await db.execute(text(f"""
        SELECT chunk_id, chunk_type, content, chunk_index,
               page_number, section_ref, subsection_ref, content_length
        FROM chunks
        WHERE document_id = :doc_id {type_clause}
        ORDER BY chunk_index
        LIMIT :limit
    """), params)
    rows = result.fetchall()
    return [dict(row._mapping) for row in rows]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a document and all its chunks (cascades to Qdrant via background task)."""
    result = await db.execute(text("""
        DELETE FROM documents WHERE document_id = :doc_id
    """), {"doc_id": document_id})
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Document not found")
