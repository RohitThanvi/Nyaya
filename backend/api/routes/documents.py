"""
Documents route v2 — summarisation, chunk viewer, source document access.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.pipeline import get_pipeline
from backend.db.session import get_db
from backend.models.domain import SummarizeRequest

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.get("")
async def list_documents(
    law: str = None,
    court: str = None,
    year: int = None,
    document_type: str = None,
    search: str = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Paginated document listing. Returns {documents, total, page, page_size}
    — matches the contract the frontend (judgments page) actually expects.
    total_chunks is computed via a correlated subquery, not a stored column
    (the column never existed in the real schema — only in the original
    repo's stale Alembic file that was deleted).
    """
    conditions = ["1=1"]
    params: dict = {"limit": page_size, "offset": (page - 1) * page_size}

    if law:
        conditions.append("d.law = :law")
        params["law"] = law
    if court:
        conditions.append("d.court = :court")
        params["court"] = court
    if year:
        conditions.append("d.year = :year")
        params["year"] = year
    if document_type:
        conditions.append("d.document_type = :document_type")
        params["document_type"] = document_type
    if search:
        conditions.append(
            "(d.citation ILIKE :search OR d.case_number ILIKE :search OR d.court_name ILIKE :search)"
        )
        params["search"] = f"%{search}%"

    where = " AND ".join(conditions)

    count_result = await db.execute(text(f"SELECT COUNT(*) FROM documents d WHERE {where}"), params)
    total = count_result.scalar() or 0

    result = await db.execute(text(f"""
        SELECT d.document_id, d.document_type, d.law, d.court, d.court_name,
               d.case_number, d.citation, d.year, d.topic, d.source_url,
               d.is_landmark, d.pages, d.original_filename, d.created_at,
               d.parties, d.bench,
               (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.document_id) AS total_chunks
        FROM documents d
        WHERE {where}
        ORDER BY d.created_at DESC
        LIMIT :limit OFFSET :offset
    """), params)
    rows = result.fetchall()

    docs = []
    for row in rows:
        d = dict(row._mapping)
        if d.get("source_url"):
            d["access_url"] = d["source_url"]
        else:
            d["access_url"] = f"/api/v1/documents/{d['document_id']}/view"
        docs.append(d)

    return {
        "documents": docs,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{document_id}")
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT d.document_id, d.document_type, d.law, d.court, d.court_name,
               d.case_number, d.citation, d.year, d.date_decided, d.bench, d.parties,
               d.topic, d.keywords, d.source_url, d.original_filename,
               d.is_landmark, d.language, d.pages, d.created_at,
               (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.document_id) AS total_chunks
        FROM documents d
        WHERE d.document_id = :doc_id
    """), {"doc_id": document_id})
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    d = dict(row._mapping)
    if d.get("source_url"):
        d["access_url"] = d["source_url"]
    else:
        d["access_url"] = f"/api/v1/documents/{document_id}/view"
    return d


@router.get("/{document_id}/view")
async def view_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """
    Serve the original document file.
    If source_url exists → redirect to it.
    If original file is stored locally → serve as FileResponse.
    Otherwise → return full text as plain text.
    """
    result = await db.execute(text("""
        SELECT source_url, original_filename, document_type, citation, court_name
        FROM documents WHERE document_id = :doc_id
    """), {"doc_id": document_id})
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    # Redirect to external source
    if row.source_url:
        return RedirectResponse(url=row.source_url, status_code=302)

    # Serve local file if it exists
    import os
    from backend.config.settings import get_settings
    cfg = get_settings()
    if row.original_filename:
        local_path = os.path.join(cfg.app.upload_dir, row.original_filename)
        if os.path.exists(local_path):
            return FileResponse(
                path=local_path,
                media_type="application/pdf",
                filename=row.original_filename,
            )

    # Fallback: return full text from chunks
    chunks_result = await db.execute(text("""
        SELECT content FROM chunks
        WHERE document_id = :doc_id
        ORDER BY chunk_index
    """), {"doc_id": document_id})
    chunks = chunks_result.fetchall()
    if not chunks:
        raise HTTPException(status_code=404, detail="Document content not available")

    full_text = "\n\n".join(r.content for r in chunks)
    title = row.citation or row.court_name or "Legal Document"
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=f"{title}\n{'='*len(title)}\n\n{full_text}",
        media_type="text/plain; charset=utf-8",
    )


@router.get("/{document_id}/full-text")
async def get_full_text(document_id: str, db: AsyncSession = Depends(get_db)):
    """Return the full reconstructed text of a document from chunks."""
    result = await db.execute(text("""
        SELECT c.content, c.chunk_index, c.chunk_type, c.section_ref, c.page_number,
               d.citation, d.court_name, d.year
        FROM chunks c
        JOIN documents d ON c.document_id = d.document_id
        WHERE c.document_id = :doc_id
        ORDER BY c.chunk_index
    """), {"doc_id": document_id})
    rows = result.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No content found for this document")

    return {
        "document_id": document_id,
        "citation": rows[0].citation,
        "court_name": rows[0].court_name,
        "year": rows[0].year,
        "chunks": [
            {
                "index": r.chunk_index,
                "type": r.chunk_type,
                "section_ref": r.section_ref,
                "page_number": r.page_number,
                "content": r.content,
            }
            for r in rows
        ],
        "full_text": "\n\n".join(r.content for r in rows),
    }


@router.post("/{document_id}/summarize")
async def summarize_document(
    document_id: str,
    pipeline: AgentPipeline = Depends(get_pipeline),
):
    request = SummarizeRequest(document_id=document_id, summary_type="full")
    result = await pipeline.run_summarize(request)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/summarize-text")
async def summarize_raw_text(
    request: SummarizeRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
):
    """
    Summarize raw pasted judgment text with no stored/ingested document.
    Use /{document_id}/summarize instead for anything already in the corpus.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="text is required for this endpoint")
    result = await pipeline.run_summarize(request)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    chunk_type: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    params: dict = {"doc_id": document_id, "limit": limit}
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
    return [dict(r._mapping) for r in result.fetchall()]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete a document and purge its vectors from Qdrant.

    Postgres cascade handles chunks/staged_chunks. Qdrant must be
    cleaned explicitly — without this, deleted document vectors stay in
    the ANN index forever, poisoning every semantic search with ghost
    results from documents the user explicitly removed.
    """
    # Verify exists before deleting
    row = await db.execute(
        text("SELECT document_id FROM documents WHERE document_id = :doc_id"),
        {"doc_id": document_id}
    )
    if not row.fetchone():
        raise HTTPException(status_code=404, detail="Document not found")

    # Collect chunk IDs before cascade delete removes them
    chunk_rows = await db.execute(
        text("SELECT chunk_id FROM chunks WHERE document_id = :doc_id"),
        {"doc_id": document_id}
    )
    chunk_ids = [str(r.chunk_id) for r in chunk_rows.fetchall()]

    # Delete from Postgres (cascade removes chunks + staged_chunks)
    await db.execute(
        text("DELETE FROM documents WHERE document_id = :doc_id"),
        {"doc_id": document_id}
    )
    await db.commit()

    # Purge from Qdrant — non-fatal: log and continue if Qdrant is unavailable
    if chunk_ids:
        try:
            from backend.retrieval.vector.retriever import VectorRetriever
            from qdrant_client.models import PointIdsList
            vr = VectorRetriever()
            client = vr._get_client()
            await client.delete(
                collection_name=vr._search_target(),
                points_selector=PointIdsList(points=chunk_ids),
                wait=False,
            )
            logger.info(
                f"Deleted {len(chunk_ids)} vectors from Qdrant "
                f"for document {document_id}"
            )
        except Exception as e:
            logger.error(
                f"Qdrant vector deletion failed for document {document_id}: {e}. "
                f"Postgres rows deleted successfully. Run admin/qdrant/rebuild "
                f"to clean orphaned vectors if this error persists."
            )
