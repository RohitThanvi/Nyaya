"""Documents route — list, get, summarize."""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.auth import get_current_active_user
from backend.api.dependencies.pipeline import get_pipeline
from backend.db.session import get_db
from backend.models.domain import SummarizeRequest, UserInDB

router = APIRouter()


@router.get("/documents")
async def list_documents(
    law: str = Query(None),
    court: str = Query(None),
    year: int = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """List documents with optional filters."""
    conditions = ["1=1"]
    params = {"limit": page_size, "offset": (page - 1) * page_size}
    if law:
        conditions.append("law = :law")
        params["law"] = law.upper()
    if court:
        conditions.append("court ILIKE :court")
        params["court"] = f"%{court}%"
    if year:
        conditions.append("year = :year")
        params["year"] = year

    where = " AND ".join(conditions)
    result = await db.execute(
        text(f"""
            SELECT document_id, document_type, law, court_name, citation, year,
                   topic, total_chunks, is_landmark, created_at
            FROM documents WHERE {where}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    docs = [dict(r._mapping) for r in result.fetchall()]
    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM documents WHERE {where}"),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total = count_result.scalar()
    return {"documents": docs, "total": total, "page": page, "page_size": page_size}


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Get a single document with its chunks."""
    result = await db.execute(
        text("SELECT * FROM documents WHERE document_id = :did"),
        {"did": document_id},
    )
    doc = result.fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")
    return dict(doc._mapping)


@router.post("/documents/{document_id}/summarize")
async def summarize_document(
    document_id: str,
    pipeline: AgentPipeline = Depends(get_pipeline),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Summarize a document from the knowledge base."""
    req = SummarizeRequest(document_id=document_id)
    return await pipeline.run_summarize(req, user_id=str(current_user.user_id))


@router.post("/summarize")
async def summarize_text(
    body: SummarizeRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Summarize provided text."""
    if not body.text:
        raise HTTPException(400, "text field required for text summarization")
    return await pipeline.run_summarize(body, user_id=str(current_user.user_id))
