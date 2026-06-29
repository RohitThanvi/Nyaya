"""
Chat route v2 — adds DELETE /sessions/{id}, fixes token refresh.
"""
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.pipeline import get_pipeline
from backend.db.session import get_db
from backend.models.domain import ChatRequest, LegalResponse

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=LegalResponse)
async def chat(
    request: ChatRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
):
    """Non-streaming chat endpoint."""
    return await pipeline.run_chat(request)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
):
    """SSE streaming chat endpoint."""
    async def event_generator():
        try:
            async for token in pipeline.run_chat_stream(request):
                data = json.dumps({"token": token, "done": False})
                yield f"data: {data}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("""
        SELECT session_id, title, created_at, updated_at
        FROM chat_sessions
        ORDER BY updated_at DESC
        LIMIT :limit
    """), {"limit": limit})
    rows = result.fetchall()
    return [
        {
            "session_id": str(r.session_id),
            "title": r.title,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("""
        SELECT message_id, role, content, citations,
               hallucination_flags, confidence, intent, latency_ms, created_at
        FROM chat_messages
        WHERE session_id = :session_id
        ORDER BY created_at
    """), {"session_id": session_id})
    rows = result.fetchall()
    return [
        {
            "message_id": str(r.message_id),
            "role": r.role,
            "content": r.content,
            "citations": r.citations or [],
            "hallucination_flags": r.hallucination_flags or [],
            "confidence": r.confidence,
            "intent": r.intent,
            "latency_ms": r.latency_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat session and all its messages."""
    result = await db.execute(text("""
        DELETE FROM chat_sessions WHERE session_id = :session_id
    """), {"session_id": session_id})
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found")
