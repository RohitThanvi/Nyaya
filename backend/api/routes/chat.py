"""
Chat API — supports both streaming (SSE) and non-streaming responses.
Persists conversation history to PostgreSQL.
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.auth import get_current_active_user
from backend.api.dependencies.pipeline import get_pipeline
from backend.db.session import get_db
from backend.models.domain import (
    ChatMessage, ChatRequest, LegalResponse, UserInDB
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=LegalResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user),
) -> LegalResponse:
    """
    Non-streaming chat. Returns complete JSON response.
    Use for programmatic access or when streaming is not needed.
    """
    response = await pipeline.run_chat(
        message=body.message,
        history=body.history,
        user_id=str(current_user.user_id),
        law_filter=body.law_filter,
    )
    await _persist_message(db, body, response, current_user)
    return response


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
    current_user: UserInDB = Depends(get_current_active_user),
) -> StreamingResponse:
    """
    Streaming chat via Server-Sent Events.
    Frontend should consume as EventSource or fetch with streaming reader.
    Each event: data: <token>\n\n
    Final event: data: [DONE]\n\n
    """
    async def event_generator():
        try:
            async for token in pipeline.stream_chat(
                message=body.message,
                history=body.history,
                user_id=str(current_user.user_id),
                law_filter=body.law_filter,
            ):
                # Escape newlines in token for SSE format
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/chat/sessions")
async def get_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """List user's chat sessions."""
    result = await db.execute(
        text("""
            SELECT session_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = :uid
            ORDER BY updated_at DESC
            LIMIT 50
        """),
        {"uid": str(current_user.user_id)},
    )
    sessions = [dict(r._mapping) for r in result.fetchall()]
    return {"sessions": sessions}


@router.get("/chat/sessions/{session_id}")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Get messages for a specific session."""
    # Verify ownership
    session = await db.execute(
        text("SELECT * FROM chat_sessions WHERE session_id = :sid AND user_id = :uid"),
        {"sid": session_id, "uid": str(current_user.user_id)},
    )
    if not session.fetchone():
        raise HTTPException(404, "Session not found")

    msgs = await db.execute(
        text("""
            SELECT message_id, role, content, citations, confidence, created_at
            FROM chat_messages WHERE session_id = :sid ORDER BY created_at
        """),
        {"sid": session_id},
    )
    return {"messages": [dict(r._mapping) for r in msgs.fetchall()]}


async def _persist_message(
    db: AsyncSession,
    request: ChatRequest,
    response: LegalResponse,
    user: UserInDB,
) -> None:
    """Save user + assistant messages to DB."""
    try:
        # Ensure session exists
        session_id = request.session_id
        if not session_id:
            result = await db.execute(
                text("""
                    INSERT INTO chat_sessions (user_id, title)
                    VALUES (:uid, :title) RETURNING session_id
                """),
                {
                    "uid": str(user.user_id),
                    "title": request.message[:80],
                },
            )
            session_id = str(result.scalar())

        # User message
        await db.execute(
            text("""
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (:sid, 'user', :content)
            """),
            {"sid": session_id, "content": request.message},
        )
        # Assistant response
        citations_json = json.dumps([c.model_dump() for c in response.citations])
        await db.execute(
            text("""
                INSERT INTO chat_messages (session_id, role, content, citations, confidence, latency_ms)
                VALUES (:sid, 'assistant', :content, :citations::jsonb, :confidence, :latency)
            """),
            {
                "sid": session_id,
                "content": response.answer,
                "citations": citations_json,
                "confidence": response.confidence,
                "latency": response.latency_ms,
            },
        )
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist chat message: {e}")
