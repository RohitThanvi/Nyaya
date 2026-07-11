"""
Chat routes — SSE streaming, session management, message persistence.

Critical fix: chat messages were NEVER written to the database.
chat_messages and chat_sessions tables existed, list_sessions and
get_session_messages routes existed, but run_chat/run_chat_stream
never did a single INSERT. Every session appeared empty on reload;
history was always rebuilt from the frontend payload (which loses
context on page refresh). Fixed by persisting both user message
and assistant response after every successful pipeline run.
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
from backend.api.dependencies.auth import get_current_user
from backend.api.dependencies.pipeline import get_pipeline
from backend.db.session import get_db
from backend.models.domain import ChatRequest, LegalResponse, UserInDB

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


# ── Message persistence helpers ───────────────────────────────────────────────

async def _ensure_session(
    db: AsyncSession,
    session_id: str,
    user_id: str,
    first_message: str,
) -> None:
    """Upsert a chat session row (idempotent — safe to call on every message)."""
    title = first_message[:80].strip()
    await db.execute(text("""
        INSERT INTO chat_sessions (session_id, user_id, title, created_at, updated_at)
        VALUES (:sid, :uid, :title, NOW(), NOW())
        ON CONFLICT (session_id) DO UPDATE SET updated_at = NOW()
    """), {"sid": session_id, "uid": user_id, "title": title})


async def _persist_turn(
    db: AsyncSession,
    session_id: str,
    user_id: str,
    user_message: str,
    response: LegalResponse,
) -> None:
    """
    Write user message + assistant response to chat_messages.
    Called after every successful pipeline run.
    Previously this never happened — sessions always appeared empty on reload.
    """
    try:
        await _ensure_session(db, session_id, user_id, user_message)

        # User turn
        await db.execute(text("""
            INSERT INTO chat_messages
                (message_id, session_id, role, content, created_at)
            VALUES (:mid, :sid, 'user', :content, NOW())
        """), {"mid": str(uuid.uuid4()), "sid": session_id, "content": user_message})

        # Assistant turn — store all verifiable fields for history
        await db.execute(text("""
            INSERT INTO chat_messages
                (message_id, session_id, role, content,
                 citations, hallucination_flags, confidence, intent,
                 latency_ms, created_at)
            VALUES
                (:mid, :sid, 'assistant', :content,
                 :citations::jsonb, :flags::jsonb, :confidence, :intent,
                 :latency::jsonb, NOW())
        """), {
            "mid":        str(uuid.uuid4()),
            "sid":        session_id,
            "content":    response.answer,
            "citations":  json.dumps([c.model_dump() if hasattr(c, "model_dump") else c
                                       for c in (response.citations or [])]),
            "flags":      json.dumps(response.hallucination_flags or []),
            "confidence": response.confidence,
            "intent":     response.intent,
            "latency":    json.dumps({"total_ms": response.latency_ms}
                                      if response.latency_ms else {}),
        })
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to persist chat turn (non-fatal): {e}")
        try:
            await db.rollback()
        except Exception:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("", response_model=LegalResponse)
async def chat(
    request: ChatRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user),
):
    """Non-streaming chat with message persistence."""
    result = await pipeline.run_chat(request, user_id=current_user.user_id)

    session_id = request.session_id or result.session_id
    await _persist_turn(db, session_id, current_user.user_id, request.message, result)

    # Always return the canonical session_id so the frontend can thread messages
    result.session_id = session_id
    return result


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user),
):
    """SSE streaming chat with message persistence after stream completes."""
    accumulated_tokens: list[str] = []
    final_response: Optional[LegalResponse] = None

    async def event_generator():
        nonlocal final_response
        try:
            async for event in pipeline.run_chat_stream(
                request, user_id=current_user.user_id
            ):
                if isinstance(event, str):
                    # Token
                    accumulated_tokens.append(event)
                    yield f"data: {json.dumps({'token': event, 'done': False})}\n\n"
                elif isinstance(event, LegalResponse):
                    # Final structured response emitted after stream completes
                    final_response = event
                    yield f"data: {json.dumps({'done': True, 'response': event.model_dump()})}\n\n"
                else:
                    yield f"data: {json.dumps({'token': str(event), 'done': False})}\n\n"

            if final_response is None:
                # Pipeline emitted only tokens — build a minimal response for persistence
                from backend.models.domain import LegalResponse as LR
                final_response = LR(
                    query=request.message,
                    session_id=request.session_id or str(uuid.uuid4()),
                    answer="".join(accumulated_tokens),
                    confidence=0.0,
                    warnings=["Structured response not available for this stream."],
                )
            if not any(accumulated_tokens):
                yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        # Persist after stream completes (non-blocking from client's perspective)
        if final_response:
            session_id = request.session_id or final_response.session_id
            await _persist_turn(
                db, session_id, current_user.user_id, request.message, final_response
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user),
):
    """List sessions for the current user only (was missing user_id filter)."""
    result = await db.execute(text("""
        SELECT session_id, title, created_at, updated_at
        FROM chat_sessions
        WHERE user_id = :uid
        ORDER BY updated_at DESC
        LIMIT :limit
    """), {"uid": current_user.user_id, "limit": limit})
    return [
        {
            "session_id": str(r.session_id),
            "title": r.title,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in result.fetchall()
    ]


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user),
):
    """Get messages for a session — scoped to current user to prevent data leaks."""
    # Verify ownership
    owns = await db.execute(text("""
        SELECT 1 FROM chat_sessions
        WHERE session_id = :sid AND user_id = :uid
    """), {"sid": session_id, "uid": current_user.user_id})
    if not owns.fetchone():
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(text("""
        SELECT message_id, role, content, citations,
               hallucination_flags, confidence, intent, latency_ms, created_at
        FROM chat_messages
        WHERE session_id = :sid
        ORDER BY created_at ASC
    """), {"sid": session_id})
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
        for r in result.fetchall()
    ]


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user),
):
    result = await db.execute(text("""
        DELETE FROM chat_sessions
        WHERE session_id = :sid AND user_id = :uid
    """), {"sid": session_id, "uid": current_user.user_id})
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found")
