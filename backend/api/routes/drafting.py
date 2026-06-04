"""Drafting route."""
from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.auth import get_current_active_user
from backend.api.dependencies.pipeline import get_pipeline
from backend.models.domain import DraftRequest, UserInDB

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/draft")
@limiter.limit("10/minute")
async def generate_draft(
    request: Request,
    body: DraftRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Generate a legal document draft using templates + retrieved law."""
    return await pipeline.run_draft(body, user_id=str(current_user.user_id))
