"""Drafting route."""
import logging
from fastapi import APIRouter, Depends
from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.pipeline import get_pipeline
from backend.models.domain import DraftRequest, LegalResponse

router = APIRouter(prefix="/draft", tags=["drafting"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=LegalResponse)
async def generate_draft(
    request: DraftRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
) -> LegalResponse:
    return await pipeline.run_draft(request)
