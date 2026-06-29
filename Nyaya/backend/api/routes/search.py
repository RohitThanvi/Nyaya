"""
Search route v2 — rate limited, returns full Citation provenance including source_url.
"""
import logging

from fastapi import APIRouter, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.pipeline import get_pipeline
from backend.models.domain import LegalResponse, SearchRequest

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)


@router.post("/", response_model=LegalResponse)
async def search(
    request: SearchRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
) -> LegalResponse:
    """
    Hybrid legal search — three-path retrieval + reranking + verification.
    Returns citations with source_url and page_number for frontend deep-linking.
    """
    return await pipeline.run_search(request)
