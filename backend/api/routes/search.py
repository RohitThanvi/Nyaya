"""Search API routes."""
from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.agents.base.pipeline import AgentPipeline
from backend.api.dependencies.auth import get_current_active_user
from backend.api.dependencies.pipeline import get_pipeline
from backend.models.domain import LegalResponse, SearchRequest, UserInDB

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/search", response_model=LegalResponse)
@limiter.limit("30/minute")
async def legal_search(
    request: Request,
    body: SearchRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
    current_user: UserInDB = Depends(get_current_active_user),
) -> LegalResponse:
    """
    Full hybrid legal search with citation-backed response.

    Runs: query understanding → BM25+ANN retrieval → reranking →
          legal mapping → LLM generation → verification → structured output.
    """
    return await pipeline.run_search(body, user_id=str(current_user.user_id))


@router.get("/search/sections/{law}/{section_number}")
async def lookup_section(
    law: str,
    section_number: str,
    pipeline: AgentPipeline = Depends(get_pipeline),
    current_user: UserInDB = Depends(get_current_active_user),
) -> LegalResponse:
    """Direct section lookup by law and section number."""
    from backend.models.domain import LawCategory, SearchRequest
    try:
        law_enum = LawCategory(law.upper())
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(400, f"Unknown law: {law}. Use BNS, BNSS, BSA, IPC, CrPC")

    query = f"{law.upper()} Section {section_number}"
    req = SearchRequest(query=query, law_filter=[law_enum], top_k=5)
    return await pipeline.run_search(req, user_id=str(current_user.user_id))
