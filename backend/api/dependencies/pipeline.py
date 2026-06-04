"""
FastAPI dependency injection for all pipeline components.
Uses module-level singletons for expensive resources (models, DB pools).
"""
from functools import lru_cache
from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base.pipeline import AgentPipeline
from backend.agents.drafting.agent import DraftingAgent
from backend.agents.legal_mapping.agent import LegalMappingAgent
from backend.agents.query_understanding.agent import QueryUnderstandingAgent
from backend.agents.summarization.agent import SummarizationAgent
from backend.agents.verification.agent import VerificationAgent
from backend.db.session import get_db
from backend.embeddings.service import EmbeddingService
from backend.retrieval.bm25.retriever import BM25Retriever
from backend.retrieval.hybrid.pipeline import HybridRetriever
from backend.retrieval.reranker.cross_encoder import Reranker
from backend.retrieval.vector.retriever import VectorRetriever
from backend.utils.llm_client import get_llm_client
from backend.utils.redis_client import get_redis_client


# ── Singleton components (expensive to create) ──

@lru_cache(maxsize=1)
def get_vector_retriever() -> VectorRetriever:
    return VectorRetriever()


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


async def get_embedding_service() -> EmbeddingService:
    redis = None
    try:
        redis = await get_redis_client()
    except Exception:
        pass
    return EmbeddingService(redis_client=redis)


# ── Request-scoped components ──

async def get_bm25_retriever(
    db: AsyncSession = Depends(get_db),
) -> BM25Retriever:
    return BM25Retriever(db=db)


async def get_hybrid_retriever(
    bm25: BM25Retriever = Depends(get_bm25_retriever),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> HybridRetriever:
    return HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=get_vector_retriever(),
        embedding_service=embedding_service,
        reranker=get_reranker(),
    )


async def get_pipeline(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> AgentPipeline:
    return AgentPipeline(
        query_agent=QueryUnderstandingAgent(),
        hybrid_retriever=retriever,
        legal_mapping_agent=LegalMappingAgent(),
        verification_agent=VerificationAgent(),
        summarization_agent=SummarizationAgent(),
        drafting_agent=DraftingAgent(),
        llm_client=get_llm_client(),
    )
