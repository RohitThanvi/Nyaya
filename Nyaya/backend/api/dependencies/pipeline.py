"""
FastAPI dependency: builds AgentPipeline per-request with proper injection.
All agents receive live DB session and LLM client — no more None injection.
"""
from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base.pipeline import AgentPipeline
from backend.db.session import get_db
from backend.embeddings.service import EmbeddingService
from backend.retrieval.bm25.retriever import BM25Retriever
from backend.retrieval.hybrid.pipeline import HybridRetriever
from backend.retrieval.reranker.cross_encoder import Reranker
from backend.retrieval.vector.retriever import VectorRetriever
from backend.utils.llm_client import LLMClient

# Module-level singletons (loaded once at startup)
_embedding_service: EmbeddingService | None = None
_vector_retriever: VectorRetriever | None = None
_reranker: Reranker | None = None
_llm_client: LLMClient | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_retriever() -> VectorRetriever:
    global _vector_retriever
    if _vector_retriever is None:
        _vector_retriever = VectorRetriever()
    return _vector_retriever


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def get_pipeline(
    db: AsyncSession = Depends(get_db),
) -> AsyncGenerator[AgentPipeline, None]:
    """
    Dependency that yields a fully-wired AgentPipeline per request.
    BM25Retriever gets the live db session.
    VerificationAgent is created inside pipeline with the same db session.
    """
    bm25 = BM25Retriever(db=db)
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=get_vector_retriever(),
        embedding_service=get_embedding_service(),
        reranker=get_reranker(),
    )
    pipeline = AgentPipeline(
        retriever=hybrid,
        llm_client=get_llm_client(),
        db=db,
    )
    yield pipeline
