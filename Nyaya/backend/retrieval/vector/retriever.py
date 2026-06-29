"""
Qdrant vector retriever — ANN path (conditional, Path 3).

on_disk_vectors is enabled via settings when corpus exceeds ~10M chunks.
Payload filters pushed down to Qdrant before ANN search (not post-filtered)
to avoid scoring all vectors then discarding — critical at TB scale.
"""
import logging
from typing import Dict, List, Optional

from backend.config.settings import get_settings
from backend.models.domain import (
    ChunkType, CourtType, DocumentMetadata, DocumentType,
    LawCategory, LegalChunk, RetrievedChunk, RetrievalPath,
)

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self):
        self._settings = get_settings()
        self._cfg = self._settings.qdrant
        self._client = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            kwargs: Dict = {
                "host": self._cfg.host,
                "port": self._cfg.port,
                "grpc_port": self._cfg.grpc_port,
                "prefer_grpc": True,   # gRPC is faster for bulk search
            }
            if self._cfg.api_key:
                kwargs["api_key"] = self._cfg.api_key
            self._client = AsyncQdrantClient(**kwargs)
        return self._client

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 30,
        law_filter: Optional[List[LawCategory]] = None,
        court_filter: Optional[List[CourtType]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_id: Optional[str] = None,
        score_threshold: float = 0.25,
    ) -> List[RetrievedChunk]:
        """
        Qdrant HNSW search with payload pre-filtering.
        All filters applied server-side before scoring — zero wasted compute.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny, Range

        conditions = []
        if law_filter:
            conditions.append(FieldCondition(
                key="law",
                match=MatchAny(any=[l.value for l in law_filter]),
            ))
        if court_filter:
            conditions.append(FieldCondition(
                key="court",
                match=MatchAny(any=[c.value for c in court_filter]),
            ))
        if year_from or year_to:
            conditions.append(FieldCondition(
                key="year",
                range=Range(gte=year_from, lte=year_to),
            ))
        if document_id:
            conditions.append(FieldCondition(
                key="document_id",
                match=MatchAny(any=[document_id]),
            ))

        qdrant_filter = Filter(must=conditions) if conditions else None

        try:
            client = self._get_client()
            results = await client.search(
                collection_name=self._cfg.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

        retrieved = []
        for hit in results:
            payload = hit.payload or {}
            metadata = DocumentMetadata(
                document_id=payload.get("document_id", ""),
                document_type=DocumentType(payload.get("document_type", "judgment")),
                law=LawCategory(payload["law"]) if payload.get("law") else None,
                court=CourtType(payload["court"]) if payload.get("court") else None,
                court_name=payload.get("court_name"),
                case_number=payload.get("case_number"),
                citation=payload.get("citation"),
                year=payload.get("year"),
                topic=payload.get("topic"),
                keywords=payload.get("keywords", []),
                source_url=payload.get("source_url"),
                is_landmark=payload.get("is_landmark", False),
                language=payload.get("language", "en"),
            )
            chunk = LegalChunk(
                chunk_id=str(hit.id),
                document_id=payload.get("document_id", ""),
                chunk_type=ChunkType(payload.get("chunk_type", "passage")),
                content=payload.get("content", ""),
                content_length=payload.get("content_length", 0),
                chunk_index=payload.get("chunk_index", 0),
                page_number=payload.get("page_number"),
                section_ref=payload.get("section_ref"),
                subsection_ref=payload.get("subsection_ref"),
                metadata=metadata,
            )
            score = float(hit.score)
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                vector_score=score,
                hybrid_score=score,
                final_score=score,
                retrieval_source=RetrievalPath.VECTOR.value,
                retrieval_method=RetrievalPath.VECTOR.value,
            ))

        logger.debug(f"Vector search → {len(retrieved)} results (threshold={score_threshold})")
        return retrieved

    async def upsert_batch(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict],
    ) -> bool:
        """Batch upsert for ingestion flush worker."""
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=cid, vector=vec, payload=pay)
            for cid, vec, pay in zip(chunk_ids, vectors, payloads)
        ]
        try:
            client = self._get_client()
            await client.upsert(
                collection_name=self._cfg.collection_name,
                points=points,
                wait=True,
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant batch upsert failed: {e}")
            return False

    async def ensure_collection(self):
        """Create Qdrant collection if it doesn't exist, with on-disk config."""
        from qdrant_client.models import (
            VectorParams, Distance, HnswConfigDiff, OptimizersConfigDiff
        )
        cfg = self._cfg
        client = self._get_client()
        existing = [c.name for c in (await client.get_collections()).collections]
        if cfg.collection_name in existing:
            logger.info(f"Qdrant collection '{cfg.collection_name}' already exists")
            return

        distance_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
        await client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config=VectorParams(
                size=cfg.vector_size,
                distance=distance_map.get(cfg.distance, Distance.COSINE),
                on_disk=cfg.on_disk_vectors,
            ),
            hnsw_config=HnswConfigDiff(m=cfg.hnsw_m, ef_construct=cfg.hnsw_ef_construct),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=10000),
            on_disk_payload=cfg.on_disk_payload,
        )
        logger.info(f"Created Qdrant collection '{cfg.collection_name}' "
                    f"(on_disk_vectors={cfg.on_disk_vectors})")
