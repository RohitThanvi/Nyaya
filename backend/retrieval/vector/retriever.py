"""
Qdrant ANN vector retriever with HNSW indexing.
Handles collection setup, upsert, and filtered ANN search.
"""
import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    Range,
    SearchRequest as QdrantSearchRequest,
    VectorParams,
    models as qmodels,
)

from backend.config.settings import get_settings
from backend.models.domain import (
    ChunkType, CourtType, DocumentMetadata, DocumentType,
    LawCategory, LegalChunk, RetrievedChunk
)

logger = logging.getLogger(__name__)

_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings().qdrant
        _qdrant_client = AsyncQdrantClient(
            host=settings.host,
            port=settings.port,
            api_key=settings.api_key,
            timeout=30,
        )
    return _qdrant_client


class VectorRetriever:
    """
    Qdrant-based ANN retrieval.

    HNSW parameters:
    - m=16: number of connections per layer (higher = better recall, more RAM)
    - ef_construct=200: build-time search breadth (higher = better index quality)
    - ef=128: query-time search breadth (higher = better recall, slower)

    Payload is stored on-disk for memory efficiency at scale.
    """

    def __init__(self):
        self._settings = get_settings().qdrant
        self._client = get_qdrant_client()

    async def ensure_collection(self) -> None:
        """Create collection with HNSW config if it doesn't exist."""
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._settings.collection_name in names:
            return

        await self._client.create_collection(
            collection_name=self._settings.collection_name,
            vectors_config=VectorParams(
                size=self._settings.vector_size,
                distance=Distance.COSINE,
                on_disk=False,  # keep vectors in RAM for speed
            ),
            hnsw_config=HnswConfigDiff(
                m=self._settings.hnsw_m,
                ef_construct=self._settings.hnsw_ef_construct,
                full_scan_threshold=10000,
                on_disk=False,
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=4,  # parallel segment optimization
                memmap_threshold=50000,
            ),
            on_disk_payload=self._settings.on_disk_payload,
        )

        # Create payload indexes for efficient filtered search
        await self._client.create_payload_index(
            collection_name=self._settings.collection_name,
            field_name="law",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._settings.collection_name,
            field_name="court",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._settings.collection_name,
            field_name="year",
            field_schema=PayloadSchemaType.INTEGER,
        )
        await self._client.create_payload_index(
            collection_name=self._settings.collection_name,
            field_name="document_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._settings.collection_name,
            field_name="chunk_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._settings.collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        logger.info(f"Qdrant collection '{self._settings.collection_name}' created")

    def _build_filter(
        self,
        law_filter: Optional[List[LawCategory]] = None,
        court_filter: Optional[List[CourtType]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_type: Optional[DocumentType] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Optional[Filter]:
        """Build Qdrant filter from metadata constraints."""
        must = []

        if law_filter:
            must.append(
                FieldCondition(key="law", match=MatchAny(any=[l.value for l in law_filter]))
            )
        if court_filter:
            must.append(
                FieldCondition(key="court", match=MatchAny(any=[c.value for c in court_filter]))
            )
        if year_from or year_to:
            must.append(
                FieldCondition(
                    key="year",
                    range=Range(
                        gte=year_from if year_from else None,
                        lte=year_to if year_to else None,
                    ),
                )
            )
        if document_type:
            must.append(
                FieldCondition(key="document_type", match=MatchValue(value=document_type.value))
            )
        if document_ids:
            must.append(
                FieldCondition(key="document_id", match=MatchAny(any=document_ids))
            )

        if not must:
            return None
        return Filter(must=must)

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        law_filter: Optional[List[LawCategory]] = None,
        court_filter: Optional[List[CourtType]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_type: Optional[DocumentType] = None,
        document_ids: Optional[List[str]] = None,
        score_threshold: float = 0.3,
    ) -> List[RetrievedChunk]:
        """ANN search with payload filtering."""
        qdrant_filter = self._build_filter(
            law_filter, court_filter, year_from, year_to, document_type, document_ids
        )

        try:
            results = await self._client.search(
                collection_name=self._settings.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,  # don't return vectors to save bandwidth
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

        retrieved = []
        for hit in results:
            payload = hit.payload or {}
            chunk = self._payload_to_chunk(str(hit.id), payload)
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    vector_score=float(hit.score),
                    retrieval_source="vector",
                )
            )

        logger.debug(f"Vector search returned {len(retrieved)} results")
        return retrieved

    def _payload_to_chunk(self, point_id: str, payload: Dict[str, Any]) -> LegalChunk:
        """Reconstruct LegalChunk from Qdrant payload."""
        metadata = DocumentMetadata(
            document_id=payload.get("document_id", ""),
            document_type=DocumentType(payload.get("document_type", "judgment")),
            law=LawCategory(payload["law"]) if payload.get("law") else None,
            court=CourtType(payload["court"]) if payload.get("court") else None,
            court_name=payload.get("court_name"),
            case_number=payload.get("case_number"),
            citation=payload.get("citation"),
            year=payload.get("year"),
            section=payload.get("section"),
            topic=payload.get("topic"),
            keywords=payload.get("keywords", []),
            source_url=payload.get("source_url"),
            is_landmark=payload.get("is_landmark", False),
            language=payload.get("language", "en"),
        )
        return LegalChunk(
            chunk_id=payload.get("chunk_id", point_id),
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

    async def upsert_chunks(
        self, chunks: List[LegalChunk], embeddings: List[List[float]]
    ) -> int:
        """Batch upsert chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")

        await self.ensure_collection()

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_type": chunk.chunk_type.value,
                "content": chunk.content,
                "content_length": chunk.content_length,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_ref": chunk.section_ref,
                "subsection_ref": chunk.subsection_ref,
                "document_type": chunk.metadata.document_type.value,
                "law": chunk.metadata.law.value if chunk.metadata.law else None,
                "court": chunk.metadata.court.value if chunk.metadata.court else None,
                "court_name": chunk.metadata.court_name,
                "case_number": chunk.metadata.case_number,
                "citation": chunk.metadata.citation,
                "year": chunk.metadata.year,
                "section": chunk.metadata.section,
                "topic": chunk.metadata.topic,
                "keywords": chunk.metadata.keywords,
                "source_url": chunk.metadata.source_url,
                "is_landmark": chunk.metadata.is_landmark,
                "language": chunk.metadata.language,
            }
            # Remove None values to save storage
            payload = {k: v for k, v in payload.items() if v is not None}

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),  # stable point IDs
                    vector=embedding,
                    payload=payload,
                )
            )

        # Batch upsert in chunks of 100
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            await self._client.upsert(
                collection_name=self._settings.collection_name,
                points=batch,
                wait=True,
            )
            total_upserted += len(batch)

        logger.info(f"Upserted {total_upserted} vectors to Qdrant")
        return total_upserted

    async def delete_document(self, document_id: str) -> None:
        """Delete all vectors for a document."""
        await self._client.delete(
            collection_name=self._settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )

    async def get_collection_info(self) -> Dict[str, Any]:
        """Return collection stats."""
        info = await self._client.get_collection(self._settings.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
            "optimizer_status": str(info.optimizer_status),
        }
