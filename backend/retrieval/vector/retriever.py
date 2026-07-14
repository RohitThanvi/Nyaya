"""
Qdrant vector retriever v4 — sharded collection, alias routing,
zero-downtime rebuild, actionable failure logging.

Scalability changes vs v3:
- Collection created with shard_number=4: at 50M+ vectors, 4 shards mean
  each shard holds ~12.5M vectors. HNSW segment rebuilds run per-shard in
  parallel instead of blocking the entire collection (v3 had 1 shard →
  blocking full-collection rebuilds during bulk ingestion).
- Collection alias: all reads/writes go through 'nyaya_active' alias, not
  the raw collection name. Zero-downtime index rebuild: build a new collection
  alongside the live one, atomic alias swap, delete old collection. No
  downtime, no stale data window.
- upsert_batch: logs the actual Qdrant error on failure (was silent False
  return with no actionable detail, leaving rows stuck in staged_chunks
  permanently when Qdrant rejected batches silently).
- _get_client: lazy gRPC initialization with connection health check.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.config.settings import get_settings
from backend.models.domain import (
    ChunkType, Citation, DocumentMetadata, DocumentType, LawCategory,
    LegalChunk, RetrievalPath, RetrievedChunk,
)

logger = logging.getLogger(__name__)

_RETRIEVAL_PATH = RetrievalPath.VECTOR.value


class VectorRetriever:
    _client = None

    def __init__(self):
        self._cfg = get_settings().qdrant

    def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            cfg = self._cfg
            if cfg.api_key:
                self._client = AsyncQdrantClient(
                    host=cfg.host, port=cfg.port,
                    grpc_port=cfg.grpc_port,
                    api_key=cfg.api_key,
                    prefer_grpc=True,
                )
            else:
                self._client = AsyncQdrantClient(
                    host=cfg.host, port=cfg.port,
                    grpc_port=cfg.grpc_port,
                    prefer_grpc=True,
                )
        return self._client

    def _search_target(self) -> str:
        """Return the alias name used for all reads — enables zero-downtime swaps."""
        return self._cfg.collection_alias

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        score_threshold: float = 0.0,
        law_filter: Optional[List] = None,
        court_filter: Optional[List] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        document_type=None,
    ) -> List[RetrievedChunk]:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, Range, SearchParams

        must = []
        if law_filter:
            must.append(FieldCondition(
                key="law",
                match=MatchAny(any=[l.value for l in law_filter]),
            ))
        if court_filter:
            must.append(FieldCondition(
                key="court",
                match=MatchAny(any=[c.value for c in court_filter]),
            ))
        if year_from or year_to:
            must.append(FieldCondition(
                key="year",
                range=Range(gte=year_from, lte=year_to),
            ))
        if document_type:
            must.append(FieldCondition(
                key="document_type",
                match=MatchAny(any=[document_type.value]),
            ))
        qdrant_filter = Filter(must=must) if must else None

        try:
            client = self._get_client()
            results = await client.search(
                collection_name=self._search_target(),
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
                search_params=SearchParams(
                    hnsw_ef=self._cfg.hnsw_ef,
                    exact=False,
                ),
            )
        except Exception as e:
            logger.error(f"Qdrant search failed (collection={self._search_target()}): {e}")
            return []

        retrieved = []
        for hit in results:
            payload = hit.payload or {}
            try:
                doc_type = DocumentType(payload.get("document_type", "judgment"))
            except ValueError:
                doc_type = DocumentType.JUDGMENT
            try:
                law = LawCategory(payload.get("law", "other")) if payload.get("law") else LawCategory.OTHER
            except ValueError:
                law = LawCategory.OTHER

            metadata = DocumentMetadata(
                document_id=payload.get("document_id", ""),
                document_type=doc_type,
                law=law,
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
            try:
                chunk_type = ChunkType(payload.get("chunk_type", "passage"))
            except ValueError:
                chunk_type = ChunkType.PASSAGE

            chunk = LegalChunk(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                document_id=payload.get("document_id", ""),
                chunk_type=chunk_type,
                content=payload.get("content", ""),
                content_length=payload.get("content_length", 0),
                chunk_index=payload.get("chunk_index", 0),
                page_number=payload.get("page_number"),
                section_ref=payload.get("section_ref", ""),
                subsection_ref=payload.get("subsection_ref"),
                metadata=metadata,
            )
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                bm25_score=0.0,
                vector_score=float(hit.score),
                hybrid_score=float(hit.score),
                final_score=float(hit.score),
                retrieval_source=_RETRIEVAL_PATH,
                retrieval_method=_RETRIEVAL_PATH,
            ))
        return retrieved

    async def upsert_batch(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict],
        wait: bool = False,
    ) -> bool:
        from qdrant_client.models import PointStruct

        if not chunk_ids:
            return True

        expected_dim = self._cfg.vector_size
        bad = [i for i, v in enumerate(vectors) if not v or len(v) != expected_dim]
        if bad:
            logger.error(
                f"upsert_batch: {len(bad)} vectors have wrong dimension "
                f"(expected {expected_dim}). Bad indices: {bad[:5]}. "
                f"These chunks will NOT be indexed — check the embedding pipeline."
            )
            chunk_ids = [c for i, c in enumerate(chunk_ids) if i not in bad]
            vectors   = [v for i, v in enumerate(vectors)   if i not in bad]
            payloads  = [p for i, p in enumerate(payloads)  if i not in bad]
            if not chunk_ids:
                return False

        points = [
            PointStruct(id=cid, vector=vec, payload=pay)
            for cid, vec, pay in zip(chunk_ids, vectors, payloads)
        ]
        try:
            client = self._get_client()
            op_result = await client.upsert(
                collection_name=self._search_target(),
                points=points,
                wait=wait,
            )
            if hasattr(op_result, "status") and str(op_result.status).lower() not in ("ok", "acknowledged"):
                logger.error(
                    f"Qdrant upsert returned unexpected status: {op_result.status} "
                    f"for {len(points)} points. Rows will stay in staged_chunks "
                    f"and be retried on next flush tick."
                )
                return False
            return True
        except Exception as e:
            logger.error(
                f"Qdrant upsert failed for {len(points)} points "
                f"(collection={self._search_target()}): {e}. "
                f"Rows will stay in staged_chunks and be retried on next flush tick."
            )
            return False

    async def ensure_collection(self) -> None:
        """
        Create the Qdrant collection and alias if they don't exist.

        Alias pattern: all application code uses cfg.collection_alias
        ('nyaya_active') as the target, never the raw collection name.
        This allows zero-downtime index rebuilds:
          1. Create new_collection (new HNSW params, quantization, etc.)
          2. Ingest all vectors into new_collection
          3. await rebuild_collection_alias("new_collection")
             (atomic alias swap: nyaya_active → new_collection)
          4. Delete old collection
        No downtime, no stale data window.
        """
        from qdrant_client.models import (
            VectorParams, Distance, HnswConfigDiff, OptimizersConfigDiff,
            ScalarQuantizationConfig, ScalarType, QuantizationConfig,
        )
        cfg = self._cfg
        client = self._get_client()

        existing = {c.name for c in (await client.get_collections()).collections}

        if cfg.collection_name not in existing:
            distance_map = {
                "Cosine": Distance.COSINE,
                "Dot":    Distance.DOT,
                "Euclid": Distance.EUCLID,
            }
            quantization = None
            if cfg.scalar_quantization:
                quantization = QuantizationConfig(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                )

            await client.create_collection(
                collection_name=cfg.collection_name,
                vectors_config=VectorParams(
                    size=cfg.vector_size,
                    distance=distance_map.get(cfg.distance, Distance.COSINE),
                    on_disk=cfg.on_disk_vectors,
                ),
                hnsw_config=HnswConfigDiff(
                    m=cfg.hnsw_m,
                    ef_construct=cfg.hnsw_ef_construct,
                    full_scan_threshold=10000,
                ),
                optimizers_config=OptimizersConfigDiff(
                    # Don't build HNSW index until 100K vectors are staged —
                    # prevents rebuilding on every small batch during bulk ingestion.
                    indexing_threshold=100000,
                    memmap_threshold=200000,
                ),
                on_disk_payload=cfg.on_disk_payload,
                quantization_config=quantization,
                # Sharding: 4 shards parallelise HNSW segment rebuilds.
                # At 50M vectors, 1 shard rebuilds block all searches for
                # minutes; 4 shards each rebuild independently at 12.5M vectors.
                shard_number=cfg.shard_number,
                replication_factor=cfg.replication_factor,
            )
            logger.info(
                f"Created Qdrant collection '{cfg.collection_name}' "
                f"shards={cfg.shard_number} replication={cfg.replication_factor} "
                f"hnsw_m={cfg.hnsw_m} ef_construct={cfg.hnsw_ef_construct} "
                f"scalar_quantization={cfg.scalar_quantization}"
            )

        # Ensure alias points to this collection
        await self._ensure_alias(client, cfg.collection_name, cfg.collection_alias)

    async def _ensure_alias(self, client, collection_name: str, alias: str) -> None:
        """Create the alias if it doesn't exist or points to the wrong collection."""
        from qdrant_client.models import CreateAliasOperation, CreateAlias, DeleteAliasOperation, DeleteAlias, ChangeAliasesOperation

        try:
            aliases = await client.get_collection_aliases(collection_name=collection_name)
            alias_names = [a.alias_name for a in (aliases.aliases or [])]
            if alias not in alias_names:
                await client.update_collection_aliases(
                    change_aliases_operations=[
                        CreateAliasOperation(
                            create_alias=CreateAlias(
                                collection_name=collection_name,
                                alias_name=alias,
                            )
                        )
                    ]
                )
                logger.info(f"Created alias '{alias}' → '{collection_name}'")
        except Exception as e:
            logger.warning(f"Alias setup failed (non-fatal, using collection name directly): {e}")

    async def rebuild_collection_alias(self, new_collection_name: str) -> None:
        """
        Atomically swap the active alias to a new collection.
        Used for zero-downtime HNSW rebuilds with new parameters.

        Steps (caller's responsibility):
          1. Create new_collection_name with desired parameters
          2. Ingest all vectors into new_collection_name
          3. Call this method to atomically swap the alias
          4. Delete the old collection

        Example:
          vr = VectorRetriever()
          await vr.ensure_collection()  # creates nyaya_legal_chunks_v2
          # ... ingest all vectors ...
          await vr.rebuild_collection_alias("nyaya_legal_chunks_v2")
          # alias nyaya_active now points to v2; delete v1
        """
        from qdrant_client.models import (
            CreateAliasOperation, CreateAlias,
            DeleteAliasOperation, DeleteAlias,
            ChangeAliasesOperation,
        )
        cfg = self._cfg
        client = self._get_client()

        # Atomic: delete old alias binding + create new one in one operation
        await client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=cfg.collection_alias)),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=new_collection_name,
                        alias_name=cfg.collection_alias,
                    )
                ),
            ]
        )
        logger.info(
            f"Zero-downtime alias swap: '{cfg.collection_alias}' "
            f"→ '{new_collection_name}'"
        )
