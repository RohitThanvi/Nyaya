"""
Admin routes — zero-downtime Qdrant rebuild, migration status, queue stats.

These endpoints require admin role and are never exposed publicly.
Mount at /api/v1/admin (already registered in main.py if debug=False is relaxed).
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies.auth import require_admin
from backend.db.session import get_db

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def admin_health(db: AsyncSession = Depends(get_db), _=Depends(require_admin)):
    """Deep health check: DB row counts, Qdrant collection info, Redis queue depths."""
    report = {}

    # DB stats
    try:
        rows = {}
        for table in ("documents", "chunks", "staged_chunks", "chat_sessions", "chat_messages"):
            r = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            rows[table] = r.scalar_one()
        report["db"] = {"status": "ok", "row_counts": rows}
    except Exception as e:
        report["db"] = {"status": "error", "error": str(e)}

    # Qdrant stats
    try:
        from backend.retrieval.vector.retriever import VectorRetriever
        from backend.config.settings import get_settings
        cfg = get_settings().qdrant
        vr = VectorRetriever()
        client = vr._get_client()
        info = await client.get_collection(cfg.collection_alias)
        report["qdrant"] = {
            "status": "ok",
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
            "collection": cfg.collection_alias,
        }
    except Exception as e:
        report["qdrant"] = {"status": "error", "error": str(e)}

    # Redis queue depths
    try:
        from backend.utils.redis_client import get_redis_client
        redis = await get_redis_client()
        queues = ["nyaya_parse", "nyaya_embed_gpu_0", "nyaya_embed_gpu_1",
                  "nyaya_embed_gpu_2", "nyaya_embed_gpu_3", "nyaya_flush", "nyaya_dlq"]
        depths = {}
        for q in queues:
            depths[q] = await redis.llen(q)
        report["celery_queues"] = {"status": "ok", "depths": depths}
    except Exception as e:
        report["celery_queues"] = {"status": "error", "error": str(e)}

    return report


@router.post("/qdrant/rebuild")
async def rebuild_qdrant_collection(
    background_tasks: BackgroundTasks,
    _=Depends(require_admin),
):
    """
    Trigger a zero-downtime Qdrant index rebuild.

    Process:
    1. Creates a new versioned collection alongside the live one
    2. Re-ingests all vectors from staged_chunks into the new collection
    3. Atomically swaps the alias (nyaya_active → new collection)
    4. Deletes the old collection

    No downtime — searches continue against the old collection until
    the atomic swap. Returns immediately; rebuild runs in background.
    Monitor progress via GET /admin/qdrant/rebuild/status.

    Use when: changing HNSW params, updating quantization, or after
    a catastrophic index corruption.
    """
    background_tasks.add_task(_run_rebuild)
    return {"status": "rebuild_started", "message": "Monitor via GET /admin/qdrant/rebuild/status"}


async def _run_rebuild():
    """Background task: full Qdrant index rebuild from chunks + documents."""
    from backend.retrieval.vector.retriever import VectorRetriever
    from backend.embeddings.service import EmbeddingService
    from backend.config.settings import get_settings
    from backend.db.session import get_db_session
    from qdrant_client.models import PointStruct
    import time

    cfg = get_settings().qdrant
    new_name = f"{cfg.collection_name}_rebuild_{int(time.time())}"
    logger.info(f"Zero-downtime rebuild starting: target collection = {new_name}")

    client = None
    try:
        vr = VectorRetriever()
        embedder = EmbeddingService()
        client = vr._get_client()

        # Create new collection with current settings
        from qdrant_client.models import (
            VectorParams, Distance, HnswConfigDiff, OptimizersConfigDiff,
            ScalarQuantizationConfig, ScalarType, ScalarQuantization,
        )
        dist = {"Cosine": Distance.COSINE, "Dot": Distance.DOT}.get(cfg.distance, Distance.COSINE)
        # QuantizationConfig is a typing.Union alias, not a class — see
        # retriever.py's ensure_collection for the same fix and full explanation.
        quant = ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
        ) if cfg.scalar_quantization else None

        await client.create_collection(
            collection_name=new_name,
            vectors_config=VectorParams(size=cfg.vector_size, distance=dist),
            hnsw_config=HnswConfigDiff(m=cfg.hnsw_m, ef_construct=cfg.hnsw_ef_construct),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=100000),
            on_disk_payload=cfg.on_disk_payload,
            quantization_config=quant,
            shard_number=cfg.shard_number,
        )
        logger.info(f"Created new collection: {new_name}")

        # Re-embed every chunk (joined with its document's metadata, same
        # payload shape used by ingest.py's live path) directly into the
        # new collection — NOT via upsert_batch, which always writes to
        # the live alias, not an arbitrary named collection.
        BATCH = 500
        offset = 0
        total_upserted = 0
        async with get_db_session() as db:
            from sqlalchemy import text
            total = (await db.execute(text("SELECT COUNT(*) FROM chunks"))).scalar_one()
            logger.info(f"Rebuild: {total:,} chunks to re-embed")

            while offset < total:
                rows = (await db.execute(text("""
                    SELECT c.chunk_id, c.document_id, c.chunk_type, c.content,
                           c.content_length, c.chunk_index, c.page_number,
                           c.section_ref, c.subsection_ref,
                           d.document_type, d.law, d.court, d.court_name,
                           d.case_number, d.citation, d.year, d.topic,
                           d.keywords, d.source_url
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    ORDER BY c.document_id, c.chunk_index
                    LIMIT :limit OFFSET :offset
                """), {"limit": BATCH, "offset": offset})).fetchall()
                if not rows:
                    break

                texts = [r.content for r in rows]
                vectors = await embedder.embed_batch(texts)

                points = []
                for r, vec in zip(rows, vectors):
                    if not vec or len(vec) != cfg.vector_size:
                        logger.error(
                            f"Rebuild: skipping chunk {r.chunk_id} — bad embedding "
                            f"dimension ({len(vec) if vec else 0}, expected {cfg.vector_size})"
                        )
                        continue
                    points.append(PointStruct(
                        id=str(r.chunk_id),
                        vector=vec,
                        payload={
                            "chunk_id": str(r.chunk_id),
                            "document_id": str(r.document_id),
                            "chunk_type": r.chunk_type,
                            "content": r.content,
                            "content_length": r.content_length,
                            "chunk_index": r.chunk_index,
                            "page_number": r.page_number,
                            "section_ref": r.section_ref,
                            "subsection_ref": r.subsection_ref,
                            "document_type": r.document_type,
                            "law": r.law,
                            "court": r.court,
                            "court_name": r.court_name,
                            "case_number": r.case_number,
                            "citation": r.citation,
                            "year": r.year,
                            "topic": r.topic,
                            "keywords": r.keywords,
                            "source_url": r.source_url,
                        },
                    ))

                if points:
                    await client.upsert(collection_name=new_name, points=points, wait=False)
                    total_upserted += len(points)

                offset += BATCH
                if offset % (BATCH * 20) == 0:
                    logger.info(f"Rebuild progress: {offset:,}/{total:,} chunks processed")

        # SAFETY CHECK — never swap the live alias onto an incomplete or
        # empty collection. A rebuild that silently under-populated the
        # new collection (embedding failures, a crashed batch, etc.) must
        # not take down production search; abort and clean up instead.
        new_info = await client.get_collection(new_name)
        new_count = new_info.points_count or 0
        logger.info(f"Rebuild: {total_upserted:,} points upserted, "
                    f"collection reports {new_count:,} of {total:,} expected")
        if total == 0 or new_count < total * 0.99:
            raise RuntimeError(
                f"Rebuild aborted before alias swap — new collection has "
                f"{new_count:,} points but {total:,} were expected "
                f"(source chunk count may have changed mid-rebuild, or "
                f"embedding failures occurred; see errors above)."
            )

        # Atomic alias swap — only now, with a verified-populated collection.
        await vr.rebuild_collection_alias(new_name)
        logger.info(f"Rebuild complete: alias 'nyaya_active' → '{new_name}'")

    except Exception as e:
        logger.error(f"Qdrant rebuild failed: {e}", exc_info=True)
        if client is not None:
            try:
                await client.delete_collection(new_name)
            except Exception:
                pass


@router.get("/migrations/status")
async def migration_status(db: AsyncSession = Depends(get_db), _=Depends(require_admin)):
    """Check which migrations have been applied."""
    import pathlib
    migrations_dir = pathlib.Path(__file__).parent.parent.parent / "db" / "migrations"

    sql_files = sorted(migrations_dir.glob("*.sql"))
    py_files  = sorted(migrations_dir.glob("0*.py"))

    applied = []
    for sql_path in sql_files:
        applied.append({"file": sql_path.name, "type": "sql", "status": "auto-applied on startup"})
    for py_path in py_files:
        applied.append({"file": py_path.name, "type": "alembic", "status": "manual (run alembic upgrade head)"})

    return {"migrations": applied, "total": len(applied)}
