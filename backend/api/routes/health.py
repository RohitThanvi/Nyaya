"""Health check endpoints."""
import time
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.session import get_db
from backend.config.settings import get_settings

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "NyayaAI", "version": get_settings().app.app_version}


@router.get("/health/detailed")
async def health_detailed(db: AsyncSession = Depends(get_db)):
    """Detailed health: checks DB, Qdrant, Redis."""
    components = {}
    t0 = time.perf_counter()

    # PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        components["postgresql"] = {"status": "ok", "latency_ms": round((time.perf_counter()-t0)*1000,1)}
    except Exception as e:
        components["postgresql"] = {"status": "error", "error": str(e)}

    # Qdrant
    try:
        from backend.retrieval.vector.retriever import get_qdrant_client
        t1 = time.perf_counter()
        client = get_qdrant_client()
        await client.get_collections()
        components["qdrant"] = {"status": "ok", "latency_ms": round((time.perf_counter()-t1)*1000,1)}
    except Exception as e:
        components["qdrant"] = {"status": "error", "error": str(e)}

    # Redis
    try:
        from backend.utils.redis_client import get_redis_client
        t2 = time.perf_counter()
        redis = await get_redis_client()
        await redis.ping()
        components["redis"] = {"status": "ok", "latency_ms": round((time.perf_counter()-t2)*1000,1)}
    except Exception as e:
        components["redis"] = {"status": "degraded", "note": "Non-fatal"}

    overall = "ok" if all(v.get("status") == "ok" for k, v in components.items() if k != "redis") else "degraded"
    return {"status": overall, "components": components}
