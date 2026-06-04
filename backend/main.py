"""
NyayaAI FastAPI Application.

Lifespan manages startup/shutdown of:
- Database connections
- Qdrant collection init
- Model warm-up
- Redis connection
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from backend.api.middleware.audit import AuditLogMiddleware
from backend.api.middleware.timing import TimingMiddleware
from backend.api.routes import auth, chat, documents, drafting, health, search, upload
from backend.config.settings import get_settings
from backend.db.session import close_db, init_db
from backend.retrieval.vector.retriever import VectorRetriever
from backend.utils.redis_client import get_redis_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    settings = get_settings()
    logger.info(f"Starting NyayaAI [{settings.app.environment}]")

    # Initialize PostgreSQL
    await init_db()
    logger.info("PostgreSQL ready")

    # Initialize Qdrant collection
    try:
        vr = VectorRetriever()
        await vr.ensure_collection()
        logger.info("Qdrant collection ready")
    except Exception as e:
        logger.warning(f"Qdrant init warning (non-fatal): {e}")

    # Warm up Redis
    try:
        redis = await get_redis_client()
        await redis.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis not available (non-fatal): {e}")

    # Pre-load models in background to warm JIT
    # (models load lazily, but we trigger them at startup)
    try:
        from backend.embeddings.service import _get_model
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_model)
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.warning(f"Model pre-load warning: {e}")

    logger.info("NyayaAI startup complete")
    yield

    # Shutdown
    await close_db()
    logger.info("NyayaAI shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="NyayaAI Legal Research Platform",
        description="Production-grade Indian legal AI with BNS/BNSS/BSA retrieval",
        version=settings.app.app_version,
        docs_url="/api/docs" if not settings.app.is_production else None,
        redoc_url="/api/redoc" if not settings.app.is_production else None,
        lifespan=lifespan,
    )

    # ── Rate Limiter ─────────────────────────────
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # ── CORS ─────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Compression ───────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Custom Middleware ─────────────────────────
    app.add_middleware(TimingMiddleware)
    app.add_middleware(AuditLogMiddleware)

    # ── Routers ───────────────────────────────────
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(drafting.router, prefix="/api/v1", tags=["Drafting"])

    # ── Global Exception Handler ──────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again."},
        )

    return app


app = create_app()
