"""
NyayaAI FastAPI application v2.

Changes from v1:
- _get_model imported correctly from embeddings.service
- SlowAPI rate limits wired per route (not just middleware)
- AuditLog middleware logs user_id from JWT when present
- Startup: ensures Qdrant collection exists before accepting traffic
- GZip compression threshold lowered to 500 bytes for legal JSON responses
- /api/v1/upload/chunked/* routes registered
- DELETE /api/v1/chat/sessions/{id} registered
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from backend.api.routes import admin, auth, chat, debug, documents, drafting, health, search, upload
from backend.config.settings import get_settings
from backend.db.session import check_db_connection
from backend.embeddings.service import _get_model

logger = logging.getLogger(__name__)
_cfg = get_settings()


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NyayaAI starting up...")

    # ── Schema init ────────────────────────────────────────────────────────
    # Apply ALL SQL migrations in order on every startup (idempotent).
    # Previously only 0001_initial.sql ran — meaning partition pruning (0003),
    # redundant trigger removal (0002), and processing flag (0004) never
    # executed on any real deployment. Now runs every *.sql file in
    # numeric order. Python Alembic migrations (.py) must still be run
    # manually via `alembic upgrade head` for schema changes that need
    # Python logic; pure-SQL idempotent migrations run here automatically.
    try:
        from backend.db.session import async_engine
        import pathlib
        migrations_dir = pathlib.Path(__file__).parent / "db" / "migrations"
        sql_files = sorted(migrations_dir.glob("*.sql"))
        if sql_files:
            async with async_engine.begin() as conn:
                # SQLAlchemy's conn.execute(text(sql)) always goes through
                # asyncpg's extended/prepared-statement protocol, which
                # rejects multiple commands in a single call ("cannot insert
                # multiple commands into a prepared statement") -- these
                # migration files are full multi-statement scripts. Naive
                # semicolon-splitting isn't safe either: the plpgsql
                # function bodies below contain their own semicolons inside
                # $$ ... $$ blocks. Dropping to the raw asyncpg connection
                # and calling .execute() with no bind params uses the
                # simple query protocol instead (same as `psql -f`), which
                # handles multi-statement scripts correctly.
                raw = await conn.get_raw_connection()
                asyncpg_conn = raw.driver_connection
                for sql_path in sql_files:
                    sql = sql_path.read_text()
                    try:
                        await asyncpg_conn.execute(sql)
                        logger.info(f"Migration applied: {sql_path.name}")
                    except Exception as e:
                        # Log but continue — most errors are "table already exists"
                        # which is fine since all DDL uses IF NOT EXISTS
                        logger.debug(f"Migration {sql_path.name}: {e} (likely already applied)")
    except Exception as e:
        logger.warning(f"Schema init warning (non-fatal): {e}")

    # ── Alembic migrations (0002+) ───────────────────────────────────────────
    # The SQL glob above only ever applies 0001_initial.sql — 0002-0005 are
    # Python/Alembic revisions and, until this call, alembic upgrade head
    # was never actually run anywhere (env.py didn't exist at all). That's
    # why the partitioned chunks table / law column from 0003 never showed
    # up on any real deployment. Runs synchronously via a thread since
    # Alembic's own engine handling is sync (psycopg2), not asyncpg.
    try:
        import asyncio
        from alembic import command
        from alembic.config import Config as AlembicConfig
        import pathlib as _pathlib

        repo_root = _pathlib.Path(__file__).resolve().parents[1]
        alembic_cfg = AlembicConfig(str(repo_root / "alembic.ini"))
        alembic_cfg.set_main_option(
            "script_location", str(repo_root / "backend" / "db" / "migrations")
        )
        await asyncio.to_thread(command.upgrade, alembic_cfg, "head")
        logger.info("Alembic migrations applied (head)")
    except Exception as e:
        logger.warning(f"Alembic migration warning (non-fatal): {e}")

    # ── Warm up embedding model ─────────────────────────────────────────────
    try:
        _get_model()
        logger.info("Embedding model warmed up")
    except Exception as e:
        logger.warning(f"Embedding model warmup skipped (non-fatal): {e}")

    # ── Ensure Qdrant collection ────────────────────────────────────────────
    try:
        from backend.retrieval.vector.retriever import VectorRetriever
        vr = VectorRetriever()
        await vr.ensure_collection()
    except Exception as e:
        logger.warning(f"Qdrant collection check skipped (Qdrant may still be starting): {e}")

    # ── DB connectivity check ───────────────────────────────────────────────
    db_ok = await check_db_connection()
    if not db_ok:
        logger.error("Database connection failed at startup — queries will fail")
    else:
        logger.info("Database connection OK")

    logger.info(f"NyayaAI v{_cfg.app.app_version} ready [env={_cfg.app.environment}]")
    yield

    logger.info("NyayaAI shutting down")


# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=_cfg.app.app_name,
        version=_cfg.app.app_version,
        description="AI-powered Indian legal research and drafting platform",
        docs_url="/docs" if not _cfg.app.is_production else None,
        redoc_url="/redoc" if not _cfg.app.is_production else None,
        lifespan=lifespan,
        redirect_slashes=False,   # <-- ADD THIS LINE
    )

    # ── Middleware stack (order matters — outermost added last) ───────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cfg.app.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=500)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Request timing + audit middleware
    @app.middleware("http")
    async def timing_and_audit(request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.0f}"

        # Audit log for mutating endpoints
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            path = request.url.path
            user_id = None
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                try:
                    from jose import jwt as _jwt
                    token = auth_header[7:]
                    payload = _jwt.decode(
                        token, _cfg.auth.secret_key,
                        algorithms=[_cfg.auth.algorithm],
                        options={"verify_exp": False},
                    )
                    user_id = payload.get("sub")
                except Exception:
                    pass
            logger.info(
                f"AUDIT | {request.method} {path} | user={user_id} "
                f"| status={response.status_code} | {elapsed_ms:.0f}ms"
            )

        return response

    # ── Routers ───────────────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(health.router,     prefix=prefix)
    app.include_router(auth.router,       prefix=prefix)
    app.include_router(search.router,     prefix=prefix)
    app.include_router(chat.router,       prefix=prefix)
    app.include_router(upload.router,     prefix=prefix)
    app.include_router(documents.router,  prefix=prefix)
    app.include_router(drafting.router,   prefix=prefix)
    app.include_router(debug.router,      prefix=prefix)
    app.include_router(admin.router,      prefix=prefix)

    @app.get("/")
    async def root():
        return {
            "service": _cfg.app.app_name,
            "version": _cfg.app.app_version,
            "environment": _cfg.app.environment,
            "status": "running",
        }

    return app


app = create_app()
