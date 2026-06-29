"""
Database session management.
Provides both async (FastAPI) and sync (Celery workers) sessions.
"""
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
_cfg = get_settings().db

# ── Async engine (FastAPI) ────────────────────────────────────────────────────
async_engine = create_async_engine(
    _cfg.async_url,
    pool_size=_cfg.pool_size,
    max_overflow=_cfg.max_overflow,
    pool_timeout=_cfg.pool_timeout,
    pool_recycle=_cfg.pool_recycle,
    pool_pre_ping=True,
    echo=_cfg.echo,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# ── Sync engine (Celery workers) ──────────────────────────────────────────────
sync_engine = create_engine(
    _cfg.sync_url,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=_cfg.echo,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for use outside FastAPI dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Sync context manager for Celery tasks."""
    session = SyncSessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def check_db_connection() -> bool:
    """Health check for DB connectivity."""
    try:
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return False
