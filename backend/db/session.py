"""
Async database engine and session management.
Uses SQLAlchemy 2.0 async API with connection pooling.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from backend.config.settings import get_settings
from backend.db.orm_models import Base

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        db = settings.db

        pool_class = NullPool if settings.app.environment == "testing" else AsyncAdaptedQueuePool

        _engine = create_async_engine(
            db.async_url,
            echo=db.echo,
            pool_size=db.pool_size if pool_class != NullPool else None,
            max_overflow=db.max_overflow if pool_class != NullPool else None,
            pool_timeout=db.pool_timeout if pool_class != NullPool else None,
            pool_recycle=db.pool_recycle if pool_class != NullPool else None,
            pool_pre_ping=True,
            poolclass=pool_class,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager that provides a DB session with automatic rollback on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for DB sessions."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables idempotently. Safe to run multiple times."""
    engine = get_engine()
    async with engine.begin() as conn:
        # Create tables only (checkfirst skips existing tables)
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)
        # Create indexes with IF NOT EXISTS to avoid duplicate errors
        index_statements = [
            "CREATE INDEX IF NOT EXISTS ix_users_email ON users (email)",
            "CREATE INDEX IF NOT EXISTS ix_users_email_active ON users (email, is_active)",
            "CREATE INDEX IF NOT EXISTS ix_documents_law_year ON documents (law, year)",
            "CREATE INDEX IF NOT EXISTS ix_documents_court_year ON documents (court, year)",
            "CREATE INDEX IF NOT EXISTS ix_documents_citation ON documents (citation)",
            "CREATE INDEX IF NOT EXISTS ix_documents_type_law ON documents (document_type, law)",
            "CREATE INDEX IF NOT EXISTS ix_chunks_document_index ON chunks (document_id, chunk_index)",
            "CREATE INDEX IF NOT EXISTS ix_chunks_section_ref ON chunks (section_ref)",
            "CREATE INDEX IF NOT EXISTS ix_chunks_qdrant_id ON chunks (qdrant_id)",
            "CREATE INDEX IF NOT EXISTS ix_chunks_type ON chunks (chunk_type)",
            "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id ON chat_sessions (user_id)",
            "CREATE INDEX IF NOT EXISTS ix_chat_messages_session ON chat_messages (session_id)",
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_user_action ON audit_logs (user_id, action)",
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_created_at ON audit_logs (created_at)",
        ]
        for stmt in index_statements:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass
    logger.info("Database tables initialized")

async def close_db() -> None:
    """Dispose engine connections. Run at shutdown."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
    logger.info("Database connections closed")

