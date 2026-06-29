"""
SQLAlchemy ORM models v2.
Adds: staged_chunks table, source_url + page_number on chunks,
      GIN index on content_tsv, original_filename on documents.
"""
from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index, Integer,
    String, Text, UniqueConstraint, ForeignKey, JSON,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
import uuid


class Base(DeclarativeBase):
    pass


class DocumentORM(Base):
    __tablename__ = "documents"

    document_id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_type     = Column(String(50), nullable=False, index=True)
    law               = Column(String(50), index=True)
    court             = Column(String(50), index=True)
    court_name        = Column(String(200))
    case_number       = Column(String(200))
    citation          = Column(String(200), index=True)
    year              = Column(Integer, index=True)
    date_decided      = Column(DateTime)
    bench             = Column(ARRAY(String))
    parties           = Column(JSONB)
    topic             = Column(String(500))
    keywords          = Column(ARRAY(String), default=list)
    source_url        = Column(String(2000))    # canonical public URL
    original_filename = Column(String(500))     # for resume/dedup on bulk ingest
    is_landmark       = Column(Boolean, default=False)
    language          = Column(String(10), default="en")
    pages             = Column(Integer, default=0)
    uploaded_by       = Column(String(100))
    created_at        = Column(DateTime, server_default=func.now())
    updated_at        = Column(DateTime, server_default=func.now(), onupdate=func.now())

    chunks            = relationship("ChunkORM", back_populates="document", cascade="all, delete-orphan")
    drafts            = relationship("LegalDraftORM", back_populates="document")

    __table_args__ = (
        Index("ix_documents_law_year", "law", "year"),
        Index("ix_documents_court_year", "court", "year"),
    )


class ChunkORM(Base):
    __tablename__ = "chunks"

    chunk_id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id    = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    chunk_type     = Column(String(50), nullable=False, default="passage")
    content        = Column(Text, nullable=False)
    content_length = Column(Integer, default=0)
    chunk_index    = Column(Integer, nullable=False, default=0)
    page_number    = Column(Integer)         # physical page in source PDF
    section_ref    = Column(String(50), index=True)
    subsection_ref = Column(String(100))
    content_tsv    = Column(TSVECTOR)       # populated by to_tsvector() at insert

    document       = relationship("DocumentORM", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
        Index("ix_chunks_section_ref", "section_ref"),
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),   # GIN for FTS
    )


class StagedChunkORM(Base):
    """
    Staging table for the distributed ingestion pipeline.
    Chunks land here after embedding; flushed to Qdrant in batches.
    """
    __tablename__ = "staged_chunks"

    chunk_id    = Column(UUID(as_uuid=True), primary_key=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    # embedding stored as JSONB array (pgvector extension recommended for production)
    embedding   = Column(JSONB, nullable=False)
    metadata    = Column(JSONB, nullable=False, default=dict)
    indexed     = Column(Boolean, default=False, nullable=False)
    created_at  = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_staged_chunks_indexed", "indexed", "document_id"),
    )


class UserORM(Base):
    __tablename__ = "users"

    user_id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    full_name       = Column(String(200), nullable=False)
    role            = Column(String(50), nullable=False, default="researcher")
    hashed_password = Column(String(255), nullable=False)
    is_active       = Column(Boolean, default=True)
    bar_enrollment  = Column(String(100))
    created_at      = Column(DateTime, server_default=func.now())
    last_login      = Column(DateTime)

    chat_sessions   = relationship("ChatSessionORM", back_populates="user")


class ChatSessionORM(Base):
    __tablename__ = "chat_sessions"

    session_id  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    title       = Column(String(500))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="SET NULL"))
    law_filter  = Column(ARRAY(String))
    created_at  = Column(DateTime, server_default=func.now())
    updated_at  = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user        = relationship("UserORM", back_populates="chat_sessions")
    messages    = relationship("ChatMessageORM", back_populates="session", cascade="all, delete-orphan")


class ChatMessageORM(Base):
    __tablename__ = "chat_messages"

    message_id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id          = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False)
    role                = Column(String(20), nullable=False)
    content             = Column(Text, nullable=False)
    citations           = Column(JSONB, default=list)    # List[Citation] serialised
    hallucination_flags = Column(ARRAY(String), default=list)
    confidence          = Column(Float)
    intent              = Column(String(50))
    latency_ms          = Column(Float)
    created_at          = Column(DateTime, server_default=func.now())

    session             = relationship("ChatSessionORM", back_populates="messages")


class LegalDraftORM(Base):
    __tablename__ = "legal_drafts"

    draft_id    = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="SET NULL"))
    draft_type  = Column(String(100), nullable=False)
    content     = Column(Text, nullable=False)
    parties     = Column(JSONB, default=dict)
    court       = Column(String(200))
    facts       = Column(Text)
    created_at  = Column(DateTime, server_default=func.now())
    updated_at  = Column(DateTime, server_default=func.now(), onupdate=func.now())

    document    = relationship("DocumentORM", back_populates="drafts")


class AuditLogORM(Base):
    __tablename__ = "audit_logs"

    log_id      = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(String(100))
    action      = Column(String(100), nullable=False)
    resource    = Column(String(100))
    resource_id = Column(String(200))
    details     = Column(JSONB, default=dict)
    ip_address  = Column(String(50))
    created_at  = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_audit_user_created", "user_id", "created_at"),
    )
