"""
SQLAlchemy ORM models for PostgreSQL.
Uses async-compatible declarative base.
"""
from datetime import datetime
from email.mime import text
from typing import Optional
import uuid

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, JSON, Enum as SAEnum,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
from sqlalchemy import text


class Base(DeclarativeBase):
    pass


class UserORM(Base):
    __tablename__ = "users"
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="researcher")
    bar_enrollment = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    sessions = relationship("ChatSessionORM", back_populates="user", lazy="selectin")
    audit_logs = relationship("AuditLogORM", back_populates="user", lazy="dynamic")


class DocumentORM(Base):
    __tablename__ = "documents"

    document_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_type = Column(String(50), nullable=False)
    law = Column(String(50), nullable=True)
    court = Column(String(100), nullable=True)
    court_name = Column(String(255), nullable=True)
    case_number = Column(String(255), nullable=True)
    citation = Column(String(255), nullable=True, index=True)
    year = Column(Integer, nullable=True, index=True)
    date_decided = Column(DateTime(timezone=True), nullable=True)
    bench = Column(ARRAY(String), nullable=True)
    parties = Column(JSON, nullable=True)
    section = Column(String(100), nullable=True)
    chapter = Column(String(255), nullable=True)
    topic = Column(String(500), nullable=True)
    keywords = Column(ARRAY(String), nullable=True)
    source_url = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    is_landmark = Column(Boolean, default=False)
    language = Column(String(10), default="en")
    raw_text_path = Column(Text, nullable=True)  # path to raw text file
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    chunks = relationship("ChunkORM", back_populates="document", lazy="dynamic",
                          cascade="all, delete-orphan")


class ChunkORM(Base):
    __tablename__ = "chunks"

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"),
                         nullable=False, index=True)
    chunk_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False, default=0)
    page_number = Column(Integer, nullable=True)
    section_ref = Column(String(255), nullable=True, index=True)
    subsection_ref = Column(String(255), nullable=True)

    # Full-text search vector for BM25-equivalent retrieval
    content_tsv = Column(TSVECTOR, nullable=True)

    # Qdrant vector ID (stored here for cross-referencing)
    qdrant_id = Column(String(36), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("DocumentORM", back_populates="chunks")


class ChatSessionORM(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"),
                     nullable=False, index=True)
    title = Column(String(500), nullable=True)
    law_context = Column(ARRAY(String), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("UserORM", back_populates="sessions")
    messages = relationship("ChatMessageORM", back_populates="session", lazy="selectin",
                            order_by="ChatMessageORM.created_at")


class ChatMessageORM(Base):
    __tablename__ = "chat_messages"

    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
                        nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSessionORM", back_populates="messages")


class SavedResearchORM(Base):
    __tablename__ = "saved_research"

    research_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"),
                     nullable=False, index=True)
    title = Column(String(500), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(JSON, nullable=False)
    tags = Column(ARRAY(String), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DraftORM(Base):
    __tablename__ = "drafts"

    draft_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"),
                     nullable=False, index=True)
    draft_type = Column(String(100), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    facts_input = Column(Text, nullable=True)
    sections_used = Column(ARRAY(String), nullable=True)
    citations_used = Column(JSON, nullable=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AuditLogORM(Base):
    __tablename__ = "audit_logs"

    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"),
                     nullable=True, index=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("UserORM", back_populates="audit_logs")


class IngestionJobORM(Base):
    __tablename__ = "ingestion_jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_type = Column(String(100), nullable=False)
    file_path = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="pending", index=True)
    documents_processed = Column(Integer, default=0)
    chunks_created = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
