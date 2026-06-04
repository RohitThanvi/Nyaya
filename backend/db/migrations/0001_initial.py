"""
Initial migration: Create all core tables with indexes and full-text search.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Extensions ──────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ── Users ────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="researcher"),
        sa.Column("bar_enrollment", sa.String(100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_email_active", "users", ["email", "is_active"])

    # ── Documents ────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_type", sa.String(50), nullable=False),
        sa.Column("law", sa.String(50), nullable=True),
        sa.Column("court", sa.String(100), nullable=True),
        sa.Column("court_name", sa.String(255), nullable=True),
        sa.Column("case_number", sa.String(255), nullable=True),
        sa.Column("citation", sa.String(255), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("date_decided", sa.DateTime(timezone=True), nullable=True),
        sa.Column("bench", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("parties", postgresql.JSONB(), nullable=True),
        sa.Column("section", sa.String(100), nullable=True),
        sa.Column("chapter", sa.String(255), nullable=True),
        sa.Column("topic", sa.String(500), nullable=True),
        sa.Column("keywords", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("is_landmark", sa.Boolean(), server_default="false"),
        sa.Column("language", sa.String(10), server_default="en"),
        sa.Column("raw_text_path", sa.Text(), nullable=True),
        sa.Column("total_chunks", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_documents_law_year", "documents", ["law", "year"])
    op.create_index("ix_documents_court_year", "documents", ["court", "year"])
    op.create_index("ix_documents_citation", "documents", ["citation"])
    op.create_index("ix_documents_type_law", "documents", ["document_type", "law"])

    # ── Chunks ───────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_type", sa.String(50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_length", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section_ref", sa.String(255), nullable=True),
        sa.Column("subsection_ref", sa.String(255), nullable=True),
        sa.Column("content_tsv", postgresql.TSVECTOR(), nullable=True),
        sa.Column("qdrant_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_chunks_document_index", "chunks", ["document_id", "chunk_index"])
    op.create_index("ix_chunks_section_ref", "chunks", ["section_ref"])
    op.create_index("ix_chunks_qdrant_id", "chunks", ["qdrant_id"])
    op.create_index("ix_chunks_type", "chunks", ["chunk_type"])

    # GIN index for full-text search
    op.execute(
        "CREATE INDEX ix_chunks_tsv ON chunks USING gin(content_tsv)"
    )

    # Trigram index for LIKE/ILIKE fast searches
    op.execute(
        "CREATE INDEX ix_chunks_content_trgm ON chunks USING gin(content gin_trgm_ops)"
    )

    # TSV trigger: auto-update content_tsv on INSERT/UPDATE
    op.execute("""
        CREATE OR REPLACE FUNCTION update_chunk_tsv() RETURNS trigger AS $$
        BEGIN
            NEW.content_tsv := to_tsvector(
                'english',
                COALESCE(NEW.content, '') || ' ' ||
                COALESCE(NEW.section_ref, '') || ' ' ||
                COALESCE(NEW.subsection_ref, '')
            );
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER chunk_tsv_trigger
        BEFORE INSERT OR UPDATE ON chunks
        FOR EACH ROW EXECUTE FUNCTION update_chunk_tsv();
    """)

    # ── Chat Sessions ─────────────────────────────────────────────────────
    op.create_table(
        "chat_sessions",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("law_context", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])

    # ── Chat Messages ─────────────────────────────────────────────────────
    op.create_table(
        "chat_messages",
        sa.Column("message_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_chat_messages_session", "chat_messages", ["session_id"])

    # ── Saved Research ─────────────────────────────────────────────────────
    op.create_table(
        "saved_research",
        sa.Column("research_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("response", postgresql.JSONB(), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # ── Drafts ─────────────────────────────────────────────────────────────
    op.create_table(
        "drafts",
        sa.Column("draft_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("draft_type", sa.String(100), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("facts_input", sa.Text(), nullable=True),
        sa.Column("sections_used", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("citations_used", postgresql.JSONB(), nullable=True),
        sa.Column("version", sa.Integer(), server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # ── Audit Logs ──────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("log_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(100), nullable=True),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_audit_logs_user_action", "audit_logs", ["user_id", "action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

    # ── Ingestion Jobs ──────────────────────────────────────────────────────
    op.create_table(
        "ingestion_jobs",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("source_type", sa.String(100), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("documents_processed", sa.Integer(), server_default="0"),
        sa.Column("chunks_created", sa.Integer(), server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
    op.drop_table("audit_logs")
    op.drop_table("drafts")
    op.drop_table("saved_research")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("users")
    op.execute("DROP FUNCTION IF EXISTS update_chunk_tsv() CASCADE")
