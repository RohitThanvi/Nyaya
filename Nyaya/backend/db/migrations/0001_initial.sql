-- NyayaAI initial schema migration
-- Run with: psql -U nyaya_user -d nyaya_ai -f 0001_initial.sql

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
-- pgvector for native vector storage (optional, falls back to JSONB)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    document_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_type     VARCHAR(50) NOT NULL,
    law               VARCHAR(50),
    court             VARCHAR(50),
    court_name        VARCHAR(200),
    case_number       VARCHAR(200),
    citation          VARCHAR(200),
    year              INTEGER,
    date_decided      TIMESTAMPTZ,
    bench             TEXT[],
    parties           JSONB,
    topic             VARCHAR(500),
    keywords          TEXT[] DEFAULT '{}',
    source_url        VARCHAR(2000),
    original_filename VARCHAR(500),
    is_landmark       BOOLEAN DEFAULT false,
    language          VARCHAR(10) DEFAULT 'en',
    pages             INTEGER DEFAULT 0,
    uploaded_by       VARCHAR(100),
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_documents_law ON documents(law);
CREATE INDEX IF NOT EXISTS ix_documents_court ON documents(court);
CREATE INDEX IF NOT EXISTS ix_documents_year ON documents(year);
CREATE INDEX IF NOT EXISTS ix_documents_citation ON documents(citation);
CREATE INDEX IF NOT EXISTS ix_documents_law_year ON documents(law, year);
CREATE INDEX IF NOT EXISTS ix_documents_original_filename ON documents(original_filename);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id    UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_type     VARCHAR(50) NOT NULL DEFAULT 'passage',
    content        TEXT NOT NULL,
    content_length INTEGER DEFAULT 0,
    chunk_index    INTEGER NOT NULL DEFAULT 0,
    page_number    INTEGER,
    section_ref    VARCHAR(50),
    subsection_ref VARCHAR(100),
    content_tsv    TSVECTOR,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_chunks_document_index ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS ix_chunks_section_ref ON chunks(section_ref);
CREATE INDEX IF NOT EXISTS ix_chunks_content_tsv ON chunks USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS ix_chunks_document_type ON chunks(document_id, chunk_type);

-- Auto-update content_tsv on insert/update
CREATE OR REPLACE FUNCTION update_chunk_tsv()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv = to_tsvector('english', NEW.content);
    NEW.content_length = length(NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks;
CREATE TRIGGER trg_chunks_tsv
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunk_tsv();

-- Staged chunks table (ingestion pipeline)
CREATE TABLE IF NOT EXISTS staged_chunks (
    chunk_id    UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    embedding   JSONB NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    indexed     BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_staged_unindexed ON staged_chunks(indexed, document_id)
    WHERE indexed = false;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    full_name       VARCHAR(200) NOT NULL,
    role            VARCHAR(50) NOT NULL DEFAULT 'researcher',
    hashed_password VARCHAR(255) NOT NULL,
    is_active       BOOLEAN DEFAULT true,
    bar_enrollment  VARCHAR(100),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_login      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);

-- Chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title       VARCHAR(500),
    document_id UUID REFERENCES documents(document_id) ON DELETE SET NULL,
    law_filter  TEXT[],
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_chat_sessions_user ON chat_sessions(user_id, updated_at DESC);

-- Chat messages
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role                VARCHAR(20) NOT NULL,
    content             TEXT NOT NULL,
    citations           JSONB DEFAULT '[]',
    hallucination_flags TEXT[] DEFAULT '{}',
    confidence          FLOAT,
    intent              VARCHAR(50),
    latency_ms          FLOAT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_chat_messages_session ON chat_messages(session_id, created_at);

-- Legal drafts
CREATE TABLE IF NOT EXISTS legal_drafts (
    draft_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(document_id) ON DELETE SET NULL,
    draft_type  VARCHAR(100) NOT NULL,
    content     TEXT NOT NULL,
    parties     JSONB DEFAULT '{}',
    court       VARCHAR(200),
    facts       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     VARCHAR(100),
    action      VARCHAR(100) NOT NULL,
    resource    VARCHAR(100),
    resource_id VARCHAR(200),
    details     JSONB DEFAULT '{}',
    ip_address  VARCHAR(50),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_audit_user_created ON audit_logs(user_id, created_at DESC);

-- Auto-update updated_at triggers
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_documents_updated
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_chat_sessions_updated
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_legal_drafts_updated
    BEFORE UPDATE ON legal_drafts
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
