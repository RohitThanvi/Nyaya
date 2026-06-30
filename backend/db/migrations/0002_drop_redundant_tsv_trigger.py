"""
Drop the per-row content_tsv trigger on chunks.

backend/ingestion/pipeline/ingest.py already computes
to_tsvector('english', :content) explicitly in the bulk INSERT statement
for every chunk. The trg_chunks_tsv BEFORE INSERT/UPDATE trigger then
recomputes the IDENTICAL tsvector again, row by row, inside the same
statement — Postgres still executes a per-row PL/pgSQL trigger call even
for multi-row INSERTs, which silently defeats most of the benefit of the
batched executemany() pattern the ingestion pipeline relies on for
throughput at scale (thousands of documents / millions of chunks).

No code path updates chunks.content without also being routed through the
same explicit-tsvector INSERT path, so dropping the trigger is safe.
"""
from alembic import op


revision = "0002_drop_redundant_tsv_trigger"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks")
    op.execute("DROP FUNCTION IF EXISTS update_chunk_tsv()")

    # content_length is also recomputed defensively at the app layer
    # (LegalChunk.content_length), so no data is lost by removing the
    # trigger that previously kept it in sync.


def downgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION update_chunk_tsv()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.content_tsv = to_tsvector('english', NEW.content);
            NEW.content_length = length(NEW.content);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_chunks_tsv
            BEFORE INSERT OR UPDATE ON chunks
            FOR EACH ROW EXECUTE FUNCTION update_chunk_tsv();
    """)
