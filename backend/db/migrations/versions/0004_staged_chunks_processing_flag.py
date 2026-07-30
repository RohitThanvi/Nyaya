"""
Add processing flag to staged_chunks + fix flush index.

Why:
  flush_staged previously fetched unindexed rows then wrote to Qdrant
  and marked indexed=true in two separate transactions. If two flush
  tasks overlapped (previous tick slow, next tick fired) they both read
  the same rows and sent duplicate upserts to Qdrant — wasted work and
  potential HNSW index thrashing during bulk ingestion.

  The processing=true flag is set atomically via UPDATE...RETURNING with
  FOR UPDATE SKIP LOCKED before reading row content, so concurrent flush
  workers each process a disjoint set of rows.
"""
from alembic import op
import sqlalchemy as sa


revision = "0004_staged_chunks_flag"
down_revision = "0003_partition_chunks_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Add processing flag (idempotent)
    conn.execute(sa.text("""
        ALTER TABLE staged_chunks
        ADD COLUMN IF NOT EXISTS processing BOOLEAN NOT NULL DEFAULT false
    """))

    # Drop old index and create better one that includes processing flag
    conn.execute(sa.text(
        "DROP INDEX IF EXISTS ix_staged_unindexed"
    ))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS ix_staged_flushable
            ON staged_chunks (created_at, chunk_id)
            WHERE indexed = false AND processing = false
    """))

    # Release any stuck processing=true rows from previous deployments
    conn.execute(sa.text("""
        UPDATE staged_chunks SET processing = false
        WHERE processing = true AND indexed = false
    """))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("DROP INDEX IF EXISTS ix_staged_flushable"))
    conn.execute(sa.text("ALTER TABLE staged_chunks DROP COLUMN IF EXISTS processing"))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS ix_staged_unindexed
            ON staged_chunks (indexed, document_id)
            WHERE indexed = false
    """))
