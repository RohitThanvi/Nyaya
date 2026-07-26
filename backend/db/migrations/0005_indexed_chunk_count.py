"""
Add documents.indexed_chunk_count — fixes a real stuck-progress bug.

Why:
  GET /upload/document/{id}/status derived "chunks indexed" as
  COUNT(*) FROM staged_chunks WHERE document_id=:id AND indexed=true.
  But flush_staged (tasks.py) DELETES rows from staged_chunks the moment
  they're successfully upserted to Qdrant — indexed=true rows never
  actually accumulate there, they're removed. So once a document finishes
  processing, staged_total (COUNT(*) for that document) drops back to 0,
  which makes the status endpoint's own "staged_total < chunk_count ->
  stage=embedding" check fire again — a fully, successfully indexed
  document can never report stage="complete". The frontend progress
  poller would spin forever on exactly the large documents this whole
  chunked-upload/background-ingestion path was built for.

  This adds a persistent counter on `documents` that flush_staged
  increments (grouped by document_id) in the same transaction it deletes
  the flushed rows, so completion status survives past the point where
  staged_chunks itself no longer has any evidence of what happened.
"""
from alembic import op
import sqlalchemy as sa


revision = "0005_indexed_chunk_count"
down_revision = "0004_staged_chunks_processing_flag"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("""
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS indexed_chunk_count INTEGER NOT NULL DEFAULT 0
    """))
    # Backfill: any document whose chunk count already matches what's on
    # disk with nothing left in staged_chunks was, in practice, already
    # fully flushed under the old (buggy) accounting — count its chunks
    # as indexed so existing documents don't regress to "embedding".
    conn.execute(sa.text("""
        UPDATE documents d
        SET indexed_chunk_count = (
            SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.document_id
        )
        WHERE NOT EXISTS (
            SELECT 1 FROM staged_chunks sc WHERE sc.document_id = d.document_id
        )
    """))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text(
        "ALTER TABLE documents DROP COLUMN IF EXISTS indexed_chunk_count"
    ))
