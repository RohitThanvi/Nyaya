"""
Partition chunks table by law category + add composite indexes for scale.

Why partitioning:
  An unpartitioned chunks table with 100M+ rows makes every BM25/FTS
  query a full-index scan regardless of how good the tsvector GIN index
  is. Postgres range/list partitioning lets each query prune to only the
  partition(s) matching the law_filter the user selected, typically
  cutting scan size by 80-90% for filtered queries (e.g. "search only
  BNS" scans the bns partition, not all 100M chunks).

Partition strategy: LIST on law (document_category). Legal queries are
  almost always filtered by law (BNS, BNSS, BSA, IPC, etc.) — this maps
  directly to partition pruning.

New composite indexes:
  - (document_id, chunk_index) for document-level retrieval ordering
  - (section_ref, document_id) for exact section lookup path
  - (chunk_type, document_id) for type-filtered retrieval

NOTE: This migration converts the existing chunks table to a partitioned
  table. If you have existing data, the conversion requires:
    1. Rename existing table to chunks_old
    2. Create new partitioned table
    3. INSERT INTO chunks SELECT * FROM chunks_old
    4. Drop chunks_old
  For a fresh install, only steps 2+ run (no data to migrate).

  Run with: alembic upgrade head
"""
from alembic import op
import sqlalchemy as sa


revision = "0003_partition_chunks_indexes"
down_revision = "0002_drop_redundant_tsv_trigger"
branch_labels = None
depends_on = None


PARTITIONS = [
    ("chunks_bns",        "bns"),
    ("chunks_bnss",       "bnss"),
    ("chunks_bsa",        "bsa"),
    ("chunks_ipc",        "ipc"),
    ("chunks_crpc",       "crpc"),
    ("chunks_evidence",   "evidence"),
    ("chunks_constitution","constitution"),
    ("chunks_judgment",   "judgment"),
    ("chunks_other",      "other"),
]


def upgrade() -> None:
    conn = op.get_bind()

    # Check if chunks is already partitioned
    result = conn.execute(sa.text("""
        SELECT relkind FROM pg_class
        WHERE relname = 'chunks' AND relnamespace = 'public'::regnamespace
    """)).fetchone()

    is_partitioned = result and result[0] == 'p'
    has_data = False

    if result and not is_partitioned:
        count = conn.execute(sa.text("SELECT COUNT(*) FROM chunks")).scalar()
        has_data = count > 0

        # Rename the existing table out of the way whenever it exists and
        # isn't already partitioned -- regardless of whether it has data.
        # Previously this only ran when has_data was true, so a freshly
        # initialized (empty) chunks table was left in place, the
        # subsequent `CREATE TABLE IF NOT EXISTS chunks (...) PARTITION BY`
        # silently no-opped (a table named chunks already existed), and
        # every `CREATE TABLE ... PARTITION OF chunks` after it failed with
        # `"chunks" is not partitioned` -- reproduced and confirmed against
        # a real Postgres instance.
        conn.execute(sa.text("ALTER TABLE chunks RENAME TO chunks_old"))

    if not is_partitioned:
        # Create the partitioned parent table
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        UUID        NOT NULL,
                document_id     UUID        NOT NULL,
                chunk_type      VARCHAR(50) NOT NULL DEFAULT 'passage',
                content         TEXT        NOT NULL,
                content_tsv     TSVECTOR    GENERATED ALWAYS AS (
                                    to_tsvector('english', content)
                                ) STORED,
                content_length  INTEGER     GENERATED ALWAYS AS (length(content)) STORED,
                chunk_index     INTEGER     NOT NULL DEFAULT 0,
                page_number     INTEGER,
                section_ref     VARCHAR(50),
                subsection_ref  VARCHAR(50),
                law             VARCHAR(50) NOT NULL DEFAULT 'other',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (chunk_id, law)
            ) PARTITION BY LIST (law)
        """))

        # Create law-specific partitions
        for partition_name, law_value in PARTITIONS:
            conn.execute(sa.text(f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                    PARTITION OF chunks FOR VALUES IN ('{law_value}')
            """))

        # Default partition catches any law values not listed above
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS chunks_default
                PARTITION OF chunks DEFAULT
        """))

        # GIN index on tsvector for full-text search — one per partition,
        # Postgres auto-creates these on child tables when created on parent.
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS ix_chunks_tsv
                ON chunks USING GIN (content_tsv)
        """))

        # Composite index: document-level chunk ordering (used when fetching
        # all chunks for a document to reconstruct context windows)
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS ix_chunks_doc_order
                ON chunks (document_id, chunk_index)
        """))

        # Composite index: exact section lookup (most common legal query pattern:
        # "Section 302 IPC" → section_ref='302', law='ipc')
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS ix_chunks_section
                ON chunks (section_ref, document_id)
                WHERE section_ref IS NOT NULL
        """))

        # Partial index for cross-encoder reranker candidate pre-filtering:
        # only SECTION and JUDGMENT_EXCERPT chunks carry verified legal holdings
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS ix_chunks_type_doc
                ON chunks (chunk_type, document_id)
        """))

        if has_data:
            # Migrate existing data — the law column needs to be populated from
            # the documents table since old chunks didn't store it directly.
            conn.execute(sa.text("""
                INSERT INTO chunks
                    (chunk_id, document_id, chunk_type, content, chunk_index,
                     page_number, section_ref, subsection_ref, law)
                SELECT
                    c.chunk_id, c.document_id, c.chunk_type, c.content,
                    c.chunk_index, c.page_number, c.section_ref, c.subsection_ref,
                    COALESCE(d.law, 'other')
                FROM chunks_old c
                LEFT JOIN documents d ON d.document_id = c.document_id
                ON CONFLICT DO NOTHING
            """))
            # Keep chunks_old until manually verified, then drop:
            # DROP TABLE chunks_old;
        else:
            # Nothing to migrate — chunks_old is just the empty pre-partition
            # table renamed out of the way above, safe to drop immediately.
            conn.execute(sa.text("DROP TABLE IF EXISTS chunks_old"))

    # Always ensure these indexes exist (idempotent on re-run)
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS ix_documents_law_year
            ON documents (law, year DESC)
    """))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS ix_documents_type_law
            ON documents (document_type, law)
    """))


def downgrade() -> None:
    # Partitioned tables can't be trivially un-partitioned — skip auto-downgrade
    pass
