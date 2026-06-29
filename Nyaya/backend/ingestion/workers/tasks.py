"""
Celery tasks for distributed ingestion.

Task chain for a single document:
  parse_document.s(file_path, doc_meta) | embed_document.s()
  (flush_staged runs on schedule via Celery Beat)
"""
import asyncio
import logging
from typing import Any, Dict, Optional

from celery import shared_task

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


@shared_task(
    bind=True,
    name="backend.ingestion.workers.tasks.parse_document",
    queue="nyaya_parse",
    max_retries=3,
    default_retry_delay=30,
)
def parse_document(
    self,
    file_path: str,
    source_url: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 1: Parse PDF and write text chunks to PostgreSQL.
    Runs on CPU workers. Returns document_id for embed task.
    """
    try:
        from backend.ingestion.parsers.document_parser import DocumentParser
        from backend.ingestion.chunkers.legal_chunker import LegalChunker
        from backend.db.session import get_sync_session
        import uuid
        from sqlalchemy import text

        parser = DocumentParser()
        chunker = LegalChunker()

        parsed = parser.parse(file_path)
        if not parsed.raw_text:
            return {"status": "empty", "file_path": file_path}

        parsed.metadata.source_url = source_url or parsed.metadata.source_url
        document_id = str(uuid.uuid4())
        parsed.document_id = document_id
        parsed.metadata.document_id = document_id

        chunks = chunker.chunk(parsed)
        if not chunks:
            return {"status": "no_chunks", "document_id": document_id}

        # Write document + chunks to PostgreSQL synchronously
        with get_sync_session() as db:
            db.execute(text("""
                INSERT INTO documents (
                    document_id, document_type, law, court, court_name,
                    citation, year, source_url, is_landmark, language,
                    pages, uploaded_by, created_at
                ) VALUES (
                    :doc_id, :doc_type, :law, :court, :court_name,
                    :citation, :year, :source_url, false, :language,
                    :pages, :uploaded_by, NOW()
                ) ON CONFLICT (document_id) DO NOTHING
            """), {
                "doc_id": document_id,
                "doc_type": parsed.metadata.document_type.value,
                "law": parsed.metadata.law.value if parsed.metadata.law else None,
                "court": parsed.metadata.court.value if parsed.metadata.court else None,
                "court_name": parsed.metadata.court_name,
                "citation": parsed.metadata.citation,
                "year": parsed.metadata.year,
                "source_url": parsed.metadata.source_url,
                "language": parsed.metadata.language,
                "pages": parsed.pages,
                "uploaded_by": user_id,
            })

            for chunk in chunks:
                db.execute(text("""
                    INSERT INTO chunks (
                        chunk_id, document_id, chunk_type, content,
                        content_length, chunk_index, page_number,
                        section_ref, subsection_ref, content_tsv
                    ) VALUES (
                        :chunk_id, :document_id, :chunk_type, :content,
                        :content_length, :chunk_index, :page_number,
                        :section_ref, :subsection_ref,
                        to_tsvector('english', :content)
                    ) ON CONFLICT (chunk_id) DO NOTHING
                """), {
                    "chunk_id": chunk.chunk_id,
                    "document_id": document_id,
                    "chunk_type": chunk.chunk_type.value,
                    "content": chunk.content,
                    "content_length": chunk.content_length,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section_ref": chunk.section_ref,
                    "subsection_ref": chunk.subsection_ref,
                })
            db.commit()

        logger.info(f"Parsed {len(chunks)} chunks → document_id={document_id}")
        return {
            "status": "parsed",
            "document_id": document_id,
            "chunk_ids": [c.chunk_id for c in chunks],
            "chunk_contents": {c.chunk_id: c.content for c in chunks},
            "chunk_payloads": {
                c.chunk_id: {
                    "chunk_id": c.chunk_id,
                    "document_id": document_id,
                    "chunk_type": c.chunk_type.value,
                    "content": c.content,
                    "content_length": c.content_length,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "section_ref": c.section_ref,
                    "document_type": parsed.metadata.document_type.value,
                    "law": parsed.metadata.law.value if parsed.metadata.law else None,
                    "court": parsed.metadata.court.value if parsed.metadata.court else None,
                    "court_name": parsed.metadata.court_name,
                    "citation": parsed.metadata.citation,
                    "year": parsed.metadata.year,
                    "source_url": parsed.metadata.source_url,
                    "is_landmark": False,
                    "language": parsed.metadata.language,
                }
                for c in chunks
            },
        }
    except Exception as exc:
        logger.error(f"parse_document failed for {file_path}: {exc}")
        raise self.retry(exc=exc)


@shared_task(
    bind=True,
    name="backend.ingestion.workers.tasks.embed_document",
    queue="nyaya_embed",
    max_retries=3,
    default_retry_delay=60,
    time_limit=600,
)
def embed_document(self, parse_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2: Embed parsed chunks and write to staging table.
    Runs on GPU workers. Staging flush → Qdrant happens separately.
    """
    if parse_result.get("status") != "parsed":
        return parse_result

    document_id = parse_result["document_id"]
    chunk_ids = parse_result["chunk_ids"]
    contents = parse_result["chunk_contents"]
    payloads = parse_result["chunk_payloads"]

    try:
        from backend.embeddings.service import EmbeddingService
        from backend.db.session import get_sync_session
        from backend.config.settings import get_settings
        from sqlalchemy import text

        cfg = get_settings().ingestion
        embedder = EmbeddingService()

        # Embed in GPU batches
        all_texts = [contents[cid] for cid in chunk_ids]
        batch_size = cfg.ingest_batch_size
        all_embeddings = []

        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            embs = embedder.embed_batch_sync(batch)
            all_embeddings.extend(embs)

        # Write to staging table
        with get_sync_session() as db:
            for cid, emb in zip(chunk_ids, all_embeddings):
                db.execute(text("""
                    INSERT INTO staged_chunks
                        (chunk_id, document_id, embedding, metadata, indexed)
                    VALUES (:chunk_id, :document_id, :embedding::vector, :meta::jsonb, false)
                    ON CONFLICT (chunk_id) DO NOTHING
                """), {
                    "chunk_id": cid,
                    "document_id": document_id,
                    "embedding": emb,
                    "meta": payloads[cid],
                })
            db.commit()

        logger.info(f"Embedded {len(chunk_ids)} chunks → staging for {document_id}")
        return {"status": "embedded", "document_id": document_id, "count": len(chunk_ids)}

    except Exception as exc:
        logger.error(f"embed_document failed for {document_id}: {exc}")
        raise self.retry(exc=exc)


@shared_task(
    name="backend.ingestion.workers.tasks.flush_staged",
    queue="nyaya_flush",
)
def flush_staged() -> Dict[str, Any]:
    """
    Stage 3 (scheduled): Flush unindexed staged chunks to Qdrant in batches.
    Runs every INGEST_FLUSH_INTERVAL_S seconds via Celery Beat.
    """
    try:
        from backend.db.session import get_sync_session
        from backend.retrieval.vector.retriever import VectorRetriever
        from backend.config.settings import get_settings
        from sqlalchemy import text
        import asyncio

        cfg = get_settings().ingestion
        vector = VectorRetriever()
        total = 0

        with get_sync_session() as db:
            result = db.execute(text("""
                SELECT chunk_id, embedding, metadata
                FROM staged_chunks
                WHERE indexed = false
                ORDER BY chunk_id
                LIMIT :batch
            """), {"batch": cfg.flush_batch_size})
            rows = result.fetchall()

        if not rows:
            return {"status": "nothing_to_flush"}

        chunk_ids = [str(r.chunk_id) for r in rows]
        vectors = [list(r.embedding) for r in rows]
        payloads = [dict(r.metadata) for r in rows]

        loop = asyncio.new_event_loop()
        success = loop.run_until_complete(
            vector.upsert_batch(chunk_ids, vectors, payloads)
        )
        loop.close()

        if success:
            with get_sync_session() as db:
                db.execute(text("""
                    UPDATE staged_chunks SET indexed = true
                    WHERE chunk_id = ANY(:ids)
                """), {"ids": chunk_ids})
                db.commit()
            total = len(chunk_ids)
            logger.info(f"Flushed {total} staged chunks to Qdrant")

        return {"status": "flushed", "count": total}

    except Exception as e:
        logger.error(f"flush_staged failed: {e}")
        return {"status": "error", "error": str(e)}
