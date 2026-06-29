"""
CLI for batch ingestion of legal documents.

Usage:
    python -m backend.ingestion.pipeline.cli --source pdf --file ./data/raw/judgments/air2024sc111.pdf
    python -m backend.ingestion.pipeline.cli --source directory --dir ./data/raw/judgments

Rewritten to call the actual IngestionPipeline API:
    IngestionPipeline(db, embedding_service, vector_retriever)
    pipeline.ingest_upload(file_path, original_filename, source_url, user_id)
    pipeline.ingest_directory(directory, recursive, resume, progress_callback)
"""
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def ingest_file(file_path: str, source_url: Optional[str] = None):
    from backend.db.session import get_db_session
    from backend.embeddings.service import EmbeddingService
    from backend.ingestion.pipeline.ingest import IngestionPipeline
    from backend.retrieval.vector.retriever import VectorRetriever

    embedder = EmbeddingService()
    vr = VectorRetriever()
    await vr.ensure_collection()

    async with get_db_session() as db:
        pipeline = IngestionPipeline(db=db, embedding_service=embedder, vector_retriever=vr)
        result = await pipeline.ingest_upload(
            file_path=file_path,
            original_filename=Path(file_path).name,
            source_url=source_url,
        )
        logger.info(
            f"✓ {Path(file_path).name} → doc_id={result['document_id'][:8]}, "
            f"chunks={result['chunks_created']}, pages={result.get('pages', 0)}"
        )
        return result


async def ingest_directory(directory: str, resume: bool = True):
    from backend.db.session import get_db_session
    from backend.embeddings.service import EmbeddingService
    from backend.ingestion.pipeline.ingest import IngestionPipeline
    from backend.retrieval.vector.retriever import VectorRetriever

    embedder = EmbeddingService()
    vr = VectorRetriever()
    await vr.ensure_collection()

    async with get_db_session() as db:
        pipeline = IngestionPipeline(db=db, embedding_service=embedder, vector_retriever=vr)

        def _progress(i, total, fname):
            logger.info(f"[{i}/{total}] {fname}")

        summary = await pipeline.ingest_directory(
            directory=directory,
            recursive=True,
            resume=resume,
            progress_callback=_progress,
        )
        logger.info(
            f"Ingestion complete: {summary['ingested']} ingested, "
            f"{summary['skipped']} skipped, {len(summary['errors'])} failed, "
            f"{summary['total_chunks']} total chunks"
        )
        if summary["errors"]:
            for e in summary["errors"]:
                logger.error(f"  ✗ {e['file']}: {e['error']}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="NyayaAI Document Ingestion CLI")
    parser.add_argument("--source", choices=["pdf", "directory"], required=True)
    parser.add_argument("--file", help="Single PDF/TXT path (for --source pdf)")
    parser.add_argument("--dir", help="Directory path (for --source directory)")
    parser.add_argument("--source-url", help="Canonical public URL for the document", default=None)
    parser.add_argument("--no-resume", action="store_true", help="Re-ingest already-indexed files")
    args = parser.parse_args()

    if args.source == "pdf":
        if not args.file:
            parser.error("--file required with --source pdf")
        asyncio.run(ingest_file(args.file, source_url=args.source_url))
    elif args.source == "directory":
        if not args.dir:
            parser.error("--dir required with --source directory")
        asyncio.run(ingest_directory(args.dir, resume=not args.no_resume))


if __name__ == "__main__":
    main()
