"""
CLI for batch ingestion of legal documents.

Usage:
    python -m backend.ingestion.pipeline.cli --source india_code --law BNS --dir ./data/raw/bns
    python -m backend.ingestion.pipeline.cli --source pdf --file ./data/raw/judgments/air2024sc111.pdf
    python -m backend.ingestion.pipeline.cli --source directory --dir ./data/raw/judgments --type judgment
"""
import argparse
import asyncio
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def ingest_file(file_path: str, doc_type: str, law: str | None):
    from backend.embeddings.service import EmbeddingService
    from backend.ingestion.chunkers.legal_chunker import LegalChunker
    from backend.ingestion.parsers.document_parser import MetadataExtractor, PDFParser
    from backend.ingestion.pipeline.ingest import IngestionPipeline
    from backend.models.domain import DocumentType, LawCategory
    from backend.retrieval.vector.retriever import VectorRetriever

    pipeline = IngestionPipeline(
        embedding_service=EmbeddingService(),
        vector_retriever=VectorRetriever(),
        chunker=LegalChunker(),
        pdf_parser=PDFParser(),
        metadata_extractor=MetadataExtractor(),
    )

    dtype = DocumentType(doc_type)
    law_enum = LawCategory(law.upper()) if law else None

    doc_id, chunks = await pipeline.ingest_pdf(
        file_path=file_path,
        document_type=dtype,
        law=law_enum,
    )
    logger.info(f"✓ {Path(file_path).name} → doc_id={doc_id[:8]}, chunks={chunks}")
    return doc_id, chunks


async def ingest_directory(directory: str, doc_type: str, law: str | None):
    paths = list(Path(directory).glob("**/*.pdf")) + list(Path(directory).glob("**/*.txt"))
    logger.info(f"Found {len(paths)} files in {directory}")
    total_chunks = 0
    failed = 0
    for path in paths:
        try:
            _, chunks = await ingest_file(str(path), doc_type, law)
            total_chunks += chunks
        except Exception as e:
            logger.error(f"✗ {path.name}: {e}")
            failed += 1
    logger.info(f"Ingestion complete: {len(paths)-failed} succeeded, {failed} failed, {total_chunks} total chunks")


def main():
    parser = argparse.ArgumentParser(description="NyayaAI Document Ingestion CLI")
    parser.add_argument("--source", choices=["pdf", "directory"], required=True)
    parser.add_argument("--file", help="Single PDF path (for --source pdf)")
    parser.add_argument("--dir", help="Directory path (for --source directory)")
    parser.add_argument("--type", default="judgment", choices=["judgment", "statute", "notification", "circular"])
    parser.add_argument("--law", help="Law category: BNS, BNSS, BSA, IPC, CrPC", default=None)
    args = parser.parse_args()

    if args.source == "pdf":
        if not args.file:
            parser.error("--file required with --source pdf")
        asyncio.run(ingest_file(args.file, args.type, args.law))
    elif args.source == "directory":
        if not args.dir:
            parser.error("--dir required with --source directory")
        asyncio.run(ingest_directory(args.dir, args.type, args.law))


if __name__ == "__main__":
    main()
