"""
Full ingestion pipeline.

Stages:
1. Parse (PDF/HTML → text + metadata)
2. Extract metadata (citation, year, court, law)
3. Chunk (semantic legal chunking)
4. Embed (batch embedding with BGE-large)
5. Index (PostgreSQL BM25 + Qdrant ANN)

Designed for both batch ingestion and single-document upload.
Uses async throughout with background worker support.
"""
import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.config.settings import get_settings
from backend.db.session import get_db_session
from backend.embeddings.service import EmbeddingService
from backend.ingestion.chunkers.legal_chunker import LegalChunker
from backend.ingestion.parsers.document_parser import (
    HTMLParser, MetadataExtractor, PDFParser
)
from backend.models.domain import (
    CourtType, DocumentMetadata, DocumentType, IngestionJob,
    LawCategory, LegalChunk, ParsedDocument
)
from backend.retrieval.vector.retriever import VectorRetriever

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Production ingestion pipeline.

    Processes documents end-to-end:
    parse → metadata → chunk → embed → pg_index + qdrant_index

    Key design choices:
    - Async throughout, releases control between heavy operations
    - Batch embedding (64 chunks at a time) to avoid OOM
    - Upsert semantics: re-indexing same doc_id replaces existing data
    - Stores raw text for future re-chunking without re-parsing
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_retriever: VectorRetriever,
        chunker: LegalChunker,
        pdf_parser: PDFParser,
        metadata_extractor: MetadataExtractor,
    ):
        self._embedder = embedding_service
        self._vector = vector_retriever
        self._chunker = chunker
        self._pdf_parser = pdf_parser
        self._meta_extractor = metadata_extractor
        self._settings = get_settings()

    async def ingest_pdf(
        self,
        file_path: str,
        document_type: DocumentType = DocumentType.JUDGMENT,
        law: Optional[LawCategory] = None,
        additional_metadata: Optional[Dict] = None,
        job_id: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Ingest a single PDF.
        Returns (document_id, chunks_created).
        """
        logger.info(f"Starting ingestion: {Path(file_path).name}")

        # 1. Parse
        raw_text, structure_hints, pages = self._pdf_parser.parse(file_path)
        if not raw_text or len(raw_text) < 100:
            raise ValueError(f"Could not extract meaningful text from {file_path}")

        # 2. Extract metadata
        if document_type == DocumentType.JUDGMENT:
            extracted_meta = self._meta_extractor.extract_judgment_metadata(
                raw_text, Path(file_path).name
            )
        else:
            extracted_meta = self._meta_extractor.extract_statute_metadata(
                raw_text, Path(file_path).name
            )

        if additional_metadata:
            extracted_meta.update(additional_metadata)

        # Determine law from extracted metadata or parameter
        doc_law = law
        if not doc_law and extracted_meta.get("law"):
            try:
                doc_law = LawCategory(extracted_meta["law"])
            except ValueError:
                pass

        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            document_type=document_type,
            law=doc_law,
            court=CourtType.SUPREME_COURT if "Supreme Court" in raw_text[:500] else None,
            court_name=extracted_meta.get("court_name"),
            case_number=extracted_meta.get("case_number"),
            citation=extracted_meta.get("citation"),
            year=extracted_meta.get("year"),
            bench=extracted_meta.get("bench"),
            parties=extracted_meta.get("parties"),
            file_path=file_path,
            language="en",
        )

        # 3. Chunk
        chunks = self._chunker.chunk(raw_text, metadata, structure_hints)
        if not chunks:
            raise ValueError(f"Chunking produced no chunks for {file_path}")

        logger.info(f"Created {len(chunks)} chunks from {pages} pages")

        # 4. Save to DB and vector store
        async with get_db_session() as db:
            await self._save_document(db, metadata, raw_text, pages, len(chunks))
            await self._save_chunks_to_pg(db, chunks)

        chunks_indexed = await self._embed_and_index(chunks)

        logger.info(
            f"Ingested {Path(file_path).name}: doc_id={metadata.document_id[:8]}, "
            f"chunks={chunks_indexed}"
        )
        return metadata.document_id, chunks_indexed

    async def ingest_upload(
        self,
        file_path: str,
        original_filename: str,
        user_id: str,
    ) -> Tuple[str, int, int]:
        """
        Ingest user-uploaded document.
        Returns (document_id, pages, chunks_created).
        """
        ext = Path(file_path).suffix.lower()
        if ext not in (".pdf", ".txt"):
            raise ValueError(f"Unsupported file type: {ext}")

        if ext == ".pdf":
            raw_text, structure_hints, pages = self._pdf_parser.parse(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            structure_hints = {}
            pages = 1

        metadata = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            document_type=DocumentType.UPLOAD,
            file_path=file_path,
            language="en",
            topic=original_filename,
        )

        chunks = self._chunker.chunk(raw_text, metadata, structure_hints)

        try:
            async with get_db_session() as db:
                await self._save_document(db, metadata, raw_text, pages, len(chunks))
                await self._save_chunks_to_pg(db, chunks)
                await db.commit()
        except Exception:
            raise

        chunks_created = await self._embed_and_index(chunks)
        return metadata.document_id, pages, chunks_created

    async def _embed_and_index(self, chunks: List[LegalChunk]) -> int:
        """Batch embed and upsert to Qdrant."""
        contents = [c.content for c in chunks]
        embeddings = await self._embedder.embed_passages_batched(
            contents, batch_size=64
        )
        upserted = await self._vector.upsert_chunks(chunks, embeddings)
        return upserted

    async def _save_document(
        self,
        db: AsyncSession,
        metadata: DocumentMetadata,
        raw_text: str,
        pages: int,
        chunk_count: int,
    ) -> None:
        """Upsert document record to PostgreSQL."""
        # Save raw text to disk (not in DB to avoid bloating)
        raw_dir = Path(self._settings.app.upload_dir) / "raw_text"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_text_path = str(raw_dir / f"{metadata.document_id}.txt")
        with open(raw_text_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        sql = text("""
            INSERT INTO documents (
                document_id, document_type, law, court, court_name,
                case_number, citation, year, date_decided, bench,
                parties, section, chapter, topic, keywords,
                source_url, file_path, is_landmark, language,
                raw_text_path, total_chunks, created_at, updated_at
            ) VALUES (
                :document_id, :document_type, :law, :court, :court_name,
                :case_number, :citation, :year, :date_decided, :bench,
                :parties, :section, :chapter, :topic, :keywords,
                :source_url, :file_path, :is_landmark, :language,
                :raw_text_path, :total_chunks, NOW(), NOW()
            )
            ON CONFLICT (document_id) DO UPDATE SET
                total_chunks = EXCLUDED.total_chunks,
                updated_at = NOW()
        """)
        await db.execute(sql, {
            "document_id": metadata.document_id,
            "document_type": metadata.document_type.value,
            "law": metadata.law.value if metadata.law else None,
            "court": metadata.court.value if metadata.court else None,
            "court_name": metadata.court_name,
            "case_number": metadata.case_number,
            "citation": metadata.citation,
            "year": metadata.year,
            "date_decided": metadata.date_decided,
            "bench": metadata.bench,
            "parties": metadata.parties,
            "section": metadata.section,
            "chapter": metadata.chapter,
            "topic": metadata.topic,
            "keywords": metadata.keywords,
            "source_url": metadata.source_url,
            "file_path": metadata.file_path,
            "is_landmark": metadata.is_landmark,
            "language": metadata.language,
            "raw_text_path": raw_text_path,
            "total_chunks": chunk_count,
        })

    async def _save_chunks_to_pg(
        self, db: AsyncSession, chunks: List[LegalChunk]
    ) -> None:
        """
        Batch insert chunks to PostgreSQL.
        TSV trigger fires automatically on INSERT.
        """
        # Delete existing chunks for this document (upsert semantics)
        if chunks:
            await db.execute(
                text("DELETE FROM chunks WHERE document_id = :doc_id"),
                {"doc_id": chunks[0].document_id}
            )

        # Batch insert
        BATCH = 100
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i: i + BATCH]
            values = []
            params: Dict = {}
            for j, chunk in enumerate(batch):
                prefix = f"c{i+j}_"
                values.append(
                    f"(:{prefix}chunk_id, :{prefix}document_id, :{prefix}chunk_type, "
                    f":{prefix}content, :{prefix}content_length, :{prefix}chunk_index, "
                    f":{prefix}page_number, :{prefix}section_ref, :{prefix}subsection_ref)"
                )
                params.update({
                    f"{prefix}chunk_id": chunk.chunk_id,
                    f"{prefix}document_id": chunk.document_id,
                    f"{prefix}chunk_type": chunk.chunk_type.value,
                    f"{prefix}content": chunk.content,
                    f"{prefix}content_length": chunk.content_length,
                    f"{prefix}chunk_index": chunk.chunk_index,
                    f"{prefix}page_number": chunk.page_number,
                    f"{prefix}section_ref": chunk.section_ref,
                    f"{prefix}subsection_ref": chunk.subsection_ref,
                })
            sql = text(
                "INSERT INTO chunks (chunk_id, document_id, chunk_type, content, "
                "content_length, chunk_index, page_number, section_ref, subsection_ref) "
                f"VALUES {', '.join(values)}"
            )
            await db.execute(sql, params)

        await db.flush()
