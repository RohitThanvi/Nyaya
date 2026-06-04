"""Upload route — handles PDF/TXT document ingestion."""
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.api.dependencies.auth import get_current_active_user
from backend.api.dependencies.pipeline import get_embedding_service, get_vector_retriever
from backend.config.settings import get_settings
from backend.embeddings.service import EmbeddingService
from backend.ingestion.chunkers.legal_chunker import LegalChunker
from backend.ingestion.parsers.document_parser import MetadataExtractor, PDFParser
from backend.ingestion.pipeline.ingest import IngestionPipeline
from backend.models.domain import UploadResponse, UserInDB
from backend.retrieval.vector.retriever import VectorRetriever

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
ALLOWED_TYPES = {"application/pdf", "text/plain"}
MAX_MB = get_settings().app.max_upload_size_mb


@router.post("/upload", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_retriever: VectorRetriever = Depends(get_vector_retriever),
    current_user: UserInDB = Depends(get_current_active_user),
) -> UploadResponse:
    """Upload and index a legal document for Q&A."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Only PDF and TXT files allowed. Got: {file.content_type}")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_MB}MB")

    # Save temporarily
    upload_dir = Path(get_settings().app.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload").suffix
    tmp_path = str(upload_dir / f"{uuid.uuid4()}{suffix}")

    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        pipeline = IngestionPipeline(
            embedding_service=embedding_service,
            vector_retriever=vector_retriever,
            chunker=LegalChunker(),
            pdf_parser=PDFParser(),
            metadata_extractor=MetadataExtractor(),
        )
        doc_id, pages, chunks = await pipeline.ingest_upload(
            file_path=tmp_path,
            original_filename=file.filename or "upload",
            user_id=str(current_user.user_id),
        )
        return UploadResponse(
            document_id=doc_id,
            filename=file.filename or "upload",
            pages=pages,
            chunks_created=chunks,
            status="success",
            message=f"Document indexed. {chunks} searchable chunks created.",
        )
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
