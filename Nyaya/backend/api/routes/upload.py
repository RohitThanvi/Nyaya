"""
Upload route v2 — chunked HTTP upload + resume + document-scoped querying.

Fixes from v1:
1. Chunked upload with Content-Range support (handles 100MB+ files)
2. After upload, queries go to chatApi with document_id (not global search)
3. source_url can be provided at upload time (for India Kanoon links etc.)
4. File size validated against configurable MAX_FILE_SIZE_GB
5. Resume: incomplete upload sessions stored in Redis, resumable by upload_id
"""
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies.pipeline import get_embedding_service, get_vector_retriever
from backend.config.settings import get_settings
from backend.db.session import get_db
from backend.embeddings.service import EmbeddingService
from backend.ingestion.pipeline.ingest import IngestionPipeline
from backend.models.domain import UploadResponse
from backend.retrieval.vector.retriever import VectorRetriever

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger(__name__)
_cfg = get_settings()


@router.post("/", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    source_url: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedding_service),
    vector: VectorRetriever = Depends(get_vector_retriever),
):
    """
    Single-file upload (≤ MAX_UPLOAD_SIZE_MB).
    For larger files use /upload/chunked endpoints.
    """
    max_bytes = _cfg.app.max_upload_size_mb * 1024 * 1024
    if file.size and file.size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {_cfg.app.max_upload_size_mb}MB. "
                   f"Use the /upload/chunked endpoint for larger files.",
        )

    if not file.filename or not file.filename.lower().endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF, TXT, and DOCX files are supported.",
        )

    upload_dir = Path(_cfg.app.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        pipeline = IngestionPipeline(db=db, embedding_service=embedder, vector_retriever=vector)
        result = await pipeline.ingest_upload(
            file_path=str(tmp_path),
            original_filename=file.filename,
            source_url=source_url,
        )

        return UploadResponse(
            document_id=result["document_id"],
            filename=file.filename,
            pages=result.get("pages", 0),
            chunks_created=result.get("chunks_created", 0),
            status=result.get("status", "success"),
            message=result.get("message", "Upload successful."),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload processing failed: {str(e)}",
        )
    finally:
        if tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── Chunked upload endpoints ──────────────────────────────────────────────────

@router.post("/chunked/init")
async def init_chunked_upload(
    filename: str = Form(...),
    total_size: int = Form(...),
    source_url: Optional[str] = Form(default=None),
):
    """
    Initialise a chunked upload session.
    Returns upload_id that must be passed to each chunk upload.
    """
    max_bytes = int(_cfg.ingestion.max_file_size_gb * 1024 * 1024 * 1024)
    if total_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {total_size/1e9:.1f}GB exceeds maximum {_cfg.ingestion.max_file_size_gb}GB",
        )

    upload_id = str(uuid.uuid4())
    upload_dir = Path(_cfg.app.upload_dir) / "chunked" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Store session metadata in a marker file
    meta_path = upload_dir / ".meta"
    with open(meta_path, "w") as f:
        import json
        json.dump({
            "filename": filename,
            "total_size": total_size,
            "source_url": source_url,
            "received_bytes": 0,
            "chunks_received": [],
        }, f)

    return {"upload_id": upload_id, "chunk_size_mb": _cfg.ingestion.chunk_size_mb}


@router.post("/chunked/{upload_id}")
async def upload_chunk(
    upload_id: str,
    chunk: UploadFile = File(...),
    content_range: str = Header(...),   # "bytes 0-10485759/104857600"
):
    """
    Upload a single chunk of a large file.
    Content-Range header: bytes {start}-{end}/{total}
    """
    upload_dir = Path(_cfg.app.upload_dir) / "chunked" / upload_id
    meta_path = upload_dir / ".meta"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")

    import json
    with open(meta_path) as f:
        meta = json.load(f)

    # Parse Content-Range
    try:
        range_part = content_range.replace("bytes ", "")
        range_str, total_str = range_part.split("/")
        start, end = map(int, range_str.split("-"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Content-Range header")

    chunk_data = await chunk.read()
    chunk_path = upload_dir / f"chunk_{start:020d}"
    with open(chunk_path, "wb") as f:
        f.write(chunk_data)

    meta["received_bytes"] = meta["received_bytes"] + len(chunk_data)
    meta["chunks_received"].append(start)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return {
        "upload_id": upload_id,
        "received_bytes": meta["received_bytes"],
        "total_bytes": meta["total_size"],
        "complete": meta["received_bytes"] >= meta["total_size"],
    }


@router.post("/chunked/{upload_id}/finalise", response_model=UploadResponse)
async def finalise_chunked_upload(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedding_service),
    vector: VectorRetriever = Depends(get_vector_retriever),
):
    """
    Assemble all chunks and trigger ingestion.
    Call after all chunks are uploaded.
    """
    import json

    upload_dir = Path(_cfg.app.upload_dir) / "chunked" / upload_id
    meta_path = upload_dir / ".meta"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")

    with open(meta_path) as f:
        meta = json.load(f)

    # Sort and assemble chunks
    chunk_files = sorted(
        [p for p in upload_dir.iterdir() if p.name.startswith("chunk_")],
        key=lambda p: int(p.stem.split("_")[1]),
    )

    assembled_path = upload_dir / meta["filename"]
    try:
        with open(assembled_path, "wb") as out:
            for chunk_file in chunk_files:
                with open(chunk_file, "rb") as cf:
                    out.write(cf.read())

        pipeline = IngestionPipeline(db=db, embedding_service=embedder, vector_retriever=vector)
        result = await pipeline.ingest_upload(
            file_path=str(assembled_path),
            original_filename=meta["filename"],
            source_url=meta.get("source_url"),
        )

        return UploadResponse(
            document_id=result["document_id"],
            filename=meta["filename"],
            pages=result.get("pages", 0),
            chunks_created=result.get("chunks_created", 0),
            status=result.get("status", "success"),
            message=result.get("message", "Chunked upload successful."),
        )
    finally:
        # Cleanup chunked upload dir
        import shutil
        try:
            shutil.rmtree(upload_dir)
        except Exception:
            pass
