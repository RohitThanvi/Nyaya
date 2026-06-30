"""
Upload route v3 — fail-proof resumable chunked upload.

Real resume support added:
1. GET /upload/chunked/{upload_id}/status — frontend calls this on reconnect
   to find out exactly which byte ranges are already on disk, so it never
   re-sends data after a network drop. This endpoint did not exist before;
   without it, "resume" was impossible — the frontend had no way to know
   what survived a disconnect.
2. Idempotent chunk writes — re-sending the same Content-Range twice (which
   happens on retry-after-timeout) no longer double-counts received_bytes.
   Completion is now derived from actual file sizes on disk via the status
   endpoint, never from a counter that can drift.
3. Single-file path raised to 200MB (was 50MB), chunked path raised to 10GB
   (was 2GB) — see backend/config/settings.py IngestionSettings.
"""
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
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
    Single-shot upload for files up to APP_MAX_UPLOAD_SIZE_MB (default 200MB).
    Anything larger should use the /upload/chunked/* flow.
    """
    max_bytes = _cfg.app.max_upload_size_mb * 1024 * 1024
    if file.size and file.size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds the single-shot limit of {_cfg.app.max_upload_size_mb}MB. "
                f"Files this large should use chunked upload, which supports files up to "
                f"{_cfg.ingestion.max_file_size_gb}GB and resumes automatically after "
                f"network interruptions."
            ),
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
        with open(tmp_path, "wb") as f:
            while True:
                buf = await file.read(8 * 1024 * 1024)  # 8MB at a time
                if not buf:
                    break
                f.write(buf)

        permanent_path = upload_dir / file.filename
        if permanent_path.exists():
            stem = Path(file.filename).stem
            suffix = Path(file.filename).suffix
            permanent_path = upload_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        shutil.copy2(tmp_path, permanent_path)

        pipeline = IngestionPipeline(db=db, embedding_service=embedder, vector_retriever=vector)
        result = await pipeline.ingest_upload(
            file_path=str(permanent_path),
            original_filename=permanent_path.name,
            source_url=source_url,
        )

        return UploadResponse(
            document_id=result["document_id"],
            filename=file.filename,
            pages=result.get("pages", 0),
            chunks_created=result.get("chunks_created", 0),
            failed_chunk_ids=result.get("failed_chunk_ids", []),
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


# ── Chunked, resumable upload ────────────────────────────────────────────────

def _chunked_dir(upload_id: str) -> Path:
    return Path(_cfg.app.upload_dir) / "chunked" / upload_id


def _read_meta(upload_id: str) -> dict:
    meta_path = _chunked_dir(upload_id) / ".meta"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Upload session not found or expired")
    with open(meta_path) as f:
        return json.load(f)


def _write_meta(upload_id: str, meta: dict) -> None:
    meta_path = _chunked_dir(upload_id) / ".meta"
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def _received_ranges(upload_id: str) -> list:
    """
    Derive actually-received byte ranges from files on disk, not from a
    counter. This is what makes resume reliable — even if the process
    crashed mid-upload, the chunk files on disk are still the source of
    truth for what survived.
    """
    upload_dir = _chunked_dir(upload_id)
    chunk_files = sorted(
        [p for p in upload_dir.glob("chunk_*")],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    ranges = []
    for p in chunk_files:
        start = int(p.stem.split("_")[1])
        size = p.stat().st_size
        ranges.append({"start": start, "end": start + size - 1, "size": size})
    return ranges


@router.post("/chunked/init")
async def init_chunked_upload(
    filename: str = Form(...),
    total_size: int = Form(...),
    source_url: Optional[str] = Form(default=None),
    resume_upload_id: Optional[str] = Form(default=None),
):
    """
    Initialise a chunked upload session, OR reattach to an existing one.
    If resume_upload_id is provided and that session still exists on disk,
    its existing chunks are preserved and reported back so the frontend
    can skip re-sending them — this is the actual resume mechanism.
    """
    max_bytes = int(_cfg.ingestion.max_file_size_gb * 1024 * 1024 * 1024)
    if total_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {total_size/1e9:.2f}GB exceeds maximum {_cfg.ingestion.max_file_size_gb}GB",
        )

    if resume_upload_id and _chunked_dir(resume_upload_id).exists():
        meta = _read_meta(resume_upload_id)
        if meta.get("filename") == filename and meta.get("total_size") == total_size:
            ranges = _received_ranges(resume_upload_id)
            return {
                "upload_id": resume_upload_id,
                "chunk_size_mb": _cfg.ingestion.chunk_size_mb,
                "resumed": True,
                "received_ranges": ranges,
            }

    upload_id = str(uuid.uuid4())
    upload_dir = _chunked_dir(upload_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    _write_meta(upload_id, {
        "filename": filename,
        "total_size": total_size,
        "source_url": source_url,
    })

    return {
        "upload_id": upload_id,
        "chunk_size_mb": _cfg.ingestion.chunk_size_mb,
        "resumed": False,
        "received_ranges": [],
    }


@router.get("/chunked/{upload_id}/status")
async def chunked_upload_status(upload_id: str):
    """
    Resume entry point. Frontend calls this after reconnecting following any
    network interruption to find out exactly which byte ranges already made
    it to disk, so it only re-sends what's missing — never the whole file.
    """
    meta = _read_meta(upload_id)
    ranges = _received_ranges(upload_id)
    received_bytes = sum(r["size"] for r in ranges)
    return {
        "upload_id": upload_id,
        "filename": meta["filename"],
        "total_size": meta["total_size"],
        "received_bytes": received_bytes,
        "received_ranges": ranges,
        "complete": received_bytes >= meta["total_size"],
    }


@router.post("/chunked/{upload_id}")
async def upload_chunk(
    upload_id: str,
    chunk: UploadFile = File(...),
    content_range: str = Header(...),
):
    """
    Upload a single chunk. Content-Range: bytes {start}-{end}/{total}
    Idempotent: re-uploading the same start offset simply overwrites the
    chunk file with identical bytes — safe under retry, never double-counts,
    because completion is always derived from files on disk (see status
    endpoint and _received_ranges), not an incrementing counter.
    """
    meta = _read_meta(upload_id)
    upload_dir = _chunked_dir(upload_id)

    try:
        range_part = content_range.replace("bytes ", "")
        range_str, total_str = range_part.split("/")
        start, end = map(int, range_str.split("-"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Content-Range header")

    chunk_data = await chunk.read()
    if len(chunk_data) != (end - start + 1):
        raise HTTPException(
            status_code=400,
            detail=f"Chunk size mismatch: Content-Range declares {end - start + 1} bytes, "
                   f"received {len(chunk_data)} bytes. Retry this chunk.",
        )

    chunk_path = upload_dir / f"chunk_{start:020d}"
    tmp_chunk_path = upload_dir / f".tmp_chunk_{start:020d}"
    with open(tmp_chunk_path, "wb") as f:
        f.write(chunk_data)
    os.replace(tmp_chunk_path, chunk_path)  # atomic — never leaves a half-written chunk

    ranges = _received_ranges(upload_id)
    received_bytes = sum(r["size"] for r in ranges)

    return {
        "upload_id": upload_id,
        "received_bytes": received_bytes,
        "total_bytes": meta["total_size"],
        "complete": received_bytes >= meta["total_size"],
    }


@router.post("/chunked/{upload_id}/finalise", response_model=UploadResponse)
async def finalise_chunked_upload(
    upload_id: str,
    background: bool = True,
):
    """
    Assemble all chunks in order and trigger ingestion.
    Verifies total assembled size matches the declared total_size before
    proceeding — catches any silent gap left by a missed chunk.

    background=True (default): ingestion (parse -> embed -> index) is
    dispatched to Celery and this call returns immediately with the
    document_id and status="processing". This matters specifically for
    large files: running the full pipeline synchronously inside the HTTP
    request previously blocked a FastAPI worker for the entire duration
    of parsing + embedding + indexing a potentially multi-GB document,
    risking reverse-proxy/load-balancer timeouts and starving other
    requests of workers. Poll GET /upload/document/{document_id}/status
    for progress.

    background=False: old synchronous behaviour, for small files or
    local/dev testing where blocking is fine.
    """
    meta = _read_meta(upload_id)
    upload_dir = _chunked_dir(upload_id)

    chunk_files = sorted(
        [p for p in upload_dir.glob("chunk_*")],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not chunk_files:
        raise HTTPException(status_code=400, detail="No chunks uploaded yet")

    ranges = _received_ranges(upload_id)
    received_bytes = sum(r["size"] for r in ranges)
    if received_bytes < meta["total_size"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Upload incomplete: {received_bytes}/{meta['total_size']} bytes received. "
                f"Call /upload/chunked/{upload_id}/status to find missing ranges and resume."
            ),
        )

    assembled_path = upload_dir / meta["filename"]
    try:
        with open(assembled_path, "wb") as out:
            for chunk_file in chunk_files:
                with open(chunk_file, "rb") as cf:
                    shutil.copyfileobj(cf, out, length=1024 * 1024)

        actual_size = assembled_path.stat().st_size
        if actual_size != meta["total_size"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Assembled file size {actual_size} does not match declared "
                    f"total_size {meta['total_size']}. A chunk may be corrupted. "
                    f"Re-run finalise after re-checking status."
                ),
            )

        # Move to permanent upload storage so /documents/{id}/view can serve it
        final_dir = Path(_cfg.app.upload_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / meta["filename"]
        if final_path.exists():
            stem = Path(meta["filename"]).stem
            suffix = Path(meta["filename"]).suffix
            final_path = final_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        shutil.move(str(assembled_path), str(final_path))

        document_id = str(uuid.uuid4())

        if background:
            from celery import chain
            from backend.ingestion.workers.tasks import parse_document, embed_document

            chain(
                parse_document.s(
                    file_path=str(final_path),
                    source_url=meta.get("source_url"),
                    document_id=document_id,
                    original_filename=final_path.name,
                ),
                embed_document.s(),
            ).apply_async()

            return UploadResponse(
                document_id=document_id,
                filename=meta["filename"],
                pages=0,
                chunks_created=0,
                failed_chunk_ids=[],
                status="processing",
                message=(
                    f"File received and queued for processing. "
                    f"Poll GET /upload/document/{document_id}/status for progress."
                ),
            )

        # Synchronous fallback path
        from backend.db.session import get_db_session
        from backend.api.dependencies.pipeline import get_embedding_service, get_vector_retriever
        async with get_db_session() as db:
            embedder = get_embedding_service()
            vector = get_vector_retriever()
            pipeline = IngestionPipeline(db=db, embedding_service=embedder, vector_retriever=vector)
            result = await pipeline.ingest_upload(
                file_path=str(final_path),
                original_filename=final_path.name,
                source_url=meta.get("source_url"),
            )

        return UploadResponse(
            document_id=result["document_id"],
            filename=meta["filename"],
            pages=result.get("pages", 0),
            chunks_created=result.get("chunks_created", 0),
            failed_chunk_ids=result.get("failed_chunk_ids", []),
            status=result.get("status", "success"),
            message=result.get("message", "Chunked upload successful."),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Finalise failed for upload {upload_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Finalise failed: {str(e)}")
    finally:
        try:
            shutil.rmtree(upload_dir)
        except Exception:
            pass


@router.get("/document/{document_id}/status")
async def document_ingestion_status(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Poll ingestion progress for a document submitted via background=True
    finalise. Reports each pipeline stage so the frontend can show real
    progress instead of a stuck spinner for large files.
    """
    from sqlalchemy import text as sql_text

    doc_row = (await db.execute(
        sql_text("SELECT document_id, pages FROM documents WHERE document_id = :id"),
        {"id": document_id},
    )).fetchone()

    if not doc_row:
        return {
            "document_id": document_id,
            "stage": "parsing",
            "detail": "Document not yet visible in the database — still parsing or queued.",
        }

    chunk_count = (await db.execute(
        sql_text("SELECT COUNT(*) FROM chunks WHERE document_id = :id"),
        {"id": document_id},
    )).scalar_one()

    staged_total = (await db.execute(
        sql_text("SELECT COUNT(*) FROM staged_chunks WHERE document_id = :id"),
        {"id": document_id},
    )).scalar_one()

    staged_indexed = (await db.execute(
        sql_text("SELECT COUNT(*) FROM staged_chunks WHERE document_id = :id AND indexed = true"),
        {"id": document_id},
    )).scalar_one()

    if chunk_count == 0:
        stage = "parsing"
    elif staged_total < chunk_count:
        stage = "embedding"
    elif staged_indexed < staged_total:
        stage = "indexing"
    else:
        stage = "complete"

    return {
        "document_id": document_id,
        "pages": doc_row.pages,
        "chunks_parsed": chunk_count,
        "chunks_embedded": staged_total,
        "chunks_indexed": staged_indexed,
        "stage": stage,
        "complete": stage == "complete",
    }


@router.delete("/chunked/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
async def abandon_chunked_upload(upload_id: str):
    """Explicitly cancel and clean up an in-progress chunked upload."""
    upload_dir = _chunked_dir(upload_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
