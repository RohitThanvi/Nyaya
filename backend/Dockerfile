# ── NyayaAI Backend Dockerfile ────────────────────────────────────────────────
# Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04 (matches RTX 6000 Ada / Ada Lovelace)
# Provides: nvidia-smi, cuda runtime, libcudnn — needed by PyTorch GPU workers.
# CPU-only fallback: build with --build-arg BASE=python:3.11-slim and set
# EMBEDDING_DEVICE=cpu in .env (useful for CI / systems without NVIDIA drivers).
ARG BASE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM ${BASE}

# ── System Python ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    build-essential libpq-dev curl \
    # OCR + PDF
    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
    poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --upgrade pip

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Python dependencies (layer-cached) ────────────────────────────────────────
# Install GPU torch FIRST as a separate layer — it's the largest dep (~2GB)
# and changes rarely. The old requirements-heavy.txt installed the CPU build
# which then got overridden by the CUDA build in requirements.txt — leaving
# whichever pip resolved last (non-deterministic). Now it's explicit and ordered.
COPY backend/requirements-gpu.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements-gpu.txt

COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# ── Application ───────────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p /app/data/raw/uploads /app/data/raw/raw_text \
             /app/data/processed /app/data/embeddings/models \
             /app/data/chunked_uploads

# Non-root for security (but can still access /dev/nvidia* via group)
RUN useradd -m -u 1000 nyaya \
    && chown -R nyaya:nyaya /app
USER nyaya

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default: API server. Celery workers override CMD in docker-compose.
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", "--http", "httptools", \
     "--log-level", "info"]
