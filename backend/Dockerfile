# ── NyayaAI Backend Dockerfile ────────────────────────────────────────────────
# Default base: python:3.11-slim — builds and runs anywhere, including WSL2
# without NVIDIA drivers configured, CI, and dev machines.
#
# For GPU production (4× RTX 6000 Ada):
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
# which sets:
#   build:
#     args:
#       BASE: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
#
# The EmbeddingService and Reranker both check torch.cuda.is_available()
# and fall back to CPU automatically — no code changes needed between modes.
ARG BASE=python:3.11-slim
FROM ${BASE}

# System dependencies — work on both slim and CUDA base images
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip build-essential libpq-dev curl \
    tesseract-ocr tesseract-ocr-eng libtesseract-dev \
    poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Python dependencies ───────────────────────────────────────────────────────
# requirements-gpu.txt (torch+cu121 ~2GB) is only installed when building
# with the CUDA base image — it's skipped on python:3.11-slim to keep
# the default build fast and runnable without NVIDIA drivers.
COPY backend/requirements-gpu.txt backend/requirements.txt ./backend/
ARG BASE=python:3.11-slim
RUN if echo "${BASE}" | grep -q "cuda"; then \
      pip install --no-cache-dir -r backend/requirements-gpu.txt; \
    else \
      pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu; \
    fi
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
