FROM python:3.11-slim

# Install system dependencies in two groups for better cache layering
# Group 1: build tools and postgres
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Group 2: OCR and image processing
# libgl1 is the correct package name in Debian bookworm (python:3.11-slim base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app
# Install Python dependencies first (layer caching)
# Heavy deps first — cached unless requirements-heavy.txt changes
COPY backend/requirements-heavy.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements-heavy.txt

# App deps — reinstalled on normal requirements.txt changes
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt
# Copy application code
COPY . .
# Create data directories
# Create data directories
RUN mkdir -p /app/data/raw/uploads /app/data/raw/raw_text \
    /app/data/processed /app/data/embeddings/models

# Non-root user for security
RUN useradd -m -u 1000 nyaya && chown -R nyaya:nyaya /app
USER nyaya

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--http", "httptools", \
     "--log-level", "info"]
