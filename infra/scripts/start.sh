#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════╗"
echo "║         NyayaAI — Local Setup        ║"
echo "╚══════════════════════════════════════╝"

# Check .env exists
if [ ! -f .env ]; then
  echo "→ Creating .env from .env.example..."
  cp .env.example .env
  echo "  ⚠  Edit .env and set LLM_GROQ_API_KEY before starting"
fi

# Check Docker
if ! command -v docker &> /dev/null; then
  echo "✗ Docker not found. Install from https://docker.com"
  exit 1
fi

echo "→ Starting infrastructure services..."
docker compose up -d postgres qdrant redis
echo "  Waiting for services to be healthy..."
sleep 8

echo "→ Running database migrations..."
docker compose run --rm backend python -m alembic upgrade head 2>/dev/null || \
  echo "  ℹ  Migrations: tables will be created on first startup"

echo "→ Starting backend and frontend..."
docker compose up -d backend frontend nginx

echo ""
echo "✓ NyayaAI is starting up"
echo ""
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/api/docs"
echo "  Qdrant:    http://localhost:6333/dashboard"
echo ""
echo "→ Run ingestion to load BNS/BNSS/BSA:"
echo "  docker compose exec backend python -m backend.ingestion.pipeline.cli --help"
echo ""
echo "→ View logs:"
echo "  docker compose logs -f backend"
