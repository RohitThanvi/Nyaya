"""
Celery application for distributed ingestion pipeline.

Two queues:
  - nyaya_parse  : CPU workers (pdfplumber / PyMuPDF / Tesseract)
  - nyaya_embed  : GPU workers (BGE-large embedding)

Worker startup:
  # Parse workers (CPU, many)
  celery -A backend.ingestion.workers.celery_app worker \
      -Q nyaya_parse -c 8 --loglevel=info -n parser@%h

  # Embed workers (GPU, few)
  celery -A backend.ingestion.workers.celery_app worker \
      -Q nyaya_embed -c 2 --loglevel=info -n embedder@%h

  # Flush worker (scheduled, single)
  celery -A backend.ingestion.workers.celery_app beat \
      --loglevel=info
"""
import logging
import os
from celery import Celery
from celery.schedules import crontab

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
_cfg = get_settings().ingestion

celery_app = Celery(
    "nyaya_ingestion",
    broker=_cfg.celery_broker,
    backend=_cfg.celery_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_acks_late=True,          # re-queue on worker crash
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1, # one task per worker at a time (GPU fairness)
    task_routes={
        "backend.ingestion.workers.tasks.parse_document":  {"queue": "nyaya_parse"},
        "backend.ingestion.workers.tasks.embed_document":  {"queue": "nyaya_embed"},
        "backend.ingestion.workers.tasks.flush_staged":    {"queue": "nyaya_flush"},
    },
    beat_schedule={
        # Flush staging table to Qdrant every 60 seconds
        "flush-staged-chunks": {
            "task": "backend.ingestion.workers.tasks.flush_staged",
            "schedule": _cfg.flush_interval_s,
        },
    },
    # Retry config
    task_max_retries=3,
    task_default_retry_delay=30,
)
