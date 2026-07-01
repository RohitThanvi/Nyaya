"""
Celery application — tuned for 4× RTX 6000 Ada.

Queue topology:
  nyaya_parse          — CPU PDF parsing (32 concurrent workers)
  nyaya_embed_gpu_{0-3}— GPU embedding, one queue per physical GPU.
                         Each worker sets CUDA_VISIBLE_DEVICES=N and
                         only consumes its own queue, so GPU assignment
                         is deterministic and never contended.
  nyaya_flush          — Qdrant batch upsert (single worker, I/O-bound)
  nyaya_dlq            — Dead-letter queue: tasks that exhausted retries
                         land here for manual inspection / replay.

Worker startup (run one command per GPU):
  CUDA_VISIBLE_DEVICES=0 celery -A backend.ingestion.workers.celery_app worker \\
      -Q nyaya_embed_gpu_0 -c 1 --loglevel=info -n embed_gpu0@%h

  CUDA_VISIBLE_DEVICES=1 celery -A backend.ingestion.workers.celery_app worker \\
      -Q nyaya_embed_gpu_1 -c 1 --loglevel=info -n embed_gpu1@%h

  CUDA_VISIBLE_DEVICES=2 celery -A backend.ingestion.workers.celery_app worker \\
      -Q nyaya_embed_gpu_2 -c 1 --loglevel=info -n embed_gpu2@%h

  CUDA_VISIBLE_DEVICES=3 celery -A backend.ingestion.workers.celery_app worker \\
      -Q nyaya_embed_gpu_3 -c 1 --loglevel=info -n embed_gpu3@%h

  celery -A backend.ingestion.workers.celery_app worker \\
      -Q nyaya_parse -c 32 --loglevel=info -n parser@%h

  celery -A backend.ingestion.workers.celery_app worker \\
      -Q nyaya_flush -c 1 --loglevel=info -n flusher@%h
"""
import logging
from celery import Celery
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
_cfg = get_settings().ingestion

celery_app = Celery(
    "nyaya_ingestion",
    broker=_cfg.celery_broker,
    backend=_cfg.celery_backend,
)

# Per-GPU embed queues — deterministic GPU assignment with no contention
GPU_EMBED_QUEUES = [f"nyaya_embed_gpu_{i}" for i in range(4)]

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,

    # Reliability: re-queue on worker crash; reject (not ack) on lost worker
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # One task per worker at a time — GPU workers must not prefetch a second
    # task while the first is still occupying the full GPU memory.
    worker_prefetch_multiplier=1,

    # Result backend TTL: keep task results for 24h for status polling,
    # then expire automatically.
    result_expires=86400,

    # Dead-letter queue: after max_retries exhausted, route to nyaya_dlq
    # instead of silently dropping. Failed tasks can be inspected and
    # replayed with: celery -A ... call nyaya_dlq --args [...]
    task_routes={
        "backend.ingestion.workers.tasks.parse_document":       {"queue": "nyaya_parse"},
        "backend.ingestion.workers.tasks.embed_chunk_batch":     {"queue": "nyaya_embed_gpu_0"},
        "backend.ingestion.workers.tasks.flush_staged":          {"queue": "nyaya_flush"},
        "backend.ingestion.workers.tasks.dead_letter_handler":   {"queue": "nyaya_dlq"},
    },

    # Retry config — exponential backoff up to 5 min
    task_max_retries=3,
    task_default_retry_delay=30,

    beat_schedule={
        "flush-staged-chunks": {
            "task": "backend.ingestion.workers.tasks.flush_staged",
            "schedule": _cfg.flush_interval_s,
        },
    },
)
