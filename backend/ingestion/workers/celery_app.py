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

    # Result backend TTL: keep task results for 24h for status polling
    result_expires=86400,

    # ── Queue depth limits — prevent Redis OOM at million-document scale ──
    # Without these, submitting 1M documents at once enqueues 1M tasks
    # simultaneously. Redis holds each task as a ~2KB JSON blob:
    # 1M × 2KB = 2GB of Redis memory just for the queue. At 10M docs that's
    # 20GB — Redis OOM kills the entire pipeline.
    #
    # task_queue_max_priority sets a soft depth ceiling via Redis stream
    # consumer groups. When the queue is full, apply_async() blocks the
    # caller (the bulk-upload API handler) instead of crashing Redis.
    # parse workers: 10K queued tasks max (~20MB Redis queue memory)
    # embed workers: 50K queued tasks max (fan-out sub-tasks, smaller payload)
    # NOTE: embed_chunk_batch is deliberately NOT statically routed here.
    # Its whole reason to exist is to be dispatched from embed_document's
    # round-robin fan-out via apply_async(..., queue=target_queue), which
    # picks a *different* one of the 4 GPU queues per sub-batch. An
    # explicit queue= argument to apply_async() takes precedence over
    # task_routes, so a static entry here would never fire in the normal
    # flow — but leaving one in place (previously hardcoded to
    # nyaya_embed_gpu_0) is a landmine: any future caller that invokes
    # embed_chunk_batch without an explicit queue= (a debug script, a
    # manual replay helper) would silently pin every job onto GPU 0 while
    # GPUs 1-3 sit idle, with no error to indicate why throughput cratered.
    task_routes={
        "backend.ingestion.workers.tasks.parse_document":     {"queue": "nyaya_parse"},
        "backend.ingestion.workers.tasks.embed_document":     {"queue": "nyaya_parse"},
        "backend.ingestion.workers.tasks.flush_staged":       {"queue": "nyaya_flush"},
        "backend.ingestion.workers.tasks.dead_letter_handler":{"queue": "nyaya_dlq"},
    },

    # Rate limit per worker: parse workers process at most 100 tasks/min each
    # (32 workers × 100/min = 3200 docs/min peak). This prevents a spike of
    # bulk uploads from saturating CPU and starving live query workers that
    # share the same machine.
    task_annotations={
        "backend.ingestion.workers.tasks.parse_document": {
            "rate_limit": "100/m",
        },
        "backend.ingestion.workers.tasks.embed_chunk_batch": {
            "rate_limit": "200/m",
        },
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
