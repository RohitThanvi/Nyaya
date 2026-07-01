"""
Semantic query cache.

Legal research queries repeat heavily — "What is the punishment under
Section 302 BNS?" and "punishment for Section 302 under BNS?" are
semantically identical but lexically different, so a simple key-value
cache misses them. This cache embeds every incoming query, then checks
cosine similarity against recent cached queries. A hit (similarity ≥
threshold) returns the cached LegalResponse instantly (~5ms) instead
of running the full retrieve → rerank → verify → assemble pipeline
(~2-5s per query).

At scale with millions of documents and thousands of concurrent users,
this is the single highest-leverage latency optimization — common
legal questions (bail provisions, evidence rules, penal code sections)
are asked hundreds of times per day.

Storage layout in Redis:
  nyaya:scache:queries   — ZSET: { query_embedding_key → timestamp }
  nyaya:scache:emb:{key} — STRING: JSON embedding vector
  nyaya:scache:res:{key} — STRING: JSON LegalResponse
  nyaya:scache:meta:{key}— STRING: JSON query metadata (original text, law_filter)
"""
import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_QUERY_ZSET     = "nyaya:scache:queries"
_EMB_PREFIX     = "nyaya:scache:emb:"
_RES_PREFIX     = "nyaya:scache:res:"
_META_PREFIX    = "nyaya:scache:meta:"
_DEFAULT_TTL    = 3600          # cached results expire after 1 hour
_SIMILARITY_THRESHOLD = 0.94    # cosine sim ≥ 0.94 = treat as same query
_MAX_SCAN       = 200           # check the 200 most recent cached queries


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _query_key(embedding: List[float]) -> str:
    # Use first 16 floats as a fast fingerprint key
    raw = json.dumps(embedding[:16], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class SemanticCache:
    """
    Usage (in the agent pipeline):

        cache = SemanticCache(redis_client)
        query_embedding = await embedder.embed_query(query_text)

        hit = await cache.get(query_embedding, law_filter)
        if hit:
            return hit   # instant response from cache

        result = await pipeline.run_full_rag(...)
        await cache.set(query_embedding, law_filter, query_text, result)
        return result
    """

    def __init__(self, redis):
        self._r = redis

    async def get(
        self,
        query_embedding: List[float],
        law_filter: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return a cached LegalResponse dict if a semantically equivalent
        query exists in cache, or None.
        """
        try:
            # Fetch the most recent _MAX_SCAN keys by score (timestamp)
            keys = await self._r.zrevrange(_QUERY_ZSET, 0, _MAX_SCAN - 1)
            if not keys:
                return None

            # Batch fetch all cached embeddings in one round-trip
            emb_keys = [_EMB_PREFIX + k for k in keys]
            raw_embs = await self._r.mget(*emb_keys)

            for cache_key_bytes, raw_emb in zip(keys, raw_embs):
                if not raw_emb:
                    continue
                cache_key = cache_key_bytes if isinstance(cache_key_bytes, str) else cache_key_bytes.decode()
                cached_emb: List[float] = json.loads(raw_emb)
                sim = _cosine(query_embedding, cached_emb)
                if sim < _SIMILARITY_THRESHOLD:
                    continue

                # Embedding match — check law_filter compatibility
                meta_raw = await self._r.get(_META_PREFIX + cache_key)
                if meta_raw:
                    meta = json.loads(meta_raw)
                    cached_law_filter = meta.get("law_filter") or []
                    incoming_law_filter = law_filter or []
                    # Filters must match exactly (sorted) — a cached "all laws"
                    # result is not appropriate for a law-specific query
                    if sorted(cached_law_filter) != sorted(incoming_law_filter):
                        continue

                res_raw = await self._r.get(_RES_PREFIX + cache_key)
                if res_raw:
                    logger.debug(
                        f"SemanticCache HIT (sim={sim:.3f}, key={cache_key[:8]})"
                    )
                    result = json.loads(res_raw)
                    result["_cache_hit"] = True
                    result["_cache_similarity"] = round(sim, 4)
                    return result

        except Exception as e:
            logger.warning(f"SemanticCache.get failed (non-fatal): {e}")

        return None

    async def set(
        self,
        query_embedding: List[float],
        law_filter: Optional[List[str]],
        query_text: str,
        result: Dict[str, Any],
        ttl: int = _DEFAULT_TTL,
    ) -> None:
        """Store a query result in the semantic cache."""
        try:
            cache_key = _query_key(query_embedding)
            now = time.time()

            pipe = self._r.pipeline()
            pipe.set(_EMB_PREFIX + cache_key, json.dumps(query_embedding), ex=ttl)
            pipe.set(_RES_PREFIX + cache_key, json.dumps(result),          ex=ttl)
            pipe.set(_META_PREFIX + cache_key, json.dumps({
                "query": query_text[:200],
                "law_filter": law_filter or [],
                "cached_at": now,
            }), ex=ttl)
            # Sorted set entry with timestamp as score for ZREVRANGE retrieval
            pipe.zadd(_QUERY_ZSET, {cache_key: now})
            # Expire the ZSET itself so old keys don't accumulate forever
            pipe.expire(_QUERY_ZSET, ttl * 2)
            # Trim to _MAX_SCAN * 2 entries (keep recent + some buffer)
            pipe.zremrangebyrank(_QUERY_ZSET, 0, -((_MAX_SCAN * 2) + 1))
            await pipe.execute()

            logger.debug(f"SemanticCache.set key={cache_key[:8]} query='{query_text[:60]}'")
        except Exception as e:
            logger.warning(f"SemanticCache.set failed (non-fatal): {e}")
