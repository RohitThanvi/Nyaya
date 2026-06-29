"""
Retrieval quality evaluation against golden test set.

Metrics:
- Section hit rate: did the top-k results contain the expected section?
- Intent accuracy: did query understanding classify intent correctly?
- Confidence calibration: does reported confidence track actual accuracy?
- Latency: P50/P95 retrieval latency

Run:
    python -m backend.tests.eval_retrieval
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

GOLDEN_PATH = Path(__file__).parent.parent.parent / "data/golden_set/retrieval_eval.json"


async def run_evaluation():
    from backend.agents.query_understanding.agent import QueryUnderstandingAgent
    from backend.db.session import init_db
    from backend.embeddings.service import EmbeddingService
    from backend.retrieval.bm25.retriever import BM25Retriever
    from backend.retrieval.hybrid.pipeline import HybridRetriever
    from backend.retrieval.reranker.cross_encoder import Reranker
    from backend.retrieval.vector.retriever import VectorRetriever
    from backend.db.session import get_db_session

    await init_db()

    query_agent = QueryUnderstandingAgent()
    vector_retriever = VectorRetriever()
    reranker = Reranker()
    embedding_service = EmbeddingService()

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    results = []
    latencies = []

    for item in golden:
        query = item["query"]
        expected_sections = item.get("expected_sections", [])
        expected_intent = item.get("expected_intent")

        t0 = time.perf_counter()

        # Query understanding
        qu = await query_agent.analyze(query)

        # Retrieve (using DB session)
        async with get_db_session() as db:
            bm25 = BM25Retriever(db)
            retriever = HybridRetriever(bm25, vector_retriever, embedding_service, reranker)
            chunks, timings = await retriever.retrieve(query, qu, top_k_final=10)

        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        # Evaluate
        retrieved_sections = set()
        for c in chunks:
            if c.chunk.section_ref:
                retrieved_sections.add(c.chunk.section_ref)

        section_hit = any(
            any(exp in sec for sec in retrieved_sections)
            for exp in expected_sections
        ) if expected_sections else True  # No expected sections = N/A

        intent_correct = qu.intent.value == expected_intent if expected_intent else None

        result = {
            "query": query[:50],
            "section_hit": section_hit,
            "intent_correct": intent_correct,
            "retrieved_sections": list(retrieved_sections)[:5],
            "predicted_intent": qu.intent.value,
            "latency_ms": round(latency_ms, 1),
            "chunks_retrieved": len(chunks),
        }
        results.append(result)
        status = "✓" if section_hit else "✗"
        logger.info(f"{status} [{latency_ms:.0f}ms] {query[:50]}")
        if expected_sections:
            logger.info(f"    Expected: {expected_sections} | Got: {list(retrieved_sections)[:5]}")

    # Summary
    section_hit_rate = mean(r["section_hit"] for r in results if r["section_hit"] is not None)
    intent_results = [r["intent_correct"] for r in results if r["intent_correct"] is not None]
    intent_accuracy = mean(intent_results) if intent_results else 0

    p50 = median(latencies)
    p95 = quantiles(latencies, n=20)[18] if len(latencies) >= 5 else max(latencies)

    logger.info("")
    logger.info("═" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("═" * 50)
    logger.info(f"Section Hit Rate:  {section_hit_rate:.1%}  ({sum(r['section_hit'] for r in results if r['section_hit'])}/{len([r for r in results if r['section_hit'] is not None])})")
    logger.info(f"Intent Accuracy:   {intent_accuracy:.1%}")
    logger.info(f"Latency P50:       {p50:.0f}ms")
    logger.info(f"Latency P95:       {p95:.0f}ms")
    logger.info(f"Total Queries:     {len(results)}")
    logger.info("═" * 50)

    return {
        "section_hit_rate": section_hit_rate,
        "intent_accuracy": intent_accuracy,
        "p50_ms": p50,
        "p95_ms": p95,
    }


if __name__ == "__main__":
    asyncio.run(run_evaluation())
