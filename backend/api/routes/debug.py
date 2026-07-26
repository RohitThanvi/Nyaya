"""
Debug route — pipeline observability for development.

Exposes a /debug/pipeline endpoint that runs the full pipeline
and returns the complete AgentState: every step's output, timings,
what was retrieved, what the compressor produced, what the verifier flagged.

Disable in production by setting APP_ENVIRONMENT=production.
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base.pipeline import AgentPipeline
from backend.agents.context_compression.agent import ContextCompressionAgent
from backend.api.dependencies.pipeline import get_pipeline
from backend.config.settings import get_settings
from backend.db.session import get_db
from backend.models.domain import SearchRequest

router = APIRouter(prefix="/debug", tags=["debug"])
logger = logging.getLogger(__name__)


def _require_dev():
    if get_settings().app.is_production:
        raise HTTPException(status_code=404, detail="Not found")


@router.post("/pipeline")
async def debug_pipeline(
    request: SearchRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
) -> Dict[str, Any]:
    """
    Run the full pipeline and return the complete internal state.
    Shows exactly what each agent produced at every step.
    """
    _require_dev()

    from backend.models.domain import AgentState, LegalIntentType
    import time

    state = AgentState(original_query=request.query)
    report: Dict[str, Any] = {"query": request.query, "steps": {}}

    # ── Step 1: Query Understanding ─────────────────────────────────────
    t0 = time.perf_counter()
    try:
        qu = await pipeline._qua.understand(
            query=request.query,
            law_filter=request.law_filter,
            court_filter=request.court_filter,
        )
        state.query_understanding = qu
        report["steps"]["1_query_understanding"] = {
            "status": "ok",
            "intent": qu.intent.value,
            "intent_confidence": qu.intent_confidence,
            "law_filter": [l.value for l in qu.law_filter] if qu.law_filter else None,
            "court_filter": [c.value for c in qu.court_filter] if qu.court_filter else None,
            "section_refs": qu.section_refs,
            "legal_entities": [
                {"text": e.text, "type": e.entity_type, "normalized": e.normalized}
                for e in qu.legal_entities
            ],
            "expanded_queries": qu.expanded_queries,
            "year_range": qu.year_range,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        report["steps"]["1_query_understanding"] = {"status": "error", "error": str(e)}

    # ── Step 2: Retrieval ───────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        qu = state.query_understanding
        queries = [request.query] + (qu.expanded_queries[:2] if qu else [])
        if len(queries) > 1:
            chunks, timings = await pipeline._retriever.retrieve_multi_query(
                queries=queries,
                query_understanding=qu,
                top_k_final=get_settings().retrieval.final_context_k,
            )
        else:
            chunks, timings = await pipeline._retriever.retrieve(
                query=request.query,
                query_understanding=qu,
                top_k_final=get_settings().retrieval.final_context_k,
            )
        state.reranked_chunks = chunks
        report["steps"]["2_retrieval"] = {
            "status": "ok",
            "queries_used": queries,
            "chunks_returned": len(chunks),
            "retrieval_timings_ms": {k: round(v, 1) for k, v in timings.items()},
            "chunks": [
                {
                    "rank": i + 1,
                    "chunk_id": rc.chunk.chunk_id[:8],
                    "document_id": rc.chunk.document_id[:8],
                    "chunk_type": rc.chunk.chunk_type.value,
                    "section_ref": rc.chunk.section_ref,
                    "retrieval_source": rc.retrieval_source,
                    "bm25_score": round(rc.bm25_score, 4),
                    "vector_score": round(rc.vector_score, 4),
                    "hybrid_score": round(rc.hybrid_score, 4),
                    "rerank_score": round(rc.rerank_score, 4),
                    "final_score": round(rc.final_score, 4),
                    "citation": rc.chunk.metadata.citation,
                    "law": rc.chunk.metadata.law.value if rc.chunk.metadata.law else None,
                    "page_number": rc.chunk.page_number,
                    "source_url": rc.chunk.metadata.source_url,
                    "content_preview": rc.chunk.content[:200],
                }
                for i, rc in enumerate(chunks)
            ],
        }
    except Exception as e:
        report["steps"]["2_retrieval"] = {"status": "error", "error": str(e)}
        state.reranked_chunks = []

    # ── Step 3: Context Compression ─────────────────────────────────────
    t0 = time.perf_counter()
    try:
        compressor = ContextCompressionAgent()
        context = compressor.compress(
            chunks=state.reranked_chunks,
            max_tokens=3000,
            deduplicate=True,
        )
        state.compressed_context = context
        report["steps"]["3_context_compression"] = {
            "status": "ok",
            "input_chunks": len(state.reranked_chunks),
            "output_chars": len(context),
            "output_tokens_approx": len(context) // 4,
            "context_preview": context[:500] + ("..." if len(context) > 500 else ""),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        report["steps"]["3_context_compression"] = {"status": "error", "error": str(e)}

    # ── Step 4: Generation ──────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        from backend.agents.base.pipeline import _PROMPTS
        intent = (
            state.query_understanding.intent.value
            if state.query_understanding
            else "general_query"
        )
        template = _PROMPTS.get(intent, _PROMPTS["general_query"])
        prompt = template.format(
            context=state.compressed_context or "No context.",
            query=request.query,
        )
        response = await pipeline._llm.complete(prompt=prompt)
        state.raw_llm_response = response
        report["steps"]["4_generation"] = {
            "status": "ok",
            "intent_used": intent,
            "prompt_chars": len(prompt),
            "prompt_tokens_approx": len(prompt) // 4,
            "response_chars": len(response),
            "response": response,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        report["steps"]["4_generation"] = {"status": "error", "error": str(e)}

    # ── Step 5: Verification ────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        verifier = pipeline._make_verifier()
        verified, flags = await verifier.verify(
            llm_response=state.raw_llm_response or "",
            retrieved_chunks=state.reranked_chunks,
            original_query=request.query,
        )
        state.verified_citations = verified
        state.hallucination_flags = flags
        report["steps"]["5_verification"] = {
            "status": "ok",
            "claims_extracted": len(verified) + len(flags),
            "verified_count": len(verified),
            "hallucination_flag_count": len(flags),
            "verified_citations": [
                {
                    "citation_text": c.citation_text,
                    "type": c.citation_type,
                    "section": c.section,
                    "source_url": c.source_url,
                    "page_number": c.page_number,
                    "snippet": c.snippet,
                    "verified": c.verified,
                }
                for c in verified
            ],
            "hallucination_flags": flags,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        report["steps"]["5_verification"] = {"status": "error", "error": str(e)}

    # ── Step 6: Assembly ────────────────────────────────────────────────
    try:
        final_response = pipeline._assembler.assemble(state)
        report["steps"]["6_assembly"] = {
            "status": "ok",
            "confidence": final_response.confidence,
            "warnings_count": len(final_response.warnings),
            "warnings": final_response.warnings,
            "relevant_sections_count": len(final_response.relevant_sections),
            "precedents_count": len(final_response.precedents),
            "citations_count": len(final_response.citations),
        }
        report["final_response"] = {
            "answer": final_response.answer,
            "intent": final_response.intent,
            "confidence": final_response.confidence,
            "citations": [c.model_dump() for c in final_response.citations],
            "relevant_sections": final_response.relevant_sections,
            "precedents": final_response.precedents,
            "warnings": final_response.warnings,
            "hallucination_flags": final_response.hallucination_flags,
        }
    except Exception as e:
        report["steps"]["6_assembly"] = {"status": "error", "error": str(e)}

    # ── Summary ─────────────────────────────────────────────────────────
    total_ms = sum(
        s.get("latency_ms", 0)
        for s in report["steps"].values()
        if isinstance(s, dict)
    )
    report["summary"] = {
        "total_latency_ms": round(total_ms, 1),
        "steps_ok": sum(
            1 for s in report["steps"].values()
            if isinstance(s, dict) and s.get("status") == "ok"
        ),
        "steps_error": sum(
            1 for s in report["steps"].values()
            if isinstance(s, dict) and s.get("status") == "error"
        ),
        "chunks_retrieved": len(state.reranked_chunks),
        "citations_verified": len(state.verified_citations),
        "hallucination_flags": len(state.hallucination_flags),
        "confidence": report.get("final_response", {}).get("confidence", 0),
    }

    return report


@router.get("/chunks")
async def debug_chunks(
    query: str,
    top_k: int = 10,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Run BM25 retrieval only and show raw scores.
    Useful for diagnosing why something is or isn't being retrieved.
    """
    _require_dev()
    from backend.retrieval.bm25.retriever import BM25Retriever
    retriever = BM25Retriever(db=db)
    results = await retriever.search(query=query, top_k=top_k)
    return {
        "query": query,
        "results": [
            {
                "rank": i + 1,
                "chunk_id": rc.chunk.chunk_id[:8],
                "document_id": rc.chunk.document_id[:8],
                "section_ref": rc.chunk.section_ref,
                "chunk_type": rc.chunk.chunk_type.value,
                "bm25_score": round(rc.bm25_score, 6),
                "retrieval_source": rc.retrieval_source,
                "citation": rc.chunk.metadata.citation,
                "law": rc.chunk.metadata.law.value if rc.chunk.metadata.law else None,
                "content_preview": rc.chunk.content[:300],
            }
            for i, rc in enumerate(results)
        ],
    }


@router.get("/document/{document_id}/chunks")
async def debug_document_chunks(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Show all chunks for a document with their types and metadata.
    Diagnose why summarisation gives wrong answers.
    """
    _require_dev()
    from sqlalchemy import text
    result = await db.execute(text("""
        SELECT chunk_id, chunk_type, chunk_index, content_length,
               page_number, section_ref, subsection_ref,
               LEFT(content, 200) AS preview
        FROM chunks
        WHERE document_id = :doc_id
        ORDER BY chunk_index
    """), {"doc_id": document_id})
    rows = result.fetchall()

    type_counts: Dict[str, int] = {}
    for r in rows:
        type_counts[r.chunk_type] = type_counts.get(r.chunk_type, 0) + 1

    return {
        "document_id": document_id,
        "total_chunks": len(rows),
        "chunk_type_distribution": type_counts,
        "chunks": [
            {
                "index": r.chunk_index,
                "type": r.chunk_type,
                "section_ref": r.section_ref,
                "page_number": r.page_number,
                "length": r.content_length,
                "preview": r.preview,
            }
            for r in rows
        ],
    }


@router.get("/db")
async def debug_db(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Database stats — document counts, chunk counts, index health."""
    _require_dev()
    from sqlalchemy import text
    stats = {}
    for table, query in [
        ("documents", "SELECT COUNT(*), COUNT(DISTINCT law), COUNT(DISTINCT court) FROM documents"),
        ("chunks", "SELECT COUNT(*), AVG(content_length)::int, COUNT(DISTINCT document_id) FROM chunks"),
        ("staged_chunks", "SELECT COUNT(*), COUNT(*) FILTER (WHERE indexed=false) FROM staged_chunks"),
        ("users", "SELECT COUNT(*) FROM users"),
    ]:
        try:
            r = (await db.execute(text(query))).fetchone()
            stats[table] = list(r)
        except Exception as e:
            stats[table] = {"error": str(e)}

    # Check tsvector index is populated
    try:
        r = await db.execute(text(
            "SELECT COUNT(*) FROM chunks WHERE content_tsv IS NOT NULL"
        ))
        stats["chunks_with_tsv"] = r.scalar()
    except Exception:
        stats["chunks_with_tsv"] = "error"

    return stats


@router.post("/summarize/{document_id}")
async def debug_summarize(
    document_id: str,
    pipeline: AgentPipeline = Depends(get_pipeline),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Run summarisation and return full intermediate state —
    how many chunks, what types, what each map step produced.
    """
    _require_dev()
    from sqlalchemy import text
    from backend.models.domain import SummarizeRequest

    # Show chunk type distribution first
    result = await db.execute(text("""
        SELECT chunk_type, COUNT(*) as n, SUM(content_length) as total_chars
        FROM chunks WHERE document_id = :doc_id
        GROUP BY chunk_type ORDER BY n DESC
    """), {"doc_id": document_id})
    type_dist = {r.chunk_type: {"count": r.n, "total_chars": r.total_chars}
                 for r in result.fetchall()}

    total_chunks = sum(v["count"] for v in type_dist.values())

    import time
    t0 = time.perf_counter()
    request = SummarizeRequest(document_id=document_id, summary_type="full")
    summary = await pipeline.run_summarize(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "document_id": document_id,
        "chunk_type_distribution": type_dist,
        "total_chunks_processed": total_chunks,
        "latency_ms": elapsed,
        "summary": summary,
    }
