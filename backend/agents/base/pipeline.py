"""
AgentPipeline v3 — zero-hallucination-focused 7-step orchestrator.

Step 1: QueryUnderstanding  — intent + entity extraction
Step 2: Retrieval           — three-path hybrid (exact + BM25 + conditional ANN)
Step 3: LegalMapping        — facts -> statutory sections, NARROWS reranked_chunks
                               to only mapper-validated sections before generation
                               (this step existed in code since v2 but was never
                               actually called anywhere — fixed in v3, see _step_map)
Step 4: ContextCompression  — ContextCompressionAgent; chunk headers now carry a
                               <CHUNK:xxxxxxxx> id the LLM is required to cite
Step 5: Generation          — LLM with grounding-rules prompt requiring every
                               claim to carry its source <CHUNK:id> tag
Step 6: Verification        — VerificationAgent checks <CHUNK:id> tags first
                               (mechanical, exact), falls back to fuzzy match,
                               then DB lookup. v2 had a NameError crash in
                               _verify_judgment that silently disabled judgment
                               citation verification entirely — fixed in v3.
Step 7: Assembly            — AssemblyAgent strips any sentence whose claim
                               failed verification from the visible answer
                               (hard enforcement, not just a warning banner),
                               then strips the internal <CHUNK:id> tags

Changes from v2 (this revision — zero-hallucination hardening):
- FIXED CRASH: _verify_judgment referenced an undefined `chunk` variable
  (copy-paste leftover), silently swallowed by the broad except in
  _step_verify — every judgment citation was unverifiable. Now fixed.
- LegalMappingAgent is now actually called in the request flow (_step_map),
  not just instantiated and ignored.
- Context chunks carry a machine-checkable <CHUNK:id> tag; generation prompts
  require every claim to cite one; verification checks that tag as the
  primary, strongest evidence path before falling back to fuzzy matching.
- AssemblyAgent now REMOVES unverified claims from the answer instead of
  leaving them in place with a warning the user might not read.
"""
import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.assembly.agent import AssemblyAgent
from backend.agents.context_compression.agent import ContextCompressionAgent
from backend.agents.drafting.agent import DraftingAgent
from backend.agents.legal_mapping.agent import LegalMappingAgent
from backend.agents.query_understanding.agent import QueryUnderstandingAgent
from backend.agents.summarization.agent import SummarizationAgent
from backend.agents.verification.agent import VerificationAgent
from backend.config.settings import get_settings
from backend.models.domain import (
    AgentState, ChatRequest, DraftRequest, LegalResponse,
    LegalIntentType, SearchRequest, SummarizeRequest,
)
from backend.retrieval.hybrid.pipeline import HybridRetriever
from backend.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── Prompt templates per intent ──────────────────────────────────────────────

# ── Grounding rules shared by every intent ───────────────────────────────────
#
# Every source chunk in {context} is preceded by a header containing a
# <CHUNK:xxxxxxxx>-style identifier (8 hex chars) embedded in brackets,
# e.g. [CHUNK:a1b2c3d4 | AIR 2025 SC 111 | §318 | BNS | p.4 | [SECTION]].
# The model is required to tag every factual legal claim with the
# <CHUNK:xxxxxxxx> id of the source it came from, placed immediately after
# the claim. This is what VerificationAgent checks mechanically — a claim
# with a tag pointing at a chunk_id that was genuinely retrieved is treated
# as grounded; everything else is fuzzy-matched as a weaker fallback and,
# failing that, stripped from the final answer entirely by AssemblyAgent.
_GROUNDING_RULES = """GROUNDING RULES (mandatory):
1. Use ONLY the legal sources provided below. Do not use outside knowledge of
   Indian law, even if you are confident it is correct — only what appears in
   the sources counts as a source for this answer.
2. Immediately after every section number, citation, or specific legal claim,
   insert the matching <CHUNK:xxxxxxxx> tag copied exactly from that source's
   header. Example: "Section 318 BNS defines cheating <CHUNK:a1b2c3d4>."
3. If the sources do not contain information needed to answer part of the
   question, explicitly say so in that part of your answer instead of filling
   the gap from memory. Do not guess section numbers.
4. Never state a section number, case citation, or punishment that does not
   appear verbatim in the sources below.
"""

_PROMPTS: Dict[str, str] = {
    LegalIntentType.PROVISION_LOOKUP.value: """You are NyayaAI, an expert Indian legal assistant.

""" + _GROUNDING_RULES + """
LEGAL SOURCES:
{context}

QUESTION: {query}

Provide a clear, accurate answer with all relevant section numbers and their exact text, each tagged with its source chunk.""",

    LegalIntentType.CASE_SEARCH.value: """You are NyayaAI, an expert Indian legal assistant.

""" + _GROUNDING_RULES + """
JUDGMENTS:
{context}

QUERY: {query}

Summarise the relevant holdings and their applicability to the query, tagging every citation with its source chunk.""",

    LegalIntentType.PROCEDURE_QUERY.value: """You are NyayaAI, an expert Indian legal assistant.

""" + _GROUNDING_RULES + """
LEGAL SOURCES:
{context}

QUERY: {query}

Provide a numbered step-by-step procedure with statutory references, each step tagged with its source chunk.""",

    LegalIntentType.GENERAL_QUERY.value: """You are NyayaAI, an expert Indian legal assistant.

""" + _GROUNDING_RULES + """
SOURCES:
{context}

QUERY: {query}

Provide a helpful, accurate answer using only the sources above.""",

    LegalIntentType.SUMMARIZATION.value: """You are NyayaAI, an expert Indian legal assistant.
Provide a structured summary of this judgment.

JUDGMENT EXCERPTS:
{context}

QUERY: {query}

Structure: Facts | Issues | Holdings | Ratio | Final Order""",

    LegalIntentType.DRAFTING_REQUEST.value: """You are NyayaAI, an expert Indian legal assistant.
{context}

DRAFTING QUERY: {query}

Complete the draft with accurate legal language. Mark any field requiring advocate verification with [VERIFY: reason].""",
}


class AgentPipeline:
    """
    Orchestrates all agents and retrieval for a single user query.
    Stateless per-request — instantiate once at startup, call per request.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        db: AsyncSession,
    ):
        self._retriever       = retriever
        self._llm             = llm_client
        self._db              = db
        self._settings        = get_settings()
        self._ret_cfg         = self._settings.retrieval

        # Agent instances
        self._qua             = QueryUnderstandingAgent(llm_client=llm_client)
        self._mapper          = LegalMappingAgent(llm_client=llm_client)
        self._compressor      = ContextCompressionAgent()
        self._summarizer      = SummarizationAgent(llm_client=llm_client)
        self._drafting        = DraftingAgent(llm_client=llm_client)
        self._assembler       = AssemblyAgent()

    def _make_verifier(self) -> VerificationAgent:
        """VerificationAgent is created per-request with live db session."""
        return VerificationAgent(db=self._db, llm_client=self._llm)

    # ──────────────────────────────────────────────────────────────────────
    # Main entry points
    # ──────────────────────────────────────────────────────────────────────

    async def run_search(self, request: SearchRequest) -> LegalResponse:
        state = AgentState(
            original_query=request.query,
        )
        await self._step_understand(state, law_filter=request.law_filter,
                                    court_filter=request.court_filter)
        await self._step_retrieve(state)
        await self._step_map(state)
        await self._step_compress(state)
        await self._step_generate(state)
        await self._step_verify(state)
        return self._assembler.assemble(state)

    async def run_chat(
        self, request: ChatRequest, user_id: Optional[str] = None
    ) -> LegalResponse:
        state = AgentState(
            original_query=request.message,
            user_id=user_id,
        )

        # Semantic cache check — skip for document-scoped queries (those are
        # already fast since they target one document's chunk set) and for
        # queries with explicit history (cache should not collapse multi-turn
        # conversation into a single-turn cached answer).
        _cache = None
        _query_embedding: Optional[List[float]] = None
        if not request.document_id and not request.history:
            try:
                from backend.utils.redis_client import get_redis_client
                from backend.utils.semantic_cache import SemanticCache
                redis = await get_redis_client()
                _cache = SemanticCache(redis)
                _query_embedding = await self._embedder.embed_query(request.message)
                cached = await _cache.get(
                    _query_embedding,
                    law_filter=[f.value if hasattr(f, "value") else f
                                for f in (request.law_filter or [])],
                )
                if cached:
                    logger.info(
                        f"SemanticCache HIT (sim={cached.get('_cache_similarity', '?')}) "
                        f"for query: {request.message[:80]}"
                    )
                    from backend.models.domain import LegalResponse as LR
                    cached.pop("_cache_hit", None)
                    cached.pop("_cache_similarity", None)
                    return LR(**cached)
            except Exception as e:
                logger.warning(f"Semantic cache lookup failed (continuing without cache): {e}")
                _cache = None

        # Document-scoped retrieval
        if request.document_id:
            t0 = time.perf_counter()
            chunks = await self._retriever.retrieve_for_document(
                document_id=request.document_id,
                query=request.message,
                top_k=self._ret_cfg.final_context_k * 2,
            )
            state.reranked_chunks = chunks
            state.latency_ms["doc_retrieval_ms"] = (time.perf_counter() - t0) * 1000
        else:
            await self._step_understand(state, law_filter=request.law_filter)
            await self._step_retrieve(state)
            await self._step_map(state)

        await self._step_compress(state)
        await self._step_generate(state, history=request.history)
        await self._step_verify(state)
        result = self._assembler.assemble(state)

        # Store in semantic cache for future identical/near-identical queries
        if _cache and _query_embedding and not request.document_id:
            try:
                await _cache.set(
                    query_embedding=_query_embedding,
                    law_filter=[f.value if hasattr(f, "value") else f
                                for f in (request.law_filter or [])],
                    query_text=request.message,
                    result=result.model_dump(),
                )
            except Exception as e:
                logger.warning(f"Semantic cache store failed (non-fatal): {e}")

        return result

    async def run_chat_stream(
        self, request: ChatRequest, user_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        SSE streaming: runs understand + retrieve + compress synchronously,
        then streams LLM generation token-by-token.
        Verification runs after stream completes (non-blocking metadata append).
        """
        state = AgentState(original_query=request.message, user_id=user_id)

        if request.document_id:
            chunks = await self._retriever.retrieve_for_document(
                document_id=request.document_id,
                query=request.message,
                top_k=self._ret_cfg.final_context_k * 2,
            )
            state.reranked_chunks = chunks
        else:
            await self._step_understand(state, law_filter=request.law_filter)
            await self._step_retrieve(state)
            await self._step_map(state)

        await self._step_compress(state)

        intent = (
            state.query_understanding.intent.value
            if state.query_understanding
            else LegalIntentType.GENERAL_QUERY.value
        )
        prompt_template = _PROMPTS.get(intent, _PROMPTS[LegalIntentType.GENERAL_QUERY.value])
        prompt = prompt_template.format(
            context=state.compressed_context or "",
            query=request.message,
        )

        full_response = ""
        async for token in self._llm.stream(prompt=prompt):
            full_response += token
            yield token

        state.raw_llm_response = full_response

        # Async post-stream verification (fire-and-forget for latency)
        try:
            verifier = self._make_verifier()
            verified, flags = await verifier.verify(
                llm_response=full_response,
                retrieved_chunks=state.reranked_chunks,
                original_query=request.message,
            )
            state.verified_citations = verified
            state.hallucination_flags = flags
        except Exception as e:
            logger.warning(f"Post-stream verification failed: {e}")

    async def run_draft(self, request: DraftRequest) -> LegalResponse:
        state = AgentState(
            original_query=f"Draft {request.draft_type.value} for: {request.facts[:200]}"
        )
        draft_text = await self._drafting.generate_draft(request)
        state.raw_llm_response = draft_text
        draft_confidence = 0.45
        return LegalResponse(
            query=state.original_query,
            session_id=state.session_id,
            intent=LegalIntentType.DRAFTING_REQUEST.value,
            answer=draft_text,
            confidence=draft_confidence,
            confidence_breakdown={
                "band": "medium",
                "retrieval_quality": 0.0,
                "chunks_used": 0,
                "source_diversity": 0.0,
                "unique_documents": 0,
                "verification_ratio": 0.0,
                "verified_claims": 0,
                "flagged_claims": 0,
            },
            warnings=[
                "This draft must be reviewed and verified by a qualified advocate "
                "before filing or use in any legal proceeding.",
                "Confidence score reflects that no source verification was performed "
                "on this draft — advocate review is mandatory.",
            ],
        )

    async def run_summarize(
        self, request: SummarizeRequest, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full hierarchical summarisation.
        Fetches ALL chunks (no LIMIT), uses typed chunk routing.
        """
        from sqlalchemy import text as sql_text
        from backend.models.domain import (
            LegalChunk, ChunkType, DocumentMetadata, DocumentType, LawCategory
        )

        document_id = request.document_id
        if not document_id:
            return {"error": "document_id required for summarisation"}

        # Fetch document metadata
        meta_result = await self._db.execute(sql_text("""
            SELECT document_type, law, court, court_name, case_number,
                   citation, year, date_decided, bench, parties, topic,
                   keywords, source_url, is_landmark, language
            FROM documents
            WHERE document_id = :doc_id
        """), {"doc_id": document_id})
        meta_row = meta_result.fetchone()

        if not meta_row:
            return {"error": f"Document {document_id} not found"}

        # Fetch ALL chunks — no LIMIT
        chunk_result = await self._db.execute(sql_text("""
            SELECT chunk_id, chunk_type, content, content_length,
                   chunk_index, page_number, section_ref, subsection_ref
            FROM chunks
            WHERE document_id = :doc_id
            ORDER BY chunk_index
        """), {"doc_id": document_id})
        chunk_rows = chunk_result.fetchall()

        if not chunk_rows:
            return {"error": f"No chunks found for document {document_id}"}

        doc_meta = DocumentMetadata(
            document_id=document_id,
            document_type=DocumentType(meta_row.document_type),
            law=LawCategory(meta_row.law) if meta_row.law else None,
            court_name=meta_row.court_name,
            case_number=meta_row.case_number,
            citation=meta_row.citation,
            year=meta_row.year,
            source_url=meta_row.source_url,
            is_landmark=bool(meta_row.is_landmark),
            language=meta_row.language or "en",
        )

        chunks = [
            LegalChunk(
                chunk_id=str(r.chunk_id),
                document_id=document_id,
                chunk_type=ChunkType(r.chunk_type) if r.chunk_type else ChunkType.PASSAGE,
                content=r.content,
                content_length=r.content_length or len(r.content),
                chunk_index=r.chunk_index,
                page_number=r.page_number,
                section_ref=r.section_ref,
                subsection_ref=r.subsection_ref,
                metadata=doc_meta,
            )
            for r in chunk_rows
        ]

        logger.info(f"Summarising {len(chunks)} chunks for document {document_id}")

        # Typed chunk routing via summarize_chunks()
        summary = await self._summarizer.summarize_chunks(
            chunks=chunks,
            document_id=document_id,
            metadata={
                "case_name": meta_row.case_number,
                "citation": meta_row.citation,
                "court_name": meta_row.court_name,
                "is_landmark": bool(meta_row.is_landmark),
            },
        )
        return summary.model_dump()

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline steps
    # ──────────────────────────────────────────────────────────────────────

    async def _step_understand(
        self, state: AgentState, **kwargs
    ):
        t0 = time.perf_counter()
        try:
            qu = await self._qua.understand(
                query=state.original_query, **kwargs
            )
            state.query_understanding = qu
            state.pipeline_trace.append({"step": "understand", "intent": qu.intent.value})
        except Exception as e:
            logger.error(f"QueryUnderstanding failed: {e}")
            state.error = str(e)
        state.latency_ms["understand_ms"] = (time.perf_counter() - t0) * 1000

    async def _step_retrieve(self, state: AgentState):
        t0 = time.perf_counter()
        try:
            qu = state.query_understanding
            queries = [state.original_query]
            if qu and qu.expanded_queries:
                queries = [state.original_query] + qu.expanded_queries[:2]

            if len(queries) > 1:
                chunks, timings = await self._retriever.retrieve_multi_query(
                    queries=queries,
                    query_understanding=qu,
                    top_k_final=self._ret_cfg.final_context_k,
                )
            else:
                chunks, timings = await self._retriever.retrieve(
                    query=state.original_query,
                    query_understanding=qu,
                    top_k_final=self._ret_cfg.final_context_k,
                )

            state.reranked_chunks = chunks
            state.latency_ms.update(timings)
            state.pipeline_trace.append({
                "step": "retrieve",
                "chunks_retrieved": len(chunks),
                "timings": timings,
            })
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state.error = str(e)
            state.reranked_chunks = []
        state.latency_ms["retrieve_total_ms"] = (time.perf_counter() - t0) * 1000

    async def _step_map(self, state: AgentState):
        """
        Pre-generation grounding gate using LegalMappingAgent.

        Was previously instantiated but never called anywhere in the real
        request flow — its hallucination-stripping logic
        (_validate_mapping_result) existed but had zero effect on what the
        LLM actually generated. Now wired in for real:

        1. Runs only for intents where facts->sections reasoning applies
           (provision_lookup, procedure_query, drafting_request) — skipped
           for case_search/summarization/general_query where it doesn't fit.
        2. The mapper validates every section it proposes against
           reranked_chunks's actual chunk_ids and section_refs, discarding
           anything it can't trace back to a real retrieved chunk.
        3. reranked_chunks is then NARROWED to only the chunks whose
           section_ref survived validation — never widened. If mapping
           fails or finds nothing, the original retrieval set is used
           unchanged: a broken mapper degrades to "no extra filtering",
           never to "can't answer at all".
        """
        qu = state.query_understanding
        applicable_intents = {
            LegalIntentType.PROVISION_LOOKUP.value,
            LegalIntentType.PROCEDURE_QUERY.value,
            LegalIntentType.DRAFTING_REQUEST.value,
        }
        intent = qu.intent.value if qu else LegalIntentType.GENERAL_QUERY.value
        if intent not in applicable_intents or not state.reranked_chunks:
            return

        t0 = time.perf_counter()
        try:
            mapping_result = await self._mapper.map_facts_to_sections(
                facts=state.original_query,
                retrieved_chunks=state.reranked_chunks,
                query_understanding=qu,
            )
            state.mapped_sections = mapping_result

            validated_refs = {
                str(s.get("section_number", "")).strip().lower()
                for s in mapping_result.get("relevant_sections", [])
                if s.get("section_number")
            }

            if validated_refs:
                narrowed = [
                    rc for rc in state.reranked_chunks
                    if (rc.chunk.section_ref or "").strip().lower() in validated_refs
                    or not rc.chunk.section_ref   # keep non-statute chunks (judgments etc.)
                ]
                if narrowed:
                    state.reranked_chunks = narrowed

            state.pipeline_trace.append({
                "step": "map",
                "sections_validated": len(mapping_result.get("relevant_sections", [])),
                "hallucinated_sections_stripped": len(
                    [w for w in mapping_result.get("warnings", []) if "not found in retrieved" in w]
                ),
            })
        except Exception as e:
            logger.error(f"Legal mapping failed (continuing with unfiltered chunks): {e}")
        state.latency_ms["map_ms"] = (time.perf_counter() - t0) * 1000

    async def _step_compress(self, state: AgentState):
        t0 = time.perf_counter()
        try:
            state.compressed_context = self._compressor.compress(
                chunks=state.reranked_chunks,
                max_tokens=self._settings.ingestion.summary_window_chars // 4,
                deduplicate=True,
            )
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            state.compressed_context = "\n\n".join(
                rc.chunk.content[:500] for rc in state.reranked_chunks[:6]
            )
        state.latency_ms["compress_ms"] = (time.perf_counter() - t0) * 1000

    async def _step_generate(
        self,
        state: AgentState,
        history: Optional[List] = None,
    ):
        t0 = time.perf_counter()
        try:
            intent = (
                state.query_understanding.intent.value
                if state.query_understanding
                else LegalIntentType.GENERAL_QUERY.value
            )
            template = _PROMPTS.get(intent, _PROMPTS[LegalIntentType.GENERAL_QUERY.value])
            context = state.compressed_context or "No relevant sources found."
            prompt = template.format(context=context, query=state.original_query)

            history_text = ""
            if history:
                history_text = "\n".join(
                    f"{m.role.upper()}: {m.content}" for m in history[-6:]
                )
                prompt = f"CONVERSATION HISTORY:\n{history_text}\n\n{prompt}"

            response = await self._llm.complete(prompt=prompt)
            state.raw_llm_response = response
            state.pipeline_trace.append({"step": "generate", "tokens": len(response) // 4})
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            state.raw_llm_response = (
                "I encountered an error while generating a response. "
                "Please try again or rephrase your question."
            )
        state.latency_ms["generate_ms"] = (time.perf_counter() - t0) * 1000

    async def _step_verify(self, state: AgentState):
        t0 = time.perf_counter()
        try:
            verifier = self._make_verifier()
            verified, flags = await verifier.verify(
                llm_response=state.raw_llm_response or "",
                retrieved_chunks=state.reranked_chunks,
                original_query=state.original_query,
            )
            state.verified_citations = verified
            state.hallucination_flags = flags
            state.pipeline_trace.append({
                "step": "verify",
                "verified": len(verified),
                "flags": len(flags),
            })
        except Exception as e:
            logger.error(f"Verification failed: {e}")
        state.latency_ms["verify_ms"] = (time.perf_counter() - t0) * 1000
        state.latency_ms["total_ms"] = sum(
            v for k, v in state.latency_ms.items()
            if k not in {"total_ms", "retrieve_total_ms"}
        )
