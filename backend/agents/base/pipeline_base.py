"""
Main Agent Pipeline Orchestrator.

Deterministic pipeline — not autonomous agents.
Each step is explicit, ordered, and traceable.

Pipeline:
1. Query Understanding
2. Hybrid Retrieval (BM25 + ANN + Rerank)
3. Legal Mapping (facts → sections)
4. Context Compression
5. LLM Generation
6. Citation Verification
7. Structured Output Assembly
"""
import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from backend.agents.drafting.agent import DraftingAgent
from backend.agents.legal_mapping.agent import LegalMappingAgent
from backend.agents.query_understanding.agent import QueryUnderstandingAgent
from backend.agents.summarization.agent import SummarizationAgent
from backend.agents.verification.agent import VerificationAgent
from backend.config.settings import get_settings
from backend.embeddings.service import EmbeddingService
from backend.models.domain import (
    AgentState, ChatMessage, DraftRequest, DraftType,
    JudgmentSummary, LegalIntentType, LegalResponse,
    QueryUnderstanding, RetrievedChunk, SearchRequest,
    SummarizeRequest
)
from backend.retrieval.bm25.retriever import BM25Retriever
from backend.retrieval.hybrid.pipeline import HybridRetriever, compress_context
from backend.retrieval.reranker.cross_encoder import Reranker
from backend.retrieval.vector.retriever import VectorRetriever
from backend.utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are NyayaAI, an expert Indian legal research assistant.

You specialize in:
- Bharatiya Nyaya Sanhita (BNS) 2023
- Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023
- Bharatiya Sakshya Adhiniyam (BSA) 2023
- Supreme Court and High Court judgments

CRITICAL RULES:
1. Answer ONLY from the provided context. Do not use outside knowledge for legal provisions.
2. Every legal claim MUST cite its source: [Section X, Law] or [Citation, Court, Year]
3. If the context does not contain enough information, explicitly say:
   "The retrieved context does not contain sufficient information to answer this conclusively."
4. Never invent section numbers, case names, or holdings.
5. Use clear, precise legal language but explain technical terms.
6. Structure your answer: (a) Direct answer, (b) Relevant provisions, (c) Procedural notes if applicable.

CONTEXT:
{context}"""


class AgentPipeline:
    """
    Core orchestration pipeline.
    All agents are injected for testability.
    """

    def __init__(
        self,
        query_agent: QueryUnderstandingAgent,
        hybrid_retriever: HybridRetriever,
        legal_mapping_agent: LegalMappingAgent,
        verification_agent: VerificationAgent,
        summarization_agent: SummarizationAgent,
        drafting_agent: DraftingAgent,
        llm_client=None,
    ):
        self._query_agent = query_agent
        self._retriever = hybrid_retriever
        self._mapping_agent = legal_mapping_agent
        self._verification_agent = verification_agent
        self._summarization_agent = summarization_agent
        self._drafting_agent = drafting_agent
        self._llm = llm_client or get_llm_client()
        self._settings = get_settings()

    def _trace(self, state: AgentState, step: str, data: Dict) -> None:
        state.pipeline_trace.append({"step": step, "data": data})

    async def run_search(self, request: SearchRequest, user_id: Optional[str] = None) -> LegalResponse:
        """
        Full search pipeline:
        understand → retrieve → map → verify → respond
        """
        t_start = time.perf_counter()
        state = AgentState(original_query=request.query, user_id=user_id)

        # Step 1: Query Understanding
        t0 = time.perf_counter()
        qu = await self._query_agent.analyze(request.query)
        # Apply API-level filters (override LLM filters if explicitly set)
        if request.law_filter:
            qu.law_filter = request.law_filter
        if request.court_filter:
            qu.court_filter = request.court_filter
        if request.year_from or request.year_to:
            qu.year_range = {
                "from": request.year_from or 1950,
                "to": request.year_to or 2025,
            }
        state.query_understanding = qu
        state.latency_ms["query_understanding"] = (time.perf_counter() - t0) * 1000
        self._trace(state, "query_understanding", {"intent": qu.intent, "entities": len(qu.legal_entities)})

        # Step 2: Hybrid Retrieval
        t0 = time.perf_counter()
        queries = [request.query] + (qu.expanded_queries or [])
        if len(queries) > 1:
            chunks, ret_timings = await self._retriever.retrieve_multi_query(
                queries=queries,
                query_understanding=qu,
                top_k_final=request.top_k,
            )
        else:
            chunks, ret_timings = await self._retriever.retrieve(
                query=request.query,
                query_understanding=qu,
                top_k_final=request.top_k,
            )
        state.reranked_chunks = chunks
        state.latency_ms.update(ret_timings)
        self._trace(state, "retrieval", {"chunks_retrieved": len(chunks)})

        if not chunks:
            return LegalResponse(
                query=request.query,
                session_id=state.session_id,
                intent=qu.intent.value,
                answer="No relevant legal provisions or judgments found for your query. Please refine your search terms or check the spelling of section numbers.",
                confidence=0.0,
                warnings=["No results found in knowledge base."],
                latency_ms=(time.perf_counter() - t_start) * 1000,
            )

        # Step 3: Legal Mapping (for provision lookups and case queries)
        mapping_result = None
        if qu.intent in (LegalIntentType.PROVISION_LOOKUP, LegalIntentType.CASE_SEARCH,
                         LegalIntentType.PROCEDURE_QUERY):
            t0 = time.perf_counter()
            mapping_result = await self._mapping_agent.map_facts_to_sections(
                facts=request.query,
                retrieved_chunks=chunks,
                query_understanding=qu,
            )
            state.latency_ms["legal_mapping"] = (time.perf_counter() - t0) * 1000
            self._trace(state, "legal_mapping", {
                "sections_found": len(mapping_result.get("relevant_sections", []))
            })

        # Step 4: Context Compression
        compressed = compress_context(chunks, max_tokens=3000)
        state.compressed_context = compressed

        # Step 5: LLM Generation
        t0 = time.perf_counter()
        messages = self._build_chat_messages(request.query, compressed, [])
        raw_response = await self._llm.complete(messages)
        state.raw_llm_response = raw_response
        state.latency_ms["llm_generation"] = (time.perf_counter() - t0) * 1000

        # Step 6: Verification
        t0 = time.perf_counter()
        flags, verified_citations, verifiability = await self._verification_agent.verify(
            llm_response=raw_response,
            retrieved_chunks=chunks,
            mapping_result=mapping_result,
        )
        state.hallucination_flags = flags
        state.verified_citations = verified_citations
        state.latency_ms["verification"] = (time.perf_counter() - t0) * 1000

        # Step 7: Assemble response
        avg_rerank_score = sum(c.final_score for c in chunks[:5]) / min(5, len(chunks))
        confidence = self._verification_agent.confidence_from_verifiability(
            verifiability, avg_rerank_score
        )

        relevant_sections = []
        precedents = []
        procedural_requirements = []

        if mapping_result:
            relevant_sections = mapping_result.get("relevant_sections", [])
            procedural_requirements = mapping_result.get("procedural_requirements", [])

        # Separate judgments from statutes for response
        for chunk in chunks[:6]:
            meta = chunk.chunk.metadata
            if meta.document_type.value == "judgment" and meta.citation:
                precedents.append({
                    "citation": meta.citation,
                    "court": meta.court_name or (meta.court.value if meta.court else ""),
                    "year": meta.year,
                    "relevance": chunk.chunk.content[:200] + "...",
                    "score": round(chunk.final_score, 3),
                })

        state.latency_ms["total"] = (time.perf_counter() - t_start) * 1000

        return LegalResponse(
            query=request.query,
            session_id=state.session_id,
            intent=qu.intent.value,
            answer=raw_response,
            relevant_sections=relevant_sections,
            precedents=precedents[:5],
            procedural_requirements=procedural_requirements,
            citations=verified_citations,
            confidence=round(confidence, 3),
            warnings=flags if flags else [],
            hallucination_flags=flags,
            latency_ms=round(state.latency_ms.get("total", 0), 1),
        )

    async def run_chat(
        self,
        message: str,
        history: List[ChatMessage],
        user_id: Optional[str] = None,
        law_filter=None,
        document_id: Optional[str] = None,
    ) -> LegalResponse:
        """Chat pipeline with conversation history."""
        t_start = time.perf_counter()
        state = AgentState(original_query=message, user_id=user_id)

        # Understand query
        qu = await self._query_agent.analyze(message)
        if law_filter:
            qu.law_filter = law_filter
        state.query_understanding = qu

        # If scoped to a document, fetch its chunks directly from DB
        if document_id:
            chunks = await self._fetch_document_chunks(document_id, message)
        else:
            chunks, timings = await self._retriever.retrieve(
                query=message,
                query_understanding=qu,
                top_k_final=self._settings.retrieval.final_context_k,
            )
        state.reranked_chunks = chunks

        if not chunks:
            doc_context = f" in document '{document_id}'" if document_id else ""
            return LegalResponse(
                query=message,
                session_id=state.session_id,
                answer=f"I could not find relevant content{doc_context} matching your query. "
                       f"Try asking about specific people, sections, dates, or legal issues mentioned in the document. "
                       f"For example: 'Who filed suit 5?' or 'What was the court's decision on title?' or 'What sections were discussed?'",
                confidence=0.1,
                warnings=["No relevant context found — try more specific terms."],
                latency_ms=(time.perf_counter() - t_start) * 1000,
            )

        compressed = compress_context(chunks, max_tokens=2500)

        # Build messages with history
        messages = self._build_chat_messages(message, compressed, history)
        raw_response = await self._llm.complete(messages)

        flags, verified_citations, verifiability = await self._verification_agent.verify(
            raw_response, chunks
        )

        avg_score = sum(c.final_score for c in chunks[:3]) / min(3, len(chunks))
        confidence = self._verification_agent.confidence_from_verifiability(verifiability, avg_score)

        return LegalResponse(
            query=message,
            session_id=state.session_id,
            intent=qu.intent.value,
            answer=raw_response,
            citations=verified_citations,
            confidence=round(confidence, 3),
            warnings=flags,
            latency_ms=round((time.perf_counter() - t_start) * 1000, 1),
        )

    async def stream_chat(
        self,
        message: str,
        history: List[ChatMessage],
        user_id: Optional[str] = None,
        law_filter=None,
        document_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat — yields text chunks as they arrive from LLM."""
        qu = await self._query_agent.analyze(message)
        if law_filter:
            qu.law_filter = law_filter

        if document_id:
            chunks = await self._fetch_document_chunks(document_id, message)
        else:
            chunks, _ = await self._retriever.retrieve(
                query=message,
                query_understanding=qu,
                top_k_final=self._settings.retrieval.final_context_k,
            )

        if not chunks:
            yield ("I could not find relevant content matching your query. "
                   "Try asking about specific people, dates, sections, or legal issues in the document. "
                   "Example: 'What was the judgment on title?' or 'Who were the plaintiffs?'")
            return

        compressed = compress_context(chunks, max_tokens=2500)
        messages = self._build_chat_messages(message, compressed, history)

        async for token in self._llm.stream(messages):
            yield token

    async def run_draft(self, request: DraftRequest, user_id: Optional[str] = None) -> Dict:
        """Drafting pipeline: retrieve relevant law → generate document."""
        # Build retrieval query from draft type + facts
        retrieval_query = f"{request.draft_type.value} {request.facts[:500]}"
        if request.sections_involved:
            retrieval_query += " " + " ".join(request.sections_involved)

        qu = QueryUnderstanding(
            original_query=retrieval_query,
            cleaned_query=retrieval_query,
            intent=LegalIntentType.DRAFTING_REQUEST,
            intent_confidence=1.0,
            draft_type=request.draft_type,
        )

        chunks, _ = await self._retriever.retrieve(
            query=retrieval_query,
            query_understanding=qu,
            top_k_final=8,
        )

        result = await self._drafting_agent.draft(
            draft_type=request.draft_type,
            facts=request.facts,
            parties=request.parties,
            retrieved_chunks=chunks,
            court=request.court,
            additional_context=request.additional_context,
        )
        result["session_id"] = str(__import__("uuid").uuid4())
        return result

    async def run_summarize(
        self, request: SummarizeRequest, user_id: Optional[str] = None
    ) -> JudgmentSummary:
        """Summarize a document by ID or raw text."""
        if request.text:
            return await self._summarization_agent.summarize_text(
                text=request.text,
                document_id="user_provided",
            )
        elif request.document_id:
            # Fetch chunks directly from DB by document_id — don't use retrieval
            # (retrieval returns top-k across all docs; small docs won't appear)
            from sqlalchemy import text as sql_text
            from backend.db.session import get_db_session
            from backend.models.domain import (
                ChunkType, DocumentMetadata, DocumentType,
                LawCategory, LegalChunk, RetrievedChunk,
            )

            async with get_db_session() as db:
                # Get document metadata
                doc_result = await db.execute(
                    sql_text("SELECT * FROM documents WHERE document_id = :id"),
                    {"id": request.document_id}
                )
                doc_row = doc_result.fetchone()
                if not doc_row:
                    raise ValueError(f"Document {request.document_id} not found")

                # Get all chunks ordered by index
                chunk_result = await db.execute(
                    sql_text("""
                        SELECT chunk_id, content, chunk_type, chunk_index, section_ref
                        FROM chunks
                        WHERE document_id = :id
                        ORDER BY chunk_index
                        LIMIT 40
                    """),
                    {"id": request.document_id}
                )
                rows = chunk_result.fetchall()

            if not rows:
                raise ValueError(f"No chunks found for document {request.document_id}")

            # Build metadata for summarization agent
            metadata = {
                "citation": getattr(doc_row, "citation", None),
                "court": getattr(doc_row, "court_name", None),
                "year": getattr(doc_row, "year", None),
                "case_name": None,
            }
            parties = getattr(doc_row, "parties", None)
            if isinstance(parties, dict):
                metadata["case_name"] = parties.get("name")
            elif isinstance(parties, str):
                metadata["case_name"] = parties

            # Concatenate chunk content and summarize
            full_text = "\n\n".join(row.content for row in rows)
            return await self._summarization_agent.summarize_text(
                text=full_text,
                document_id=request.document_id,
                metadata=metadata,
            )
        else:
            raise ValueError("Either text or document_id must be provided")

    async def _fetch_document_chunks(
        self, document_id: str, query: str, top_k: int = 15
    ) -> List:
        """
        Fetch chunks for a specific document ranked by relevance to query.
        Uses Postgres FTS first, falls back to vector search within document.
        """
        from sqlalchemy import text as sql_text
        from backend.db.session import get_db_session
        from backend.models.domain import (
            ChunkType, DocumentMetadata, DocumentType,
            LawCategory, LegalChunk, RetrievedChunk,
        )

        # Strip common query prefixes that contain doc title noise
        # e.g. "Summarize the key legal principles from the case: Ram Temple case - Copy.pdf"
        # → "key legal principles"
        clean_query = query
        for prefix in ["summarize the key", "summarize", "what is", "explain",
                        "from the case:", "from the document:", "in this case",
                        "who were", "what are"]:
            if clean_query.lower().startswith(prefix):
                clean_query = clean_query[len(prefix):].strip(" :-")
        # Remove filename artifacts
        if "." in clean_query.split()[-1] if clean_query.split() else False:
            clean_query = " ".join(clean_query.split()[:-1])
        # If query became too short, use original
        if len(clean_query.strip()) < 5:
            clean_query = query

        async with get_db_session() as db:
            # Try FTS ranked search within document
            result = await db.execute(sql_text("""
                SELECT chunk_id, content, chunk_type, chunk_index, section_ref,
                       ts_rank(content_tsv, plainto_tsquery('english', :q)) AS rank
                FROM chunks
                WHERE document_id = :doc_id
                  AND content_tsv @@ plainto_tsquery('english', :q)
                ORDER BY rank DESC
                LIMIT :k
            """), {"doc_id": document_id, "q": clean_query, "k": top_k})
            rows = result.fetchall()

            # Fallback 1: ILIKE search on content (catches names, nouns FTS misses)
            if not rows:
                # Extract key words (>4 chars, not stop words)
                stop = {'from', 'this', 'that', 'with', 'were', 'what', 'which',
                        'have', 'been', 'case', 'legal', 'court', 'suit', 'file'}
                keywords = [w.strip('.,?!') for w in clean_query.split()
                            if len(w) > 4 and w.lower() not in stop][:4]

                if keywords:
                    like_clause = " OR ".join(f"content ILIKE :kw{i}" for i in range(len(keywords)))
                    params = {"doc_id": document_id, "k": top_k}
                    for i, kw in enumerate(keywords):
                        params[f"kw{i}"] = f"%{kw}%"
                    result = await db.execute(sql_text(f"""
                        SELECT chunk_id, content, chunk_type, chunk_index, section_ref,
                               0.5 AS rank
                        FROM chunks
                        WHERE document_id = :doc_id AND ({like_clause})
                        ORDER BY chunk_index
                        LIMIT :k
                    """), params)
                    rows = result.fetchall()

            # Fallback 2: return highest-content chunks (not just first N — skip short header chunks)
            if not rows:
                result = await db.execute(sql_text("""
                    SELECT chunk_id, content, chunk_type, chunk_index, section_ref,
                           1.0 AS rank
                    FROM chunks
                    WHERE document_id = :doc_id
                      AND content_length > 200
                    ORDER BY chunk_index
                    LIMIT :k
                """), {"doc_id": document_id, "k": top_k})
                rows = result.fetchall()

        if not rows:
            return []

        meta = DocumentMetadata(
            document_type=DocumentType.JUDGMENT,
            law=LawCategory.OTHER,
            language="en",
        )

        chunks = []
        for row in rows:
            lc = LegalChunk(
                chunk_id=str(row.chunk_id),
                document_id=str(document_id),
                chunk_type=ChunkType.PASSAGE,
                content=row.content,
                content_length=len(row.content),
                chunk_index=row.chunk_index,
                section_ref=row.section_ref or "",
                metadata=meta,
            )
            rc = RetrievedChunk(
                chunk=lc,
                bm25_score=float(row.rank),
                vector_score=float(row.rank),
                final_score=float(row.rank),
                retrieval_method="document_fts",
            )
            chunks.append(rc)

        return chunks

        return chunks

    def _build_chat_messages(
        self,
        query: str,
        context: str,
        history: List[ChatMessage],
    ) -> List[Dict[str, str]]:
        """Build message list for LLM with system prompt, history, and context."""
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT.format(context=context)}
        ]
        # Include last 6 history messages to stay within context limit
        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": query})
        return messages
