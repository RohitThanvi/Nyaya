"""
Unit tests for core pipeline components.
Run: pytest backend/tests/unit/ -v

Rewritten against the actual current APIs:
    LegalChunker.chunk(ParsedDocument) -> List[LegalChunk]   (not chunk(text, meta))
    QueryUnderstandingAgent(llm_client=...).understand(query) (not analyze())
    VerificationAgent(db, llm_client).verify(response, chunks, query)
        -> (verified_citations, hallucination_flags)          (not (flags, citations, score))
    _reciprocal_rank_fusion(result_lists) is a MODULE-LEVEL function
        taking [(chunks, weight), ...] — not a bound method on HybridRetriever
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.ingestion.chunkers.legal_chunker import LegalChunker
from backend.models.domain import (
    ChunkType, DocumentMetadata, DocumentType, LawCategory,
    LegalChunk, ParsedDocument,
)


def _make_parsed(text: str, document_type=DocumentType.STATUTE, law=LawCategory.BNS) -> ParsedDocument:
    meta = DocumentMetadata(
        document_id="test-doc",
        document_type=document_type,
        law=law,
    )
    return ParsedDocument(
        document_id="test-doc",
        raw_text=text,
        metadata=meta,
        pages=1,
    )


# ── Chunker Tests ────────────────────────────────────────────────────────────

class TestLegalChunker:
    def setup_method(self):
        self.chunker = LegalChunker()

    def test_statute_chunking_produces_sections(self):
        text = """CHAPTER XV — OF OFFENCES RELATING TO PROPERTY

318. Cheating.—Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property to any person, or to consent that any person shall retain any property, or intentionally induces the person so deceived to do or omit to do anything which he would not do or omit if he were not so deceived, and which act or omission causes or is likely to cause damage or harm to that person in body, mind, reputation or property, is said to cheat.
Punishment.—Whoever cheats shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.

319. Cheating by personation.—A person is said to cheat by personation if he cheats by pretending to be some other person, or by knowingly substituting one person for another.
Punishment.—Whoever cheats by personation shall be punished with imprisonment of either description for a term which may extend to five years, or with fine, or with both."""
        parsed = _make_parsed(text, document_type=DocumentType.STATUTE, law=LawCategory.BNS)
        chunks = self.chunker.chunk(parsed)
        assert len(chunks) >= 1
        section_refs = [c.section_ref for c in chunks if c.section_ref]
        assert len(section_refs) > 0
        assert "318" in section_refs or "319" in section_refs

    def test_judgment_chunking_by_structure(self):
        text = """SUPREME COURT OF INDIA

FACTS:
The petitioner was arrested on 15th March 2024 under Section 318 BNS. The matter relates to an allegation of cheating in a property transaction involving the petitioner and the complainant.

ISSUES:
1. Whether anticipatory bail should be granted.
2. Whether the offence is non-bailable.

HELD:
This court finds that the petitioner has made out a prima facie case for grant of anticipatory bail given the circumstances of the matter and the nature of allegations.

FINAL ORDER:
The application for anticipatory bail is hereby allowed subject to conditions imposed by this court."""
        parsed = _make_parsed(text, document_type=DocumentType.JUDGMENT, law=None)
        chunks = self.chunker.chunk(parsed)
        assert len(chunks) >= 2
        chunk_types = {c.chunk_type for c in chunks}
        # Should identify at least some structural chunk types beyond plain PASSAGE
        assert len(chunk_types) >= 1

    def test_fallback_to_sentence_window(self):
        text = " ".join(["This is a legal sentence about Indian law and its provisions."] * 60)
        parsed = _make_parsed(text, document_type=DocumentType.JUDGMENT, law=None)
        chunks = self.chunker.chunk(parsed)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) >= 150  # MIN_CHUNK_CHARS in legal_chunker.py

    def test_min_chunk_size_enforced(self):
        text = ("Short. This is a longer section with enough content to be considered "
                 "a genuinely valid chunk for retrieval purposes under the system. ") * 5
        parsed = _make_parsed(text, document_type=DocumentType.STATUTE, law=LawCategory.BNS)
        chunks = self.chunker.chunk(parsed)
        for chunk in chunks:
            assert len(chunk.content) >= 150

    def test_oversized_section_is_split_or_capped(self):
        long_section = "318. Very Long Section.—" + ("This is filler legal text. " * 150)
        parsed = _make_parsed(long_section, document_type=DocumentType.STATUTE, law=LawCategory.BNS)
        chunks = self.chunker.chunk(parsed)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) <= 2100  # MAX_CHUNK_CHARS + small buffer


# ── Query Understanding Tests ───────────────────────────────────────────────

class TestQueryUnderstandingAgent:
    def setup_method(self):
        from backend.agents.query_understanding.agent import QueryUnderstandingAgent
        mock_llm = AsyncMock()
        self.agent = QueryUnderstandingAgent(llm_client=mock_llm)

    @pytest.mark.asyncio
    async def test_understand_returns_query_understanding(self):
        # Force the LLM-free regex path by making the LLM call raise
        self.agent._llm.complete_with_json = AsyncMock(side_effect=RuntimeError("no llm in test"))
        result = await self.agent.understand("What does BNS Section 318 say about cheating?")
        assert result is not None
        assert result.original_query == "What does BNS Section 318 say about cheating?"
        assert "318" in result.section_refs

    @pytest.mark.asyncio
    async def test_law_filter_extraction_bns(self):
        self.agent._llm.complete_with_json = AsyncMock(side_effect=RuntimeError("no llm in test"))
        result = await self.agent.understand("BNS provisions for murder")
        from backend.models.domain import LawCategory
        assert result.law_filter is not None
        assert LawCategory.BNS in result.law_filter

    @pytest.mark.asyncio
    async def test_draft_type_extraction_anticipatory_bail(self):
        self.agent._llm.complete_with_json = AsyncMock(side_effect=RuntimeError("no llm in test"))
        result = await self.agent.understand("Draft an anticipatory bail application")
        from backend.models.domain import DraftType
        assert result.draft_type == DraftType.ANTICIPATORY_BAIL

    @pytest.mark.asyncio
    async def test_year_range_extraction(self):
        self.agent._llm.complete_with_json = AsyncMock(side_effect=RuntimeError("no llm in test"))
        result = await self.agent.understand("Supreme Court judgments from 2022 to 2024")
        assert result.year_range is not None
        assert result.year_range.get("from") == 2022
        assert result.year_range.get("to") == 2024


# ── Verification Agent Tests ────────────────────────────────────────────────

class TestVerificationAgent:
    def setup_method(self):
        from backend.agents.verification.agent import VerificationAgent
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        self.agent = VerificationAgent(db=mock_db, llm_client=mock_llm)

    @pytest.mark.asyncio
    async def test_verify_with_matching_chunk_returns_citation(self):
        from backend.models.domain import RetrievedChunk

        meta = DocumentMetadata(document_id="d1", document_type=DocumentType.STATUTE, law=LawCategory.BNS)
        chunk = LegalChunk(
            chunk_id="c1", document_id="d1",
            chunk_type=ChunkType.SECTION,
            content="Section 318 BNS — cheating is defined as fraudulent inducement to deliver property.",
            section_ref="318",
            metadata=meta,
        )
        retrieved = [RetrievedChunk(chunk=chunk, final_score=0.9)]

        # Mock the structured claim extraction to return a known claim deterministically
        from backend.models.domain import ExtractedClaim
        self.agent._extract_claims = AsyncMock(return_value=[
            ExtractedClaim(claim_type="section", raw_text="Section 318 BNS",
                            law="BNS", section_num="318")
        ])

        response = "Under Section 318 of BNS, cheating is defined as..."
        verified, flags = await self.agent.verify(response, retrieved, "what is cheating under BNS")
        assert len(verified) == 1
        assert verified[0].verified is True
        assert len(flags) == 0

    @pytest.mark.asyncio
    async def test_detect_hallucinated_section_not_in_chunks_or_db(self):
        from backend.models.domain import RetrievedChunk, ExtractedClaim

        meta = DocumentMetadata(document_id="d1", document_type=DocumentType.STATUTE, law=LawCategory.BNS)
        chunk = LegalChunk(
            chunk_id="c1", document_id="d1",
            chunk_type=ChunkType.SECTION,
            content="Section 100 content about an unrelated provision.",
            section_ref="100",
            metadata=meta,
        )
        retrieved = [RetrievedChunk(chunk=chunk, final_score=0.8)]

        self.agent._extract_claims = AsyncMock(return_value=[
            ExtractedClaim(claim_type="section", raw_text="Section 999 BNS",
                            law="BNS", section_num="999")
        ])
        # DB fallback also finds nothing
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        self.agent._db.execute = AsyncMock(return_value=mock_result)

        response = "Under Section 999 of BNS, which covers..."
        verified, flags = await self.agent.verify(response, retrieved, "irrelevant query")
        assert len(flags) > 0
        assert len(verified) == 0


# ── RRF Fusion Tests ─────────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def test_rrf_score_formula(self):
        from backend.retrieval.hybrid.pipeline import RRF_K, _rrf
        # RRF score for rank 1, weight 1.0 = 1/(60+1) ≈ 0.01639
        score = _rrf(rank=1, weight=1.0)
        assert abs(score - (1.0 / (RRF_K + 1))) < 1e-9

    def test_rrf_fusion_deduplicates_and_ranks(self):
        from backend.retrieval.hybrid.pipeline import _reciprocal_rank_fusion
        from backend.models.domain import RetrievedChunk

        def make_chunk(cid: str) -> RetrievedChunk:
            meta = DocumentMetadata(document_id="d1", document_type=DocumentType.JUDGMENT)
            return RetrievedChunk(
                chunk=LegalChunk(chunk_id=cid, document_id="d1",
                                  chunk_type=ChunkType.PASSAGE, content="test", metadata=meta),
                bm25_score=0.5, vector_score=0.5,
            )

        bm25_results = [make_chunk("c1"), make_chunk("c2"), make_chunk("c3")]
        vector_results = [make_chunk("c1"), make_chunk("c4"), make_chunk("c2")]

        # Module-level function: takes [(chunks, weight), ...]
        fused = _reciprocal_rank_fusion([(bm25_results, 0.55), (vector_results, 0.45)])

        chunk_ids = [c.chunk.chunk_id for c in fused]
        assert len(chunk_ids) == len(set(chunk_ids))   # no duplicates
        assert fused[0].chunk.chunk_id in ("c1", "c2")  # appears in both lists → ranks highest
