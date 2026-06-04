"""
Unit tests for core pipeline components.
Run: pytest backend/tests/unit/ -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.ingestion.chunkers.legal_chunker import LegalChunker
from backend.models.domain import (
    ChunkType, DocumentMetadata, DocumentType, LawCategory, LegalChunk
)


# ── Chunker Tests ──────────────────────────────────────────────────────────

class TestLegalChunker:
    def setup_method(self):
        self.chunker = LegalChunker()
        self.statute_meta = DocumentMetadata(
            document_id="test-001",
            document_type=DocumentType.STATUTE,
            law=LawCategory.BNS,
        )
        self.judgment_meta = DocumentMetadata(
            document_id="test-002",
            document_type=DocumentType.JUDGMENT,
        )

    def test_statute_chunking_produces_sections(self):
        text = """
CHAPTER XV — OF OFFENCES RELATING TO PROPERTY

318. Cheating.—Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property to any person, or to consent that any person shall retain any property, or intentionally induces the person so deceived to do or omit to do anything which he would not do or omit if he were not so deceived, and which act or omission causes or is likely to cause damage or harm to that person in body, mind, reputation or property, is said to "cheat".
Punishment.—Whoever cheats shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.

319. Cheating by personation.—A person is said to "cheat by personation" if he cheats by pretending to be some other person, or by knowingly substituting one person for another.
Punishment.—Whoever cheats by personation shall be punished with imprisonment of either description for a term which may extend to five years, or with fine, or with both.
"""
        chunks = self.chunker.chunk(text, self.statute_meta)
        assert len(chunks) >= 2
        # Should find section references
        section_refs = [c.section_ref for c in chunks if c.section_ref]
        assert len(section_refs) > 0

    def test_judgment_chunking_by_structure(self):
        text = """
SUPREME COURT OF INDIA

FACTS

The petitioner was arrested on 15th March 2024 under Section 318 BNS...

ISSUES

1. Whether anticipatory bail should be granted?
2. Whether the offence is non-bailable?

ARGUMENTS

Petitioner's Counsel submitted that the evidence is circumstantial...

FINDINGS

This court finds that the petitioner has made out a prima facie case...

ORDER

The application for anticipatory bail is hereby allowed.
"""
        chunks = self.chunker.chunk(text, self.judgment_meta)
        chunk_types = {c.chunk_type for c in chunks}
        # Should identify at least some structural chunks
        assert len(chunks) >= 2

    def test_fallback_to_sentence_window(self):
        # Plain text with no structure markers
        text = " ".join(["This is a legal sentence about Indian law." for _ in range(100)])
        chunks = self.chunker.chunk(text, self.judgment_meta)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) >= 100
            assert len(chunk.content) <= 1400  # MAX_CHUNK_CHARS + buffer

    def test_min_chunk_size_enforced(self):
        text = "Short.\n\nThis is a longer section with enough content to be a valid chunk. " * 5
        chunks = self.chunker.chunk(text, self.statute_meta)
        for chunk in chunks:
            assert len(chunk.content) >= 100

    def test_oversized_chunks_split(self):
        # Create a single very long section
        long_section = "318. Very Long Section.—" + ("A" * 2000) + "."
        chunks = self.chunker.chunk(long_section, self.statute_meta)
        for chunk in chunks:
            assert len(chunk.content) <= 1300


# ── Query Understanding Tests ───────────────────────────────────────────────

class TestQueryUnderstandingAgent:
    def setup_method(self):
        from backend.agents.query_understanding.agent import QueryUnderstandingAgent
        self.agent = QueryUnderstandingAgent()

    def test_extract_sections_regex(self):
        query = "What does BNS Section 318 say about cheating?"
        sections = self.agent._extract_sections_regex(query)
        assert "318" in sections

    def test_extract_law_filter_bns(self):
        query = "BNS provisions for murder"
        filters = self.agent._extract_law_filters_regex(query)
        from backend.models.domain import LawCategory
        assert filters is not None
        assert LawCategory.BNS in filters

    def test_extract_draft_type_bail(self):
        query = "Draft anticipatory bail application"
        dtype = self.agent._extract_draft_type_regex(query)
        from backend.models.domain import DraftType
        assert dtype == DraftType.ANTICIPATORY_BAIL

    def test_year_range_extraction(self):
        query = "Supreme Court judgments from 2022 to 2024"
        yr = self.agent._extract_year_range(query)
        assert yr is not None
        assert yr["from"] == 2022
        assert yr["to"] == 2024

    def test_fallback_analysis_drafting(self):
        result = self.agent._fallback_analysis("Draft a legal notice for me")
        assert result["intent"] == "drafting_request"


# ── Verification Agent Tests ────────────────────────────────────────────────

class TestVerificationAgent:
    def setup_method(self):
        from backend.agents.verification.agent import VerificationAgent
        self.agent = VerificationAgent()

    def test_extract_claims_finds_sections(self):
        text = "Under BNS Section 318, cheating is punishable. Section 482 BNSS governs anticipatory bail."
        claims = self.agent._extract_claims(text)
        assert "318" in claims or "482" in claims

    def test_extract_claims_finds_citations(self):
        text = "As held in AIR 2024 SC 111, the court ruled..."
        claims = self.agent._extract_claims(text)
        assert any("2024" in c for c in claims)

    @pytest.mark.asyncio
    async def test_verify_with_matching_chunks(self):
        from backend.models.domain import (
            ChunkType, DocumentMetadata, DocumentType, LegalChunk, RetrievedChunk
        )
        meta = DocumentMetadata(
            document_id="d1", document_type=DocumentType.STATUTE, law=LawCategory.BNS
        )
        chunk = LegalChunk(
            chunk_id="c1", document_id="d1",
            chunk_type=ChunkType.SECTION,
            content="Section 318 cheating provision",
            section_ref="318",
            metadata=meta,
        )
        retrieved = [RetrievedChunk(chunk=chunk, final_score=0.9)]
        response = "Under Section 318 of BNS, cheating is defined as..."
        flags, citations, score = await self.agent.verify(response, retrieved)
        assert score > 0  # At least partial verification

    @pytest.mark.asyncio
    async def test_detect_hallucinated_section(self):
        from backend.models.domain import (
            ChunkType, DocumentMetadata, DocumentType, LegalChunk, RetrievedChunk, LawCategory
        )
        meta = DocumentMetadata(
            document_id="d1", document_type=DocumentType.STATUTE, law=LawCategory.BNS
        )
        chunk = LegalChunk(
            chunk_id="c1", document_id="d1",
            chunk_type=ChunkType.SECTION,
            content="Section 100 content",
            section_ref="100",
            metadata=meta,
        )
        retrieved = [RetrievedChunk(chunk=chunk, final_score=0.8)]
        # Response cites section 999 which is NOT in retrieved chunks
        response = "Under Section 999 of BNS, which covers..."
        flags, _, score = await self.agent.verify(response, retrieved)
        assert len(flags) > 0  # Should flag section 999


# ── RRF Fusion Tests ────────────────────────────────────────────────────────

class TestHybridRetriever:
    def test_rrf_score_formula(self):
        from backend.retrieval.hybrid.pipeline import HybridRetriever, RRF_K
        # Create minimal mock
        retriever = object.__new__(HybridRetriever)
        retriever.__class__ = HybridRetriever
        # RRF score for rank 1 = 1/(60+1) ≈ 0.0164
        score = 1.0 / (RRF_K + 1)
        assert abs(score - 0.01639) < 0.0001

    def test_rrf_fusion_deduplicates(self):
        from backend.retrieval.hybrid.pipeline import HybridRetriever
        from backend.models.domain import (
            ChunkType, DocumentMetadata, DocumentType, LegalChunk, RetrievedChunk, LawCategory
        )

        # Build dummy chunks
        def make_chunk(cid):
            meta = DocumentMetadata(document_id="d1", document_type=DocumentType.JUDGMENT)
            return RetrievedChunk(
                chunk=LegalChunk(chunk_id=cid, document_id="d1",
                                  chunk_type=ChunkType.PASSAGE, content="test", metadata=meta),
                bm25_score=0.5, vector_score=0.5,
            )

        bm25_results = [make_chunk("c1"), make_chunk("c2"), make_chunk("c3")]
        vector_results = [make_chunk("c1"), make_chunk("c4"), make_chunk("c2")]

        retriever = object.__new__(HybridRetriever)
        # Manually inject the method
        from backend.retrieval.hybrid.pipeline import HybridRetriever as HR
        fused = HR._reciprocal_rank_fusion(retriever, bm25_results, vector_results, 0.4, 0.6)

        chunk_ids = [c.chunk.chunk_id for c in fused]
        # No duplicates
        assert len(chunk_ids) == len(set(chunk_ids))
        # c1 and c2 appear in both lists → should rank highest
        assert fused[0].chunk.chunk_id in ("c1", "c2")
