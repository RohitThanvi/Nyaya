"""
Core domain models for NyayaAI.
Extended with source_url on Citation, page-level anchors, retrieval_method
field on RetrievedChunk, and IngestionJob staging support.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class DocumentType(str, Enum):
    JUDGMENT   = "judgment"
    STATUTE    = "statute"
    NOTIFICATION = "notification"
    CIRCULAR   = "circular"
    UPLOAD     = "upload"


class LawCategory(str, Enum):
    BNS          = "BNS"
    BNSS         = "BNSS"
    BSA          = "BSA"
    IPC          = "IPC"
    CRPC         = "CrPC"
    CONSTITUTION = "Constitution"
    OTHER        = "Other"


class CourtType(str, Enum):
    SUPREME_COURT  = "Supreme Court"
    HIGH_COURT     = "High Court"
    DISTRICT_COURT = "District Court"
    TRIBUNAL       = "Tribunal"
    OTHER          = "Other"


class ChunkType(str, Enum):
    # Judgment structural chunks
    FACTS       = "facts"
    ISSUES      = "issues"
    ARGUMENTS   = "arguments"
    FINDINGS    = "findings"
    RATIO       = "ratio"
    FINAL_ORDER = "final_order"
    # Statute chunks
    CHAPTER     = "chapter"
    SECTION     = "section"
    SUBSECTION  = "subsection"
    EXPLANATION = "explanation"
    PUNISHMENT  = "punishment"
    # Fallback
    PASSAGE     = "passage"


class LegalIntentType(str, Enum):
    PROVISION_LOOKUP  = "provision_lookup"
    CASE_SEARCH       = "case_search"
    PROCEDURE_QUERY   = "procedure_query"
    DRAFTING_REQUEST  = "drafting_request"
    SUMMARIZATION     = "summarization"
    GENERAL_QUERY     = "general_query"


class DraftType(str, Enum):
    BAIL_APPLICATION      = "bail_application"
    ANTICIPATORY_BAIL     = "anticipatory_bail"
    LEGAL_NOTICE          = "legal_notice"
    AFFIDAVIT             = "affidavit"
    COMPLAINT             = "complaint"
    FIR_QUASHING_PETITION = "fir_quashing_petition"
    WRITTEN_STATEMENT     = "written_statement"
    VAKALATNAMA           = "vakalatnama"


class UserRole(str, Enum):
    ADMIN      = "admin"
    ADVOCATE   = "advocate"
    RESEARCHER = "researcher"
    STUDENT    = "student"
    GUEST      = "guest"


class RetrievalPath(str, Enum):
    """Which retrieval path produced a chunk — used for diagnostics."""
    EXACT_LOOKUP  = "exact_lookup"    # Primary key / citation string match
    BM25          = "bm25"            # PostgreSQL tsvector / Elasticsearch
    VECTOR        = "vector"          # Qdrant ANN
    HYBRID        = "hybrid"          # BM25 + vector fused
    DOCUMENT_FTS  = "document_fts"    # Scoped FTS within a single document


# ─────────────────────────────────────────────
# Document Metadata
# ─────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Rich metadata for any legal document."""
    document_id:   str            = Field(default_factory=lambda: str(uuid4()))
    document_type: DocumentType
    law:           Optional[LawCategory]  = None
    court:         Optional[CourtType]   = None
    court_name:    Optional[str]  = None
    case_number:   Optional[str]  = None
    citation:      Optional[str]  = None
    year:          Optional[int]  = None
    date_decided:  Optional[datetime] = None
    bench:         Optional[List[str]] = None
    parties:       Optional[Dict[str, str]] = None
    section:       Optional[str]  = None
    chapter:       Optional[str]  = None
    topic:         Optional[str]  = None
    keywords:      List[str]      = Field(default_factory=list)
    source_url:    Optional[str]  = None   # canonical public URL for the document
    file_path:     Optional[str]  = None
    is_landmark:   bool           = False
    language:      str            = "en"
    created_at:    datetime       = Field(default_factory=datetime.utcnow)
    updated_at:    datetime       = Field(default_factory=datetime.utcnow)


class LegalChunk(BaseModel):
    """A single retrievable unit from a legal document."""
    chunk_id:       str           = Field(default_factory=lambda: str(uuid4()))
    document_id:    str
    chunk_type:     ChunkType
    content:        str
    content_length: int           = 0
    chunk_index:    int           = 0
    page_number:    Optional[int] = None   # physical page in source PDF
    section_ref:    Optional[str] = None   # e.g. "318"
    subsection_ref: Optional[str] = None   # e.g. "318(2)(b)"
    metadata:       DocumentMetadata
    embedding:      Optional[List[float]] = None
    bm25_tokens:    Optional[List[str]]   = None

    def model_post_init(self, __context: Any) -> None:
        self.content_length = len(self.content)


# ─────────────────────────────────────────────
# Retrieval Models
# ─────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval with full scoring breakdown."""
    chunk:            LegalChunk
    bm25_score:       float = 0.0
    vector_score:     float = 0.0
    hybrid_score:     float = 0.0
    rerank_score:     float = 0.0
    final_score:      float = 0.0
    retrieval_source: str   = "hybrid"   # RetrievalPath value
    retrieval_method: str   = "hybrid"   # alias kept for legacy compatibility


class Citation(BaseModel):
    """
    A verified legal citation with full provenance.
    source_url + page_number allow the frontend to link directly to the
    source document at the exact page that was used.
    """
    citation_id:    str           = Field(default_factory=lambda: str(uuid4()))
    document_id:    str
    chunk_id:       str
    section:        Optional[str] = None
    subsection:     Optional[str] = None
    paragraph:      Optional[int] = None
    page_number:    Optional[int] = None   # page in source PDF
    citation_text:  str                    # e.g. "AIR 2025 SC 111"
    citation_type:  str                    # judgment | statute | notification
    court:          Optional[str] = None
    year:           Optional[int] = None
    source_url:     Optional[str] = None   # direct link to source document
    snippet:        Optional[str] = None   # first 150 chars of matching chunk
    relevance_note: Optional[str] = None
    verified:       bool          = False


# ─────────────────────────────────────────────
# Query Models
# ─────────────────────────────────────────────

class LegalEntity(BaseModel):
    text:         str
    entity_type:  str   # LAW_SECTION | CASE_NAME | LEGAL_CONCEPT | CRIME_TYPE | COURT
    confidence:   float = 1.0
    normalized:   Optional[str] = None


class ExtractedClaim(BaseModel):
    """
    A structured legal claim extracted from LLM output.
    Used by VerificationAgent for precise cross-checking.
    """
    claim_type:    str            # "section" | "judgment" | "article"
    raw_text:      str            # exact text as it appeared in LLM output
    law:           Optional[str]  = None   # "BNS" | "BNSS" | "BSA" | "IPC" etc.
    section_num:   Optional[str]  = None   # normalized: "318"
    citation_str:  Optional[str]  = None   # "AIR 2025 SC 111"
    chunk_tag:     Optional[str]  = None   # 8-char chunk_id from a nearby
                                            # <CHUNK:xxxxxxxx> tag — mechanical
                                            # proof this claim traces to a chunk
                                            # actually retrieved for this query


class QueryUnderstanding(BaseModel):
    original_query:    str
    cleaned_query:     str
    intent:            LegalIntentType
    intent_confidence: float
    legal_entities:    List[LegalEntity]  = Field(default_factory=list)
    law_filter:        Optional[List[LawCategory]]  = None
    court_filter:      Optional[List[CourtType]]    = None
    year_range:        Optional[Dict[str, int]]     = None
    section_refs:      List[str]          = Field(default_factory=list)
    expanded_queries:  List[str]          = Field(default_factory=list)
    draft_type:        Optional[DraftType] = None


# ─────────────────────────────────────────────
# Agent Pipeline Models
# ─────────────────────────────────────────────

class AgentState(BaseModel):
    session_id:          str           = Field(default_factory=lambda: str(uuid4()))
    user_id:             Optional[str] = None
    original_query:      str
    query_understanding: Optional[QueryUnderstanding] = None
    retrieved_chunks:    List[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks:     List[RetrievedChunk] = Field(default_factory=list)
    # Output of LegalMappingAgent — chunks that survived its pre-generation
    # validation (citation_chunk_id or section_ref actually present in
    # reranked_chunks). When populated, _step_compress uses THIS set instead
    # of the raw reranked_chunks, so the LLM never even sees a chunk the
    # mapper already determined was irrelevant or improperly cited —
    # shrinking the hallucination surface before generation, not just
    # catching it after.
    mapped_sections:     Optional[Dict[str, Any]] = None
    compressed_context:  Optional[str] = None
    citations:           List[Citation] = Field(default_factory=list)
    verified_citations:  List[Citation] = Field(default_factory=list)
    raw_llm_response:    Optional[str]  = None
    structured_response: Optional[Dict[str, Any]] = None
    hallucination_flags: List[str]      = Field(default_factory=list)
    pipeline_trace:      List[Dict[str, Any]] = Field(default_factory=list)
    error:               Optional[str]  = None
    created_at:          datetime       = Field(default_factory=datetime.utcnow)
    latency_ms:          Dict[str, float] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# API Request / Response Models
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:             str           = Field(..., min_length=3, max_length=1000)
    law_filter:        Optional[List[LawCategory]] = None
    court_filter:      Optional[List[CourtType]]   = None
    year_from:         Optional[int] = None
    year_to:           Optional[int] = None
    document_type:     Optional[DocumentType] = None
    top_k:             int           = Field(default=10, ge=1, le=50)
    include_statutes:  bool          = True
    include_judgments: bool          = True


class ChatMessage(BaseModel):
    role:      str      = Field(..., pattern="^(user|assistant|system)$")
    content:   str      = Field(..., min_length=1, max_length=10000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    session_id:  Optional[str]  = None
    message:     str            = Field(..., min_length=3, max_length=5000)
    history:     List[ChatMessage] = Field(default_factory=list, max_length=20)
    law_filter:  Optional[List[LawCategory]] = None
    document_id: Optional[str]  = None
    stream:      bool           = Field(default=True)


class DraftRequest(BaseModel):
    draft_type:         DraftType
    facts:              str  = Field(..., min_length=50, max_length=10000)
    parties:            Dict[str, str] = Field(default_factory=dict)
    court:              Optional[str]  = None
    sections_involved:  List[str]      = Field(default_factory=list)
    additional_context: Optional[str]  = None


class SummarizeRequest(BaseModel):
    document_id:  Optional[str] = None
    text:         Optional[str] = None
    summary_type: str           = Field(default="full")


class LegalResponse(BaseModel):
    query:                   str
    session_id:              str
    intent:                  Optional[str] = None
    answer:                  str
    relevant_sections:       List[Dict[str, Any]] = Field(default_factory=list)
    precedents:              List[Dict[str, Any]] = Field(default_factory=list)
    procedural_requirements: List[str]            = Field(default_factory=list)
    citations:               List[Citation]        = Field(default_factory=list)
    confidence:              float                 = Field(ge=0.0, le=1.0)
    warnings:                List[str]             = Field(default_factory=list)
    hallucination_flags:     List[str]             = Field(default_factory=list)
    latency_ms:              Optional[float]       = None
    # Debug fields — populated when pipeline_trace is available
    pipeline_trace:          List[Dict[str, Any]]  = Field(default_factory=list)
    retrieval_debug:         Optional[Dict[str, Any]] = None
    timestamp:               datetime              = Field(default_factory=datetime.utcnow)


class JudgmentSummary(BaseModel):
    document_id:        str
    case_name:          Optional[str] = None
    citation:           Optional[str] = None
    court:              Optional[str] = None
    date_decided:       Optional[datetime] = None
    facts:              str
    issues:             List[str]          = Field(default_factory=list)
    arguments:          Dict[str, str]     = Field(default_factory=dict)
    findings:           str
    ratio_decidendi:    Optional[str]      = None
    final_order:        str
    sections_discussed: List[str]          = Field(default_factory=list)
    is_landmark:        bool               = False
    summary_brief:      str


class UploadResponse(BaseModel):
    document_id:    str
    filename:       str
    pages:          int
    chunks_created: int
    failed_chunk_ids: List[str] = Field(default_factory=list)
    status:         str   # "success" | "partial" | "empty"
    message:        str


# ─────────────────────────────────────────────
# User & Auth Models
# ─────────────────────────────────────────────

class UserLogin(BaseModel):
    email:    str
    password: str


class UserCreate(BaseModel):
    email:          str
    password:       str = Field(..., min_length=8)
    full_name:      str
    role:           UserRole = UserRole.RESEARCHER
    bar_enrollment: Optional[str] = None


class UserInDB(BaseModel):
    user_id:        str = Field(default_factory=lambda: str(uuid4()))
    email:          str
    full_name:      str
    role:           UserRole
    hashed_password: str
    is_active:      bool = True
    bar_enrollment: Optional[str]  = None
    created_at:     datetime       = Field(default_factory=datetime.utcnow)
    last_login:     Optional[datetime] = None


class TokenPayload(BaseModel):
    sub:  str
    role: str
    exp:  int


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int


# ─────────────────────────────────────────────
# Ingestion Models
# ─────────────────────────────────────────────

class IngestionJob(BaseModel):
    job_id:               str      = Field(default_factory=lambda: str(uuid4()))
    source_type:          str
    file_path:            Optional[str] = None
    url:                  Optional[str] = None
    status:               str      = "pending"
    documents_processed:  int      = 0
    chunks_created:       int      = 0
    error:                Optional[str] = None
    created_at:           datetime = Field(default_factory=datetime.utcnow)
    completed_at:         Optional[datetime] = None


class StagedChunk(BaseModel):
    """
    Intermediate model used by the ingestion worker pipeline.
    Written to staging table after embedding; flushed to Qdrant in batches.
    """
    chunk_id:    str
    document_id: str
    embedding:   List[float]
    metadata:    Dict[str, Any]
    indexed:     bool = False


class ParsedDocument(BaseModel):
    document_id:   str  = Field(default_factory=lambda: str(uuid4()))
    raw_text:      str
    metadata:      DocumentMetadata
    structure:     Dict[str, Any] = Field(default_factory=dict)
    pages:         int  = 0
    parse_method:  str  = "pdfplumber"
    parse_quality: float = 1.0
