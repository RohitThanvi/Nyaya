"""
Core domain models for NyayaAI.
All data structures used across the system are defined here.
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
    JUDGMENT = "judgment"
    STATUTE = "statute"
    NOTIFICATION = "notification"
    CIRCULAR = "circular"
    UPLOAD = "upload"


class LawCategory(str, Enum):
    BNS = "BNS"          # Bharatiya Nyaya Sanhita
    BNSS = "BNSS"        # Bharatiya Nagarik Suraksha Sanhita
    BSA = "BSA"          # Bharatiya Sakshya Adhiniyam
    IPC = "IPC"          # Indian Penal Code (legacy references)
    CRPC = "CrPC"        # Criminal Procedure Code (legacy)
    CONSTITUTION = "Constitution"
    OTHER = "Other"


class CourtType(str, Enum):
    SUPREME_COURT = "Supreme Court"
    HIGH_COURT = "High Court"
    DISTRICT_COURT = "District Court"
    TRIBUNAL = "Tribunal"
    OTHER = "Other"


class ChunkType(str, Enum):
    # Judgment chunks
    FACTS = "facts"
    ISSUES = "issues"
    ARGUMENTS = "arguments"
    FINDINGS = "findings"
    RATIO = "ratio"
    FINAL_ORDER = "final_order"
    # Statute chunks
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    EXPLANATION = "explanation"
    PUNISHMENT = "punishment"
    # Fallback
    PASSAGE = "passage"


class LegalIntentType(str, Enum):
    PROVISION_LOOKUP = "provision_lookup"
    CASE_SEARCH = "case_search"
    PROCEDURE_QUERY = "procedure_query"
    DRAFTING_REQUEST = "drafting_request"
    SUMMARIZATION = "summarization"
    GENERAL_QUERY = "general_query"


class DraftType(str, Enum):
    BAIL_APPLICATION = "bail_application"
    ANTICIPATORY_BAIL = "anticipatory_bail"
    LEGAL_NOTICE = "legal_notice"
    AFFIDAVIT = "affidavit"
    COMPLAINT = "complaint"
    FIR_QUASHING_PETITION = "fir_quashing_petition"
    WRITTEN_STATEMENT = "written_statement"
    VAKALATNAMA = "vakalatnama"


class UserRole(str, Enum):
    ADMIN = "admin"
    ADVOCATE = "advocate"
    RESEARCHER = "researcher"
    STUDENT = "student"
    GUEST = "guest"


# ─────────────────────────────────────────────
# Document Metadata
# ─────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Rich metadata for any legal document."""
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    document_type: DocumentType
    law: Optional[LawCategory] = None
    court: Optional[CourtType] = None
    court_name: Optional[str] = None          # e.g., "Bombay High Court"
    case_number: Optional[str] = None
    citation: Optional[str] = None            # e.g., "AIR 2025 SC 111"
    year: Optional[int] = None
    date_decided: Optional[datetime] = None
    bench: Optional[List[str]] = None         # List of judges
    parties: Optional[Dict[str, str]] = None  # {"petitioner": "...", "respondent": "..."}
    section: Optional[str] = None
    chapter: Optional[str] = None
    topic: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    is_landmark: bool = False
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class LegalChunk(BaseModel):
    """A single retrievable unit from a legal document."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_type: ChunkType
    content: str
    content_length: int = 0
    chunk_index: int = 0           # position within document
    page_number: Optional[int] = None
    section_ref: Optional[str] = None    # e.g., "Section 318"
    subsection_ref: Optional[str] = None # e.g., "318(2)(b)"
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    bm25_tokens: Optional[List[str]] = None  # preprocessed tokens for BM25

    def model_post_init(self, __context: Any) -> None:
        self.content_length = len(self.content)


# ─────────────────────────────────────────────
# Retrieval Models
# ─────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval with scoring."""
    chunk: LegalChunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    retrieval_source: str = "hybrid"  # bm25 | vector | hybrid


class Citation(BaseModel):
    """A verified legal citation."""
    citation_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_id: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    paragraph: Optional[int] = None
    citation_text: str          # e.g., "AIR 2025 SC 111"
    citation_type: str          # judgment | statute | notification
    court: Optional[str] = None
    year: Optional[int] = None
    relevance_note: Optional[str] = None
    verified: bool = False


# ─────────────────────────────────────────────
# Query Models
# ─────────────────────────────────────────────

class LegalEntity(BaseModel):
    """An extracted legal entity from user query."""
    text: str
    entity_type: str  # LAW_SECTION | CASE_NAME | LEGAL_CONCEPT | CRIME_TYPE | COURT
    confidence: float = 1.0
    normalized: Optional[str] = None  # canonical form


class QueryUnderstanding(BaseModel):
    """Output from the Query Understanding Agent."""
    original_query: str
    cleaned_query: str
    intent: LegalIntentType
    intent_confidence: float
    legal_entities: List[LegalEntity] = Field(default_factory=list)
    law_filter: Optional[List[LawCategory]] = None
    court_filter: Optional[List[CourtType]] = None
    year_range: Optional[Dict[str, int]] = None  # {"from": 2020, "to": 2025}
    section_refs: List[str] = Field(default_factory=list)
    expanded_queries: List[str] = Field(default_factory=list)  # for multi-query retrieval
    draft_type: Optional[DraftType] = None


# ─────────────────────────────────────────────
# Agent Pipeline Models
# ─────────────────────────────────────────────

class AgentState(BaseModel):
    """Shared state passed through the agent pipeline."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    original_query: str
    query_understanding: Optional[QueryUnderstanding] = None
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks: List[RetrievedChunk] = Field(default_factory=list)
    compressed_context: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    verified_citations: List[Citation] = Field(default_factory=list)
    raw_llm_response: Optional[str] = None
    structured_response: Optional[Dict[str, Any]] = None
    hallucination_flags: List[str] = Field(default_factory=list)
    pipeline_trace: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: Dict[str, float] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# API Request / Response Models
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    law_filter: Optional[List[LawCategory]] = None
    court_filter: Optional[List[CourtType]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    document_type: Optional[DocumentType] = None
    top_k: int = Field(default=10, ge=1, le=50)
    include_statutes: bool = True
    include_judgments: bool = True


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=3, max_length=5000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)
    law_filter: Optional[List[LawCategory]] = None
    stream: bool = Field(default=True)


class DraftRequest(BaseModel):
    draft_type: DraftType
    facts: str = Field(..., min_length=50, max_length=10000)
    parties: Dict[str, str] = Field(default_factory=dict)
    court: Optional[str] = None
    sections_involved: List[str] = Field(default_factory=list)
    additional_context: Optional[str] = None


class SummarizeRequest(BaseModel):
    document_id: Optional[str] = None
    text: Optional[str] = None
    summary_type: str = Field(default="full")  # full | brief | issue_focused


class LegalResponse(BaseModel):
    """Standard structured output for all legal queries."""
    query: str
    session_id: str
    intent: Optional[str] = None
    answer: str
    relevant_sections: List[Dict[str, Any]] = Field(default_factory=list)
    precedents: List[Dict[str, Any]] = Field(default_factory=list)
    procedural_requirements: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    hallucination_flags: List[str] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JudgmentSummary(BaseModel):
    """Structured judgment summary."""
    document_id: str
    case_name: Optional[str] = None
    citation: Optional[str] = None
    court: Optional[str] = None
    date_decided: Optional[datetime] = None
    facts: str
    issues: List[str] = Field(default_factory=list)
    arguments: Dict[str, str] = Field(default_factory=dict)  # party -> argument
    findings: str
    ratio_decidendi: Optional[str] = None
    final_order: str
    sections_discussed: List[str] = Field(default_factory=list)
    is_landmark: bool = False
    summary_brief: str  # 2-3 sentence executive summary


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks_created: int
    status: str
    message: str


# ─────────────────────────────────────────────
# User & Auth Models
# ─────────────────────────────────────────────

class UserLogin(BaseModel):
    """Minimal model for login — only email + password required."""
    email: str
    password: str


class UserCreate(BaseModel):
    email: str
    password: str = Field(..., min_length=8)
    full_name: str
    role: UserRole = UserRole.RESEARCHER
    bar_enrollment: Optional[str] = None  # for advocates


class UserInDB(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    full_name: str
    role: UserRole
    hashed_password: str
    is_active: bool = True
    bar_enrollment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None


class TokenPayload(BaseModel):
    sub: str   # user_id
    role: str
    exp: int


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


# ─────────────────────────────────────────────
# Ingestion Models
# ─────────────────────────────────────────────

class IngestionJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: str  # india_code | supreme_court | upload
    file_path: Optional[str] = None
    url: Optional[str] = None
    status: str = "pending"  # pending | processing | completed | failed
    documents_processed: int = 0
    chunks_created: int = 0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class ParsedDocument(BaseModel):
    """Output of the document parsing stage."""
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_text: str
    metadata: DocumentMetadata
    structure: Dict[str, Any] = Field(default_factory=dict)  # detected structure
    pages: int = 0
    parse_method: str = "pdfminer"  # pdfminer | tesseract | pdfplumber
    parse_quality: float = 1.0  # 0-1, lower = more OCR artifacts
