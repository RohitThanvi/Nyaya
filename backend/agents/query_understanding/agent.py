"""
Query Understanding Agent.

Responsibilities:
1. Classify legal intent (provision lookup, case search, drafting, etc.)
2. Extract legal entities (sections, case names, crimes, courts)
3. Identify metadata filters (law, court, year range)
4. Generate expanded queries for multi-query retrieval
5. Normalize section references to canonical form
"""
import json
import logging
import re
from typing import Dict, List, Optional

from backend.config.settings import get_settings
from backend.models.domain import (
    DraftType, LawCategory, LegalEntity, LegalIntentType,
    QueryUnderstanding, CourtType
)
from backend.utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# Section number patterns for direct extraction without LLM
SECTION_PATTERNS = [
    (r"\b(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?(?:\(\d+\))?(?:\([a-z]\))?)\b", "BNS/BNSS/BSA"),
    (r"\bBNS\s+(\d+[A-Za-z]?)\b", "BNS"),
    (r"\bBNSS\s+(\d+[A-Za-z]?)\b", "BNSS"),
    (r"\bBSA\s+(\d+[A-Za-z]?)\b", "BSA"),
    (r"\bIPC\s+(\d+[A-Za-z]?)\b", "IPC"),
    (r"\bCrPC\s+(\d+[A-Za-z]?)\b", "CrPC"),
    (r"\bArticle\s+(\d+[A-Za-z]?)\b", "Constitution"),
]

LAW_KEYWORDS: Dict[str, LawCategory] = {
    "bns": LawCategory.BNS,
    "bharatiya nyaya sanhita": LawCategory.BNS,
    "bnss": LawCategory.BNSS,
    "bharatiya nagarik suraksha sanhita": LawCategory.BNSS,
    "bsa": LawCategory.BSA,
    "bharatiya sakshya adhiniyam": LawCategory.BSA,
    "ipc": LawCategory.IPC,
    "indian penal code": LawCategory.IPC,
    "crpc": LawCategory.CRPC,
    "constitution": LawCategory.CONSTITUTION,
}

COURT_KEYWORDS: Dict[str, CourtType] = {
    "supreme court": CourtType.SUPREME_COURT,
    "sc": CourtType.SUPREME_COURT,
    "high court": CourtType.HIGH_COURT,
    "hc": CourtType.HIGH_COURT,
    "district court": CourtType.DISTRICT_COURT,
    "sessions court": CourtType.DISTRICT_COURT,
    "tribunal": CourtType.TRIBUNAL,
}

# Ordered MOST-SPECIFIC FIRST. "anticipatory bail" must be checked before
# "bail application" — a query like "draft an anticipatory bail application"
# contains both substrings, and dict iteration is insertion-ordered in
# Python, so the old ordering (general before specific) always matched
# "bail application" first and silently misclassified every anticipatory
# bail request as a plain bail application — wrong template, wrong sections
# (BNSS 480 instead of 482), a real product-correctness bug, not cosmetic.
DRAFT_KEYWORDS: Dict[str, DraftType] = {
    "anticipatory bail": DraftType.ANTICIPATORY_BAIL,
    "bail application": DraftType.BAIL_APPLICATION,
    "fir quashing": DraftType.FIR_QUASHING_PETITION,
    "quashing petition": DraftType.FIR_QUASHING_PETITION,
    "legal notice": DraftType.LEGAL_NOTICE,
    "written statement": DraftType.WRITTEN_STATEMENT,
    "vakalatnama": DraftType.VAKALATNAMA,
    "affidavit": DraftType.AFFIDAVIT,
    "complaint": DraftType.COMPLAINT,
    "notice": DraftType.LEGAL_NOTICE,
}

SYSTEM_PROMPT = """You are a legal query analyst for Indian law. 
Analyze the user query and return ONLY a JSON object with this structure:

{
  "intent": "<one of: provision_lookup, case_search, procedure_query, drafting_request, summarization, general_query>",
  "intent_confidence": <0.0-1.0>,
  "legal_entities": [
    {"text": "<entity text>", "entity_type": "<LAW_SECTION|CASE_NAME|LEGAL_CONCEPT|CRIME_TYPE|COURT>", "normalized": "<canonical form>"}
  ],
  "law_filter": ["<BNS|BNSS|BSA|IPC|CrPC|Constitution>"] or null,
  "court_filter": ["<Supreme Court|High Court|District Court>"] or null,
  "year_range": {"from": <year>, "to": <year>} or null,
  "section_refs": ["<section number>"] or [],
  "expanded_queries": ["<alternative query 1>", "<alternative query 2>"],
  "draft_type": "<bail_application|anticipatory_bail|legal_notice|affidavit|complaint|fir_quashing_petition|written_statement|vakalatnama>" or null
}

Rules:
- For provision_lookup: user wants to know what a specific section says
- For case_search: user wants judgments on a topic or fact pattern
- For procedure_query: user asks about legal procedure (how to file, steps, etc.)
- For drafting_request: user wants to draft a legal document
- For summarization: user wants a document summarized
- expanded_queries: 2 semantically different phrasings of the same query (for better retrieval)
- Be precise with Indian law. BNS replaces IPC, BNSS replaces CrPC, BSA replaces Indian Evidence Act."""


class QueryUnderstandingAgent:
    """
    Hybrid rule+LLM query understanding.

    Strategy:
    1. Fast rule-based extraction first (regex patterns)
    2. LLM classification for intent + complex entity extraction
    3. Merge results (rules supplement LLM)

    This reduces LLM calls for simple section lookups while
    handling complex natural language queries properly.
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client or get_llm_client()

    def _extract_sections_regex(self, query: str) -> List[str]:
        """Fast regex-based section number extraction."""
        sections = []
        for pattern, law in SECTION_PATTERNS:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for m in matches:
                sections.append(m.group(1))
        return list(set(sections))

    def _extract_law_filters_regex(self, query: str) -> Optional[List[LawCategory]]:
        """Fast keyword-based law category extraction."""
        ql = query.lower()
        found = []
        for kw, law in LAW_KEYWORDS.items():
            if kw in ql:
                found.append(law)
        return list(set(found)) if found else None

    def _extract_court_filters_regex(self, query: str) -> Optional[List[CourtType]]:
        ql = query.lower()
        found = []
        for kw, court in COURT_KEYWORDS.items():
            if kw in ql:
                found.append(court)
        return list(set(found)) if found else None

    def _extract_draft_type_regex(self, query: str) -> Optional[DraftType]:
        ql = query.lower()
        for kw, dtype in DRAFT_KEYWORDS.items():
            if kw in ql:
                return dtype
        return None

    def _extract_year_range(self, query: str) -> Optional[Dict[str, int]]:
        """Extract year references from query."""
        years = re.findall(r"\b(19[0-9]{2}|20[0-2][0-9])\b", query)
        if len(years) == 1:
            return {"from": int(years[0]), "to": int(years[0])}
        elif len(years) >= 2:
            return {"from": min(int(y) for y in years), "to": max(int(y) for y in years)}
        return None

    async def understand(self, query: str, **kwargs) -> QueryUnderstanding:
        """
        Full query understanding pipeline.
        Returns QueryUnderstanding with all extracted information.
        """
        # Rule-based fast extraction
        sections_regex = self._extract_sections_regex(query)
        law_filter_regex = self._extract_law_filters_regex(query)
        court_filter_regex = self._extract_court_filters_regex(query)
        draft_type_regex = self._extract_draft_type_regex(query)
        year_range = self._extract_year_range(query)

        # LLM-based analysis
        try:
            llm_result = await self._llm.complete_with_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Query: {query}"},
                ]
            )
        except Exception as e:
            logger.warning(f"LLM query understanding failed, using fallback: {e}")
            llm_result = self._fallback_analysis(query)

        # Parse LLM result
        intent_str = llm_result.get("intent", "general_query")
        try:
            intent = LegalIntentType(intent_str)
        except ValueError:
            intent = LegalIntentType.GENERAL_QUERY

        intent_confidence = float(llm_result.get("intent_confidence", 0.7))

        # Parse legal entities
        legal_entities = []
        for ent_raw in llm_result.get("legal_entities", []):
            try:
                legal_entities.append(
                    LegalEntity(
                        text=ent_raw.get("text", ""),
                        entity_type=ent_raw.get("entity_type", "LEGAL_CONCEPT"),
                        confidence=0.9,
                        normalized=ent_raw.get("normalized"),
                    )
                )
            except Exception:
                continue

        # Merge rule-based and LLM filters (rules take precedence for precision)
        law_filter_llm = llm_result.get("law_filter")
        if law_filter_llm:
            try:
                law_filter_llm = [LawCategory(l) for l in law_filter_llm]
            except ValueError:
                law_filter_llm = None

        final_law_filter = law_filter_regex or law_filter_llm

        court_filter_llm = llm_result.get("court_filter")
        if court_filter_llm:
            try:
                court_filter_llm = [CourtType(c) for c in court_filter_llm]
            except ValueError:
                court_filter_llm = None

        final_court_filter = court_filter_regex or court_filter_llm

        llm_sections = llm_result.get("section_refs", [])
        final_sections = list(set(sections_regex + [str(s) for s in llm_sections]))

        year_range_llm = llm_result.get("year_range")
        final_year_range = year_range or year_range_llm

        draft_type_llm = llm_result.get("draft_type")
        if draft_type_llm:
            try:
                draft_type_llm = DraftType(draft_type_llm)
            except ValueError:
                draft_type_llm = None
        final_draft_type = draft_type_regex or draft_type_llm

        expanded_queries = llm_result.get("expanded_queries", [])[:3]

        # Clean query for retrieval
        cleaned_query = query.strip()

        return QueryUnderstanding(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            intent_confidence=intent_confidence,
            legal_entities=legal_entities,
            law_filter=final_law_filter,
            court_filter=final_court_filter,
            year_range=final_year_range,
            section_refs=final_sections,
            expanded_queries=expanded_queries,
            draft_type=final_draft_type,
        )

    def _fallback_analysis(self, query: str) -> Dict:
        """Rule-only fallback when LLM is unavailable."""
        intent = "general_query"
        ql = query.lower()

        if any(w in ql for w in ["section", "bns", "bnss", "bsa", "ipc", "article"]):
            intent = "provision_lookup"
        elif any(w in ql for w in ["draft", "write", "prepare", "application"]):
            intent = "drafting_request"
        elif any(w in ql for w in ["judgment", "case", "ruling", "held", "decided"]):
            intent = "case_search"
        elif any(w in ql for w in ["how to", "procedure", "file", "steps"]):
            intent = "procedure_query"

        return {
            "intent": intent,
            "intent_confidence": 0.6,
            "legal_entities": [],
            "law_filter": None,
            "court_filter": None,
            "year_range": None,
            "section_refs": [],
            "expanded_queries": [],
            "draft_type": None,
        }
