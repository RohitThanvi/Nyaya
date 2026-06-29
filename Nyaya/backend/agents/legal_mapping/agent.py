"""
Legal Mapping Agent.

Maps case facts to relevant BNS/BNSS/BSA sections.
Uses retrieved context + LLM reasoning.
Returns structured provision analysis with confidence scores.
"""
import logging
from typing import Dict, List, Optional

from backend.models.domain import (
    LawCategory, LegalResponse, QueryUnderstanding,
    RetrievedChunk, Citation
)
from backend.utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)

LEGAL_MAPPING_SYSTEM = """You are a senior Indian criminal law expert specializing in BNS (Bharatiya Nyaya Sanhita 2023), BNSS (Bharatiya Nagarik Suraksha Sanhita 2023), and BSA (Bharatiya Sakshya Adhiniyam 2023).

Your task: Given case facts and retrieved legal provisions, identify ALL applicable sections and their relevance.

STRICT RULES:
1. ONLY cite sections that appear in the provided context. NEVER invent section numbers.
2. For each section, state: what it covers, why it applies, elements that must be proved.
3. Note procedural requirements from BNSS.
4. Note evidentiary requirements from BSA.
5. If context is insufficient, explicitly state uncertainty.
6. Indicate confidence: HIGH (section clearly applies), MEDIUM (possibly applicable), LOW (tangentially related).

Return valid JSON matching this structure exactly:
{
  "relevant_sections": [
    {
      "section_number": "318",
      "law": "BNS",
      "title": "Cheating",
      "relevance": "The accused induced the complainant by deception...",
      "elements_to_prove": ["dishonest inducement", "delivery of property", "deception"],
      "confidence": "HIGH",
      "punishment": "Up to 7 years imprisonment",
      "citation_chunk_id": "<chunk_id from context>"
    }
  ],
  "procedural_requirements": [
    "FIR must be filed under BNSS Section 173",
    "Cognizable offence - police can arrest without warrant"
  ],
  "evidentiary_notes": [
    "Electronic records admissible under BSA Section 63"
  ],
  "overall_confidence": 0.85,
  "warnings": ["Section 420 IPC equivalent is BNS Section 318 - verify applicability"],
  "analysis_summary": "<2-3 sentence overall analysis>"
}"""


class LegalMappingAgent:
    """
    Maps case facts to applicable legal provisions.

    Strategy:
    1. Use retrieved chunks as ground truth for section content
    2. LLM reasons over retrieved context to identify applicable sections
    3. Every cited section must have a chunk_id reference (traceability)
    4. Confidence scoring based on provision clarity + evidence strength
    """

    def __init__(self):
        self._llm = get_llm_client()

    def _build_context_string(self, chunks: List[RetrievedChunk]) -> str:
        """Build context from retrieved chunks with chunk IDs for traceability."""
        parts = []
        for i, rc in enumerate(chunks, 1):
            chunk = rc.chunk
            meta = chunk.metadata
            header = f"[CHUNK {chunk.chunk_id[:8]}]"
            if meta.law:
                header += f" {meta.law.value}"
            if chunk.section_ref:
                header += f" Section {chunk.section_ref}"
            if meta.citation:
                header += f" ({meta.citation})"
            parts.append(f"{header}\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    async def map_facts_to_sections(
        self,
        facts: str,
        retrieved_chunks: List[RetrievedChunk],
        query_understanding: Optional[QueryUnderstanding] = None,
    ) -> Dict:
        """
        Core legal mapping: facts → applicable sections.
        Returns structured JSON with sections, confidence, and warnings.
        """
        if not retrieved_chunks:
            return {
                "relevant_sections": [],
                "procedural_requirements": [],
                "evidentiary_notes": [],
                "overall_confidence": 0.0,
                "warnings": ["No relevant legal provisions found in knowledge base."],
                "analysis_summary": "Insufficient context to map facts to legal provisions.",
            }

        context = self._build_context_string(retrieved_chunks[:8])  # Top 8 chunks

        intent_context = ""
        if query_understanding and query_understanding.section_refs:
            intent_context = f"\nSpecifically check these sections if found in context: {', '.join(query_understanding.section_refs)}"

        messages = [
            {"role": "system", "content": LEGAL_MAPPING_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"CASE FACTS:\n{facts}\n\n"
                    f"RETRIEVED LEGAL PROVISIONS:\n{context}"
                    f"{intent_context}"
                ),
            },
        ]

        try:
            result = await self._llm.complete_with_json(messages)
            return self._validate_mapping_result(result, retrieved_chunks)
        except Exception as e:
            logger.error(f"Legal mapping failed: {e}")
            return {
                "relevant_sections": [],
                "procedural_requirements": [],
                "evidentiary_notes": [],
                "overall_confidence": 0.0,
                "warnings": [f"Legal analysis unavailable: {str(e)}"],
                "analysis_summary": "Analysis could not be completed.",
            }

    def _validate_mapping_result(
        self, result: Dict, chunks: List[RetrievedChunk]
    ) -> Dict:
        """
        Validate that cited sections exist in retrieved context.
        Remove any hallucinated section references.
        """
        valid_chunk_ids = {rc.chunk.chunk_id[:8] for rc in chunks}
        valid_section_refs = set()
        for rc in chunks:
            if rc.chunk.section_ref:
                valid_section_refs.add(rc.chunk.section_ref.lower())
                valid_section_refs.add(rc.chunk.section_ref)

        validated_sections = []
        hallucination_flags = []

        for section in result.get("relevant_sections", []):
            section_num = str(section.get("section_number", ""))
            chunk_id_ref = section.get("citation_chunk_id", "")[:8] if section.get("citation_chunk_id") else ""

            # Check if chunk_id exists in our retrieved set
            if chunk_id_ref and chunk_id_ref in valid_chunk_ids:
                validated_sections.append(section)
            elif section_num.lower() in valid_section_refs or section_num in valid_section_refs:
                # Section number found in retrieved content
                validated_sections.append(section)
            else:
                hallucination_flags.append(
                    f"Section {section_num} cited but not found in retrieved context"
                )
                logger.warning(f"Potential hallucination: Section {section_num} not in context")

        result["relevant_sections"] = validated_sections
        if hallucination_flags:
            result.setdefault("warnings", []).extend(hallucination_flags)

        return result
