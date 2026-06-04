"""
Verification Agent.

Post-generation verification:
1. Check every cited section exists in retrieved context
2. Check citation strings match stored documents
3. Flag statements unsupported by context
4. Compute verifiability score
"""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from backend.db.repositories.chunk_repo import ChunkRepository
from backend.models.domain import Citation, RetrievedChunk

logger = logging.getLogger(__name__)

# Patterns to detect legal claims in LLM output
LEGAL_CLAIM_PATTERNS = [
    r"[Ss]ection\s+(\d+[A-Za-z]?(?:\(\d+\))?)",
    r"\bBNS\s+(\d+[A-Za-z]?)\b",
    r"\bBNSS\s+(\d+[A-Za-z]?)\b",
    r"\bBSA\s+(\d+[A-Za-z]?)\b",
    r"\bIPC\s+(\d+[A-Za-z]?)\b",
    r"AIR\s+\d{4}\s+SC\s+\d+",           # citation patterns
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
    r"\d{4}\s+SCR\s+\d+",
]


class VerificationAgent:
    """
    Citation and hallucination verification.

    Process:
    1. Extract all section/citation references from LLM output
    2. Cross-check against retrieved chunks (ground truth)
    3. Query DB for sections not in retrieved context
    4. Flag anything that can't be verified
    5. Return hallucination flags + verifiability score
    """

    def __init__(self, chunk_repo: Optional[ChunkRepository] = None):
        self._chunk_repo = chunk_repo

    def _extract_claims(self, text: str) -> Set[str]:
        """Extract all legal references from a text string."""
        claims = set()
        for pattern in LEGAL_CLAIM_PATTERNS:
            matches = re.findall(pattern, text)
            for m in matches:
                claims.add(m.strip())
        return claims

    def _build_known_refs(self, chunks: List[RetrievedChunk]) -> Set[str]:
        """Build set of verifiable references from retrieved context."""
        known = set()
        for rc in chunks:
            if rc.chunk.section_ref:
                known.add(rc.chunk.section_ref.lower())
                known.add(rc.chunk.section_ref)
            if rc.chunk.subsection_ref:
                known.add(rc.chunk.subsection_ref.lower())
            if rc.chunk.metadata.citation:
                known.add(rc.chunk.metadata.citation)
            # Also add numbers only (e.g., "318" extracted from "Section 318")
            if rc.chunk.section_ref:
                num_only = re.sub(r"[^\d]", "", rc.chunk.section_ref)
                if num_only:
                    known.add(num_only)
        return known

    async def verify(
        self,
        llm_response: str,
        retrieved_chunks: List[RetrievedChunk],
        mapping_result: Optional[Dict] = None,
    ) -> Tuple[List[str], List[Citation], float]:
        """
        Verify LLM response against retrieved context.

        Returns:
            hallucination_flags: List of flagged unverified claims
            verified_citations: List of citations that passed verification
            verifiability_score: 0.0-1.0 (1.0 = all claims verified)
        """
        hallucination_flags = []
        verified_citations = []

        known_refs = self._build_known_refs(retrieved_chunks)
        claims_in_response = self._extract_claims(llm_response)

        if not claims_in_response:
            return [], [], 1.0  # No claims made → nothing to verify

        verified_count = 0
        for claim in claims_in_response:
            claim_lower = claim.lower()
            # Check exact match or number match
            is_verified = (
                claim in known_refs or
                claim_lower in known_refs or
                any(claim in ref or claim_lower in ref.lower() for ref in known_refs)
            )

            if is_verified:
                verified_count += 1
                # Build citation object
                matching_chunk = next(
                    (rc for rc in retrieved_chunks
                     if rc.chunk.section_ref and claim in rc.chunk.section_ref),
                    None
                )
                if matching_chunk:
                    verified_citations.append(
                        Citation(
                            document_id=matching_chunk.chunk.document_id,
                            chunk_id=matching_chunk.chunk.chunk_id,
                            section=matching_chunk.chunk.section_ref,
                            citation_text=matching_chunk.chunk.metadata.citation or claim,
                            citation_type=matching_chunk.chunk.metadata.document_type.value,
                            court=matching_chunk.chunk.metadata.court_name,
                            year=matching_chunk.chunk.metadata.year,
                            verified=True,
                        )
                    )
            else:
                # Check DB if chunk_repo is available
                if self._chunk_repo:
                    try:
                        exists = await self._chunk_repo.section_exists(claim)
                        if exists:
                            verified_count += 1
                        else:
                            hallucination_flags.append(
                                f"Section/citation '{claim}' could not be verified in knowledge base"
                            )
                    except Exception:
                        hallucination_flags.append(
                            f"Could not verify '{claim}' — treat with caution"
                        )
                else:
                    # Without DB access, flag uncertain claims
                    if len(claim) > 3:  # Skip very short matches (noise)
                        hallucination_flags.append(
                            f"Claim '{claim}' not found in retrieved context — verify independently"
                        )

        verifiability_score = verified_count / len(claims_in_response) if claims_in_response else 1.0

        # Also verify sections from mapping_result
        if mapping_result:
            mapping_sections = [
                s.get("section_number", "") 
                for s in mapping_result.get("relevant_sections", [])
            ]
            for sec in mapping_sections:
                if sec and sec not in known_refs and sec.lower() not in known_refs:
                    hallucination_flags.append(
                        f"Mapped section '{sec}' not in retrieved context"
                    )

        if hallucination_flags:
            logger.warning(
                f"Verification: {len(hallucination_flags)} unverified claims detected. "
                f"Verifiability: {verifiability_score:.2f}"
            )
        else:
            logger.debug(
                f"Verification passed. Verifiability: {verifiability_score:.2f}"
            )

        return hallucination_flags, verified_citations, verifiability_score

    def confidence_from_verifiability(
        self, verifiability_score: float, retrieval_score: float
    ) -> float:
        """
        Compute overall response confidence.
        Blend retrieval quality with citation verifiability.
        """
        return min(1.0, 0.5 * verifiability_score + 0.5 * retrieval_score)
