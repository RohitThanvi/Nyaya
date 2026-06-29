"""
Verification Agent — rebuilt from scratch.

Fixes from v1:
1. ChunkRepository is now properly injected (not optional None)
2. Citation extraction uses structured LLM call → typed ExtractedClaim list
   instead of fragile regex on raw LLM output
3. Section matching is normalised (strips parens, spaces, punctuation)
   before comparison — '318(2)(a) BNS' == '318' BNS
4. DB fallback: claims not in top-k trigger targeted DB query
   → returns source_url + page_number + snippet for frontend linking
5. False positive fix: '3' no longer matches '303' (exact normalised match only)
6. Each verified Citation carries: source_url, page_number, snippet (150 chars)
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import get_settings
from backend.models.domain import (
    Citation, ExtractedClaim, LawCategory, RetrievedChunk,
)
from backend.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── BNS / BNSS / IPC equivalence map for normalised matching ─────────────────
_LAW_ALIASES: Dict[str, List[str]] = {
    "BNS":  ["BNS", "IPC"],
    "BNSS": ["BNSS", "CrPC", "CRPC"],
    "BSA":  ["BSA", "Evidence Act"],
    "IPC":  ["IPC", "BNS"],
}


def _normalise_section(raw: str) -> str:
    """
    Strip subsection suffixes to bare number + optional letter.
    '318(2)(a)' → '318'    '3A' → '3A'    'Section 302' → '302'
    """
    cleaned = re.sub(r"(?i)section\s*", "", raw.strip())
    m = re.match(r"(\d+[A-Za-z]?)", cleaned)
    return m.group(1) if m else cleaned.lower()


def _normalise_citation(raw: str) -> str:
    """Remove punctuation noise for citation string matching."""
    return re.sub(r"[().,\s]+", " ", raw).strip().lower()


class VerificationAgent:
    """
    Verifies legal claims in LLM output against:
      1. Retrieved chunks (in-context)
      2. PostgreSQL knowledge base (DB fallback for claims not in top-k)
    Returns a list of Citation objects with full provenance.
    """

    def __init__(self, db: AsyncSession, llm_client: LLMClient):
        self._db = db
        self._llm = llm_client
        self._settings = get_settings()

    async def verify(
        self,
        llm_response: str,
        retrieved_chunks: List[RetrievedChunk],
        original_query: str,
    ) -> Tuple[List[Citation], List[str]]:
        """
        Main entry point.
        Returns (verified_citations, hallucination_flags).
        """
        # Step 1: structured extraction of claims from LLM output
        claims = await self._extract_claims(llm_response, original_query)
        if not claims:
            return [], []

        verified: List[Citation] = []
        flags: List[str] = []

        for claim in claims:
            citation, flag = await self._verify_claim(claim, retrieved_chunks)
            if citation:
                verified.append(citation)
            if flag:
                flags.append(flag)

        logger.info(
            f"Verification: {len(claims)} claims → "
            f"{len(verified)} verified, {len(flags)} flagged"
        )
        return verified, flags

    # ──────────────────────────────────────────────────────────────────────
    # Step 1 — Structured claim extraction
    # ──────────────────────────────────────────────────────────────────────

    async def _extract_claims(
        self, response_text: str, query: str
    ) -> List[ExtractedClaim]:
        """
        Ask LLM to extract structured claims from its own output.
        Returns typed ExtractedClaim list, not raw regex strings.
        """
        prompt = f"""You are a legal citation extraction system.

From the following legal response text, extract every legal citation as a JSON array.
Each item must have:
  - claim_type: "section" | "judgment" | "article" | "rule"
  - raw_text: exact text as it appears
  - law: law name if present ("BNS", "BNSS", "BSA", "IPC", "CrPC", "Constitution", etc.) or null
  - section_num: bare section number like "318" or null
  - citation_str: full citation string like "AIR 2025 SC 111" or null

Return ONLY a valid JSON array, no other text.

RESPONSE TEXT:
{response_text[:4000]}

ORIGINAL QUERY: {query}"""

        try:
            raw = await self._llm.complete(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1000,
            )
            import json
            # Strip markdown fences if present
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            items = json.loads(clean)
            claims = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                claims.append(ExtractedClaim(
                    claim_type=str(item.get("claim_type", "section")),
                    raw_text=str(item.get("raw_text", "")),
                    law=item.get("law"),
                    section_num=item.get("section_num"),
                    citation_str=item.get("citation_str"),
                ))
            return claims
        except Exception as e:
            logger.warning(f"Claim extraction failed ({e}), falling back to regex")
            return self._regex_extract_claims(response_text)

    def _regex_extract_claims(self, text: str) -> List[ExtractedClaim]:
        """Regex fallback when LLM extraction fails."""
        claims = []
        seen = set()

        # Section patterns: "Section 318 BNS", "BNS 318", "s. 318"
        for m in re.finditer(
            r"(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?)(?:\s*\([^)]+\))*"
            r"(?:\s+(?:of\s+)?(?:the\s+)?)?(BNS|BNSS|BSA|IPC|CrPC|Constitution)?",
            text, re.IGNORECASE
        ):
            num = m.group(1)
            law = (m.group(2) or "").upper() or None
            key = f"section:{num}:{law}"
            if key not in seen:
                seen.add(key)
                claims.append(ExtractedClaim(
                    claim_type="section",
                    raw_text=m.group(0).strip(),
                    law=law,
                    section_num=num,
                ))

        # Judgment citations: "AIR 2025 SC 111", "(2023) 4 SCC 200"
        for m in re.finditer(
            r"(?:AIR\s+\d{4}\s+\w+\s+\d+|"
            r"\(\d{4}\)\s+\d+\s+SCC\s+\d+|"
            r"\d{4}\s+\w+\s+\d+)",
            text, re.IGNORECASE
        ):
            raw = m.group(0).strip()
            key = f"judgment:{raw}"
            if key not in seen:
                seen.add(key)
                claims.append(ExtractedClaim(
                    claim_type="judgment",
                    raw_text=raw,
                    citation_str=raw,
                ))

        return claims

    # ──────────────────────────────────────────────────────────────────────
    # Step 2 — Verify individual claim
    # ──────────────────────────────────────────────────────────────────────

    async def _verify_claim(
        self,
        claim: ExtractedClaim,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[Optional[Citation], Optional[str]]:
        """
        Attempts to verify a claim against:
        a) In-context retrieved chunks (fast, no DB)
        b) PostgreSQL knowledge base (fallback, with source_url + page_number)
        """
        if claim.claim_type == "section" and claim.section_num:
            return await self._verify_section(claim, retrieved_chunks)
        elif claim.claim_type == "judgment" and claim.citation_str:
            return await self._verify_judgment(claim, retrieved_chunks)
        return None, None

    async def _verify_section(
        self,
        claim: ExtractedClaim,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[Optional[Citation], Optional[str]]:
        norm_num = _normalise_section(claim.section_num)
        claim_law = (claim.law or "").upper()
        alias_laws = _LAW_ALIASES.get(claim_law, [claim_law]) if claim_law else []

        # a) Check retrieved chunks first (zero DB cost)
        for rc in retrieved_chunks:
            chunk = rc.chunk
            chunk_sec = _normalise_section(chunk.section_ref or "")
            chunk_law = (chunk.metadata.law.value if chunk.metadata.law else "").upper()

            # Exact normalised match — no substring matching (fixes '3' ∈ '303' bug)
            sec_match = chunk_sec == norm_num
            law_match = (not claim_law) or (chunk_law in alias_laws) or (claim_law in alias_laws)

            if sec_match and law_match:
                meta = chunk.metadata
                return Citation(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    section=norm_num,
                    subsection=claim.raw_text if "(" in claim.raw_text else None,
                    page_number=chunk.page_number,
                    citation_text=claim.raw_text,
                    citation_type="statute",
                    court=meta.court.value if meta.court else None,
                    year=meta.year,
                    source_url=meta.source_url,
                    snippet=chunk.content[:150],
                    verified=True,
                ), None

        # b) DB fallback — targeted lookup
        return await self._db_verify_section(claim, norm_num, alias_laws)

    async def _db_verify_section(
        self,
        claim: ExtractedClaim,
        norm_num: str,
        alias_laws: List[str],
    ) -> Tuple[Optional[Citation], Optional[str]]:
        try:
            params: Dict = {"section": norm_num}
            law_clause = ""
            if alias_laws:
                law_clause = "AND d.law = ANY(:laws)"
                params["laws"] = alias_laws

            result = await self._db.execute(text(f"""
                SELECT c.chunk_id, c.document_id, c.content, c.page_number,
                       d.source_url, d.law, d.year, d.court
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE (c.section_ref = :section OR c.subsection_ref ILIKE :section_like)
                      {law_clause}
                LIMIT 1
            """), {**params, "section_like": f"{norm_num}%"})
            row = result.fetchone()
        except Exception as e:
            logger.error(f"DB section verify failed: {e}")
            row = None

        if row:
            return Citation(
                document_id=str(row.document_id),
                chunk_id=str(row.chunk_id),
                section=norm_num,
                page_number=row.page_number,
                citation_text=claim.raw_text,
                citation_type="statute",
                year=row.year,
                source_url=row.source_url,
                snippet=row.content[:150] if row.content else None,
                verified=True,
            ), None

        # Could not verify — flag as unverified
        return None, (
            f"Section {claim.raw_text} could not be verified in the knowledge base. "
            f"Please check the original statute."
        )

    async def _verify_judgment(
        self,
        claim: ExtractedClaim,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[Optional[Citation], Optional[str]]:
        norm_cit = _normalise_citation(claim.citation_str)

        # a) In-context chunks
        for rc in retrieved_chunks:
            chunk_cit = _normalise_citation(chunk.metadata.citation or "")
            if chunk_cit and chunk_cit == norm_cit:
                meta = rc.chunk.metadata
                return Citation(
                    document_id=rc.chunk.document_id,
                    chunk_id=rc.chunk.chunk_id,
                    citation_text=claim.citation_str,
                    citation_type="judgment",
                    court=meta.court.value if meta.court else None,
                    year=meta.year,
                    page_number=rc.chunk.page_number,
                    source_url=meta.source_url,
                    snippet=rc.chunk.content[:150],
                    verified=True,
                ), None

        # b) DB fallback
        try:
            result = await self._db.execute(text("""
                SELECT d.document_id, d.source_url, d.citation, d.year, d.court,
                       c.chunk_id, c.content, c.page_number
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                                   AND c.chunk_type = 'final_order'
                WHERE LOWER(REGEXP_REPLACE(d.citation, '[().,\\s]+', ' ', 'g')) = :norm_cit
                LIMIT 1
            """), {"norm_cit": norm_cit})
            row = result.fetchone()
        except Exception as e:
            logger.error(f"DB judgment verify failed: {e}")
            row = None

        if row:
            return Citation(
                document_id=str(row.document_id),
                chunk_id=str(row.chunk_id) if row.chunk_id else "",
                citation_text=claim.citation_str,
                citation_type="judgment",
                court=row.court,
                year=row.year,
                page_number=row.page_number,
                source_url=row.source_url,
                snippet=row.content[:150] if row.content else None,
                verified=True,
            ), None

        return None, (
            f"Judgment citation '{claim.citation_str}' not found in the knowledge base. "
            f"This citation may be incorrect or not yet indexed."
        )
