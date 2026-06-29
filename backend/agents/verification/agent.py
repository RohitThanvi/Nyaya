"""
Verification Agent v3 — chunk-ID-grounded, no fuzzy-match-only fallback.

Root-cause fixes from v2, found by reading the code directly:

1. CRASHING BUG FIXED: _verify_judgment referenced an undefined variable
   `chunk` (copy-paste leftover from _verify_section's `for rc in ...`
   loop, where the loop var was `rc` but the body used `chunk`). This threw
   a NameError on every single judgment-citation claim, which the pipeline's
   broad try/except silently swallowed — meaning judgment citations were
   NEVER actually verified, just silently passed through with neither a
   verified=True citation NOR a hallucination flag. This was the single
   most severe correctness bug in the whole pipeline: half of all citation
   types had zero real verification while appearing to work.

2. PRIMARY VERIFICATION PATH CHANGED: claims are now checked first against
   the <CHUNK:xxxxxxxx> tag the LLM was instructed to emit (see
   _PROMPTS in pipeline.py and ContextCompressionAgent._build_header).
   This is a mechanical, exact match against chunk_id[:8] of what was
   ACTUALLY retrieved for THIS query — not a fuzzy section-number or
   citation-string match that could accidentally match an unrelated chunk
   that happens to share a number. Fuzzy matching remains as a fallback
   for responses that don't include the tag (e.g. user-facing prose where
   we strip the tags before display), but the chunk-ID path is now tried
   first and is strictly stronger evidence of grounding.

3. HARD ENFORCEMENT: claims that fail BOTH the chunk-ID check and the DB
   fallback are not just flagged — the offending sentence is now stripped
   from the final answer by AssemblyAgent (see agents/assembly/agent.py)
   so an unverifiable claim never reaches the user dressed up as fact.
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

_LAW_ALIASES: Dict[str, List[str]] = {
    "BNS":  ["BNS", "IPC"],
    "BNSS": ["BNSS", "CrPC", "CRPC"],
    "BSA":  ["BSA", "Evidence Act"],
    "IPC":  ["IPC", "BNS"],
}

_CHUNK_TAG_PATTERN = re.compile(r"<CHUNK:([a-f0-9]{4,8})>", re.IGNORECASE)


def _normalise_section(raw: str) -> str:
    cleaned = re.sub(r"(?i)section\s*", "", raw.strip())
    m = re.match(r"(\d+[A-Za-z]?)", cleaned)
    return m.group(1) if m else cleaned.lower()


def _normalise_citation(raw: str) -> str:
    return re.sub(r"[().,\s]+", " ", raw).strip().lower()


class VerificationAgent:
    """
    Verifies legal claims in LLM output against:
      1. <CHUNK:id> tags the LLM was instructed to emit — exact, mechanical,
         strongest evidence (primary path)
      2. Fuzzy section/citation matching against retrieved chunks (fallback,
         for responses without tags — e.g. already-stripped display text)
      3. PostgreSQL knowledge base — last-resort fallback with source_url
         and page_number for frontend linking
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
        """Main entry point. Returns (verified_citations, hallucination_flags)."""
        claims = await self._extract_claims(llm_response, original_query)
        if not claims:
            return [], []

        chunk_by_short_id = {rc.chunk.chunk_id[:8]: rc for rc in retrieved_chunks}

        verified: List[Citation] = []
        flags: List[str] = []

        for claim in claims:
            citation, flag = await self._verify_claim(claim, retrieved_chunks, chunk_by_short_id)
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
        Extract claims with their accompanying <CHUNK:id> tag when present.
        chunk_tag is the new field carrying the mechanical pointer.
        """
        prompt = f"""You are a legal citation extraction system.

From the following legal response text, extract every legal citation as a JSON array.
Each item must have:
  - claim_type: "section" | "judgment" | "article" | "rule"
  - raw_text: exact text as it appears
  - law: law name if present ("BNS", "BNSS", "BSA", "IPC", "CrPC", "Constitution", etc.) or null
  - section_num: bare section number like "318" or null
  - citation_str: full citation string like "AIR 2025 SC 111" or null
  - chunk_tag: the 8-character id from a nearby <CHUNK:xxxxxxxx> tag if one
    appears adjacent to this claim in the text, or null if none is present

Return ONLY a valid JSON array, no other text.

RESPONSE TEXT:
{response_text[:4000]}

ORIGINAL QUERY: {query}"""

        try:
            raw = await self._llm.complete(prompt=prompt, temperature=0.0, max_tokens=1200)
            import json
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
                    chunk_tag=item.get("chunk_tag"),
                ))
            return claims
        except Exception as e:
            logger.warning(f"Claim extraction failed ({e}), falling back to regex")
            return self._regex_extract_claims(response_text)

    def _regex_extract_claims(self, response_text: str) -> List[ExtractedClaim]:
        """
        Regex fallback. Also extracts the nearest preceding <CHUNK:id> tag
        within 200 chars, so the chunk-ID verification path still works even
        when structured LLM extraction fails.
        """
        claims = []
        seen = set()

        def _nearest_chunk_tag(match_start: int, match_end: int) -> Optional[str]:
            """
            FIX: the grounding prompt instructs the LLM to place the
            <CHUNK:id> tag IMMEDIATELY AFTER the claim it supports —
            e.g. "Section 318 BNS defines cheating <CHUNK:a1b2c3d4>." —
            but this originally only searched BACKWARD from the claim's
            start position, so it could never find a tag that comes after
            the claim, which is the only place the LLM is ever told to put
            it. Verified directly: a forward-only fix here was the
            difference between the mechanical chunk-ID path actually firing
            versus silently falling back to fuzzy matching on every real
            response. Now searches both directions: forward first (the
            documented, expected position), backward as a fallback for
            responses that don't follow the exact format.
            """
            forward_window = response_text[match_end:match_end + 200]
            forward_tags = _CHUNK_TAG_PATTERN.findall(forward_window)
            if forward_tags:
                return forward_tags[0]

            backward_window = response_text[max(0, match_start - 200):match_start]
            backward_tags = _CHUNK_TAG_PATTERN.findall(backward_window)
            return backward_tags[-1] if backward_tags else None

        for m in re.finditer(
            r"(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?)(?:\s*\([^)]+\))*"
            r"(?:\s+(?:of\s+)?(?:the\s+)?)?(BNS|BNSS|BSA|IPC|CrPC|Constitution)?",
            response_text, re.IGNORECASE
        ):
            num = m.group(1)
            law = (m.group(2) or "").upper() or None
            key = f"section:{num}:{law}"
            if key not in seen:
                seen.add(key)
                claim = ExtractedClaim(
                    claim_type="section", raw_text=m.group(0).strip(),
                    law=law, section_num=num,
                    chunk_tag=_nearest_chunk_tag(m.start(), m.end()),
                )
                claims.append(claim)

        for m in re.finditer(
            r"(?:AIR\s+\d{4}\s+\w+\s+\d+|"
            r"\(\d{4}\)\s+\d+\s+SCC\s+\d+|"
            r"\d{4}\s+\w+\s+\d+)",
            response_text, re.IGNORECASE
        ):
            raw = m.group(0).strip()
            key = f"judgment:{raw}"
            if key not in seen:
                seen.add(key)
                claim = ExtractedClaim(
                    claim_type="judgment", raw_text=raw, citation_str=raw,
                    chunk_tag=_nearest_chunk_tag(m.start(), m.end()),
                )
                claims.append(claim)

        return claims

    # ──────────────────────────────────────────────────────────────────────
    # Step 2 — Verify individual claim
    # ──────────────────────────────────────────────────────────────────────

    async def _verify_claim(
        self,
        claim: ExtractedClaim,
        retrieved_chunks: List[RetrievedChunk],
        chunk_by_short_id: Dict[str, RetrievedChunk],
    ) -> Tuple[Optional[Citation], Optional[str]]:
        # Primary path: exact chunk-ID match — but ONLY trusted after
        # confirming the tagged chunk's actual content matches what the
        # claim asserts (section number or citation string). Without this
        # cross-check, a hallucinating LLM could tag a REAL retrieved chunk
        # while asserting a DIFFERENT, fabricated section number or
        # citation — e.g. "Section 999 BNS ... <CHUNK:a1b2c3d4>" where
        # a1b2c3d4 is actually about Section 318. Trusting the tag alone
        # would mark that as verified=True, which is worse than no
        # mechanical check at all. Found by directly testing this exact
        # scenario, not assumed safe.
        chunk_tag = claim.chunk_tag
        if chunk_tag and chunk_tag.lower() in chunk_by_short_id:
            rc = chunk_by_short_id[chunk_tag.lower()]
            if self._claim_matches_chunk(claim, rc):
                return self._citation_from_chunk(claim, rc), None
            logger.warning(
                f"Claim '{claim.raw_text}' tagged <CHUNK:{chunk_tag}> but that "
                f"chunk's content does not match the claim — tag rejected, "
                f"falling through to fuzzy/DB verification."
            )

        if claim.claim_type == "section" and claim.section_num:
            return await self._verify_section(claim, retrieved_chunks)
        elif claim.claim_type == "judgment" and claim.citation_str:
            return await self._verify_judgment(claim, retrieved_chunks)
        return None, None

    def _claim_matches_chunk(self, claim: ExtractedClaim, rc: RetrievedChunk) -> bool:
        """
        Confirms the tagged chunk's actual section_ref or citation genuinely
        supports the specific claim text, not just that the chunk_id exists
        somewhere in the retrieved set. This is what makes the chunk-tag
        path strictly ADDITIVE evidence on top of content matching, rather
        than a bypass of it.
        """
        chunk = rc.chunk
        if claim.claim_type in ("section", "article", "rule") and claim.section_num:
            chunk_sec = _normalise_section(chunk.section_ref or "")
            claim_sec = _normalise_section(claim.section_num)
            return bool(chunk_sec) and chunk_sec == claim_sec

        if claim.claim_type == "judgment" and claim.citation_str:
            chunk_cit = _normalise_citation(chunk.metadata.citation or "")
            claim_cit = _normalise_citation(claim.citation_str)
            return bool(chunk_cit) and chunk_cit == claim_cit

        return False

    def _citation_from_chunk(self, claim: ExtractedClaim, rc: RetrievedChunk) -> Citation:
        chunk = rc.chunk
        meta = chunk.metadata
        citation_type = "statute" if claim.claim_type in ("section", "article", "rule") else "judgment"
        return Citation(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            section=chunk.section_ref,
            subsection=chunk.subsection_ref,
            page_number=chunk.page_number,
            citation_text=claim.raw_text,
            citation_type=citation_type,
            court=meta.court.value if meta.court else None,
            year=meta.year,
            source_url=meta.source_url,
            snippet=chunk.content[:150],
            verified=True,
        )

    async def _verify_section(
        self,
        claim: ExtractedClaim,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[Optional[Citation], Optional[str]]:
        norm_num = _normalise_section(claim.section_num)
        claim_law = (claim.law or "").upper()
        alias_laws = _LAW_ALIASES.get(claim_law, [claim_law]) if claim_law else []

        for rc in retrieved_chunks:
            chunk = rc.chunk
            chunk_sec = _normalise_section(chunk.section_ref or "")
            chunk_law = (chunk.metadata.law.value if chunk.metadata.law else "").upper()

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

        return None, (
            f"{claim.raw_text} could not be verified in the knowledge base. "
            f"This claim has been removed from the answer — please check the original statute."
        )

    async def _verify_judgment(
        self,
        claim: ExtractedClaim,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[Optional[Citation], Optional[str]]:
        """
        FIX: loop variable is `rc`, body now correctly uses `rc.chunk` —
        the v2 bug referenced an undefined `chunk` name here, which raised
        NameError on every call and was silently swallowed by the pipeline's
        broad except clause, meaning this method never actually ran to
        completion for ANY judgment citation.
        """
        norm_cit = _normalise_citation(claim.citation_str)

        for rc in retrieved_chunks:
            chunk_cit = _normalise_citation(rc.chunk.metadata.citation or "")
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
            f"This claim has been removed from the answer — the citation may be incorrect "
            f"or not yet indexed."
        )
