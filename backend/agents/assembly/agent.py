"""
Response Assembly Agent v2 — now enforces zero-hallucination at the answer level.

Responsibilities:
- Structures the final LegalResponse from all pipeline outputs
- HARD ENFORCEMENT: strips the specific sentence containing any claim that
  failed VerificationAgent's check (chunk-ID match -> fuzzy match -> DB
  fallback, in that order) before the answer ever reaches the user. This
  replaces the old behaviour of leaving the unverified claim in place behind
  a warning banner the user might not read.
- Strips internal <CHUNK:xxxxxxxx> grounding tags from the visible answer —
  they exist only for VerificationAgent to check claims mechanically.
- Builds relevant_sections and precedents lists with full Citation provenance
- Attaches source_url + page_number to every citation for frontend linking
- Computes a calibrated confidence score from retrieval quality, source
  diversity, and verification ratio (v3 — fixed double-penalization of
  hallucination flags and an unrealistically generous no-citation
  baseline; see _compute_confidence). Returns a transparent breakdown
  alongside the score, not just a bare number.
- Adds procedural requirements for procedure_query intent
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.models.domain import (
    AgentState, Citation, LegalIntentType, LegalResponse,
)

logger = logging.getLogger(__name__)

_CHUNK_TAG_PATTERN = re.compile(r"\s*<CHUNK:[a-f0-9]{4,8}>", re.IGNORECASE)

# Matches a sentence ending in a punctuation mark, used to locate and strip
# the specific sentence containing an unverifiable claim rather than
# discarding the whole answer.
_SENTENCE_SPLIT_PATTERN = re.compile(
    # Split after sentence-ending punctuation followed by whitespace and a
    # capital letter. Rejects digit-starts so citation numbers like
    # "AIR 2023 SC 214. The court held" don't split at "2" in "214".
    r'(?<=[.!?])\s+(?=[A-Z"\'])',
)


class AssemblyAgent:
    """
    Converts AgentState (all pipeline outputs) → LegalResponse (API response).
    Called as the final step in AgentPipeline.
    """

    def assemble(self, state: AgentState) -> LegalResponse:
        """Build the final structured response from all pipeline outputs."""
        qu = state.query_understanding
        intent = qu.intent.value if qu else LegalIntentType.GENERAL_QUERY.value

        raw_answer = state.raw_llm_response or "I was unable to generate a response for this query."

        # Hard enforcement: strip any sentence whose claim failed verification
        # (recorded in hallucination_flags) before the user ever sees it,
        # rather than relying on a warning banner the user might not read.
        # The <CHUNK:id> grounding tags are then removed since they are
        # internal verification scaffolding, not something a lawyer should
        # have to read in the final answer.
        answer = self._strip_unverified_claims(raw_answer, state.hallucination_flags)
        answer = _CHUNK_TAG_PATTERN.sub("", answer).strip()
        if not answer:
            answer = (
                "I could not produce a response that was fully supported by the "
                "retrieved sources. Please rephrase your question or check the "
                "relevant sections and precedents below directly."
            )

        relevant_sections = self._build_relevant_sections(state)
        precedents = self._build_precedents(state)
        procedural = self._build_procedural(state)
        confidence, confidence_breakdown = self._compute_confidence(state)
        warnings = self._build_warnings(state)

        # Build retrieval debug summary (always — lightweight)
        retrieval_debug = {
            "chunks_retrieved": len(state.reranked_chunks),
            "retrieval_sources": list({rc.retrieval_source for rc in state.reranked_chunks}),
            "top_scores": [
                {
                    "rank": i + 1,
                    "final_score": round(rc.final_score, 4),
                    "source": rc.retrieval_source,
                    "section_ref": rc.chunk.section_ref,
                    "chunk_type": rc.chunk.chunk_type.value,
                    "content_preview": rc.chunk.content[:100],
                }
                for i, rc in enumerate(state.reranked_chunks[:5])
            ],
            "latency_breakdown_ms": {
                k: round(v, 1) for k, v in state.latency_ms.items()
            },
        } if state.reranked_chunks else None

        return LegalResponse(
            query=state.original_query,
            session_id=state.session_id,
            intent=intent,
            answer=answer,
            relevant_sections=relevant_sections,
            precedents=precedents,
            procedural_requirements=procedural,
            citations=state.verified_citations,
            confidence=confidence,
            confidence_breakdown=confidence_breakdown,
            warnings=warnings,
            hallucination_flags=state.hallucination_flags,
            latency_ms=state.latency_ms.get("total_ms"),
            pipeline_trace=state.pipeline_trace,
            retrieval_debug=retrieval_debug,
        )

    def strip_grounding_artifacts(self, answer: str, hallucination_flags: List[str]) -> str:
        """
        Public entry point for callers that build a LegalResponse without
        going through assemble() (currently: AgentPipeline.run_draft) but
        still need the same hard enforcement — strip unverified-claim
        sentences, then strip internal <CHUNK:id> tags — rather than
        re-implementing it or, worse, returning CHUNK tags straight to
        the advocate reviewing the draft.
        """
        answer = self._strip_unverified_claims(answer, hallucination_flags)
        answer = _CHUNK_TAG_PATTERN.sub("", answer).strip()
        if not answer:
            answer = (
                "I could not produce a draft that was fully supported by the "
                "retrieved sources. Please check the relevant sections and "
                "precedents directly, or provide more specific facts."
            )
        return answer

    def _strip_unverified_claims(self, answer: str, hallucination_flags: List[str]) -> str:
        """
        Removes the specific sentence containing each unverified claim,
        rather than discarding the whole answer or leaving the claim in
        place behind a warning banner.

        hallucination_flags are messages like:
          "Section 999 BNS could not be verified in the knowledge base. ..."
          "Judgment citation 'AIR 2099 SC 1' not found ..."
        The raw claim text is extracted from each flag (it's quoted or
        follows a known prefix) and used to find + remove the matching
        sentence from the answer.
        """
        if not hallucination_flags or not answer:
            return answer

        sentences = _SENTENCE_SPLIT_PATTERN.split(answer)
        flagged_texts: List[str] = []

        for flag in hallucination_flags:
            # "X could not be verified..." -> "X"
            # (VerificationAgent._db_verify_section's flag wording — kept
            # in sync here; flag.raw_text already includes "Section N..."
            # when applicable, so the prefix is not duplicated)
            m = re.match(r"(.+?) could not be verified", flag)
            if m:
                flagged_texts.append(m.group(1))
                continue
            # "Judgment citation 'X' not found..." -> "X"
            m = re.match(r"Judgment citation '(.+?)' not found", flag)
            if m:
                flagged_texts.append(m.group(1))
                continue

        if not flagged_texts:
            return answer

        kept_sentences = []
        for sentence in sentences:
            contains_flagged = any(
                flagged.lower() in sentence.lower() for flagged in flagged_texts
            )
            if not contains_flagged:
                kept_sentences.append(sentence)
            else:
                logger.info(f"Stripped unverified-claim sentence from answer: {sentence[:100]}")

        return " ".join(kept_sentences).strip()

    def _build_relevant_sections(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Build statute sections list from verified citations + top retrieved chunks.
        Each item includes source_url and page_number for frontend deep-linking.
        """
        sections: List[Dict[str, Any]] = []
        seen_chunks: set = set()

        # Verified statute citations first
        for cit in state.verified_citations:
            if cit.citation_type != "statute":
                continue
            if cit.chunk_id in seen_chunks:
                continue
            seen_chunks.add(cit.chunk_id)
            item: Dict[str, Any] = {
                "section": cit.section,
                "subsection": cit.subsection,
                "citation_text": cit.citation_text,
                "snippet": cit.snippet,
                "source_url": cit.source_url,
                "page_number": cit.page_number,
                "verified": True,
            }
            if cit.source_url and cit.page_number:
                item["deep_link"] = f"{cit.source_url}#page={cit.page_number}"
            sections.append(item)

        # Fill from top retrieved chunks (statute sections not yet added)
        from backend.models.domain import ChunkType, DocumentType
        for rc in state.reranked_chunks:
            chunk = rc.chunk
            if chunk.chunk_id in seen_chunks:
                continue
            if chunk.metadata.document_type not in (DocumentType.STATUTE,) and not chunk.section_ref:
                continue
            if not chunk.section_ref:
                continue
            seen_chunks.add(chunk.chunk_id)
            meta = chunk.metadata
            item = {
                "section": chunk.section_ref,
                "subsection": chunk.subsection_ref,
                "citation_text": f"Section {chunk.section_ref} {meta.law.value if meta.law else ''}".strip(),
                "snippet": chunk.content[:200],
                "source_url": meta.source_url,
                "page_number": chunk.page_number,
                "verified": False,
                "relevance_score": round(rc.final_score, 4),
            }
            if meta.source_url and chunk.page_number:
                item["deep_link"] = f"{meta.source_url}#page={chunk.page_number}"
            sections.append(item)
            if len(sections) >= 5:
                break

        return sections

    def _build_precedents(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Build precedent judgments list from verified citations + retrieved judgment chunks.
        """
        precedents: List[Dict[str, Any]] = []
        seen_docs: set = set()

        # Verified judgment citations
        for cit in state.verified_citations:
            if cit.citation_type != "judgment":
                continue
            if cit.document_id in seen_docs:
                continue
            seen_docs.add(cit.document_id)
            item: Dict[str, Any] = {
                "citation": cit.citation_text,
                "court": cit.court,
                "year": cit.year,
                "snippet": cit.snippet,
                "source_url": cit.source_url,
                "page_number": cit.page_number,
                "verified": True,
            }
            if cit.source_url:
                page_anchor = f"#page={cit.page_number}" if cit.page_number else ""
                item["deep_link"] = f"{cit.source_url}{page_anchor}"
            precedents.append(item)

        # Fill from retrieved judgment chunks
        from backend.models.domain import DocumentType
        for rc in state.reranked_chunks:
            meta = rc.chunk.metadata
            if meta.document_type != DocumentType.JUDGMENT:
                continue
            if meta.document_id in seen_docs:
                continue
            if not meta.citation:
                continue
            seen_docs.add(meta.document_id)
            item = {
                "citation": meta.citation,
                "court": meta.court_name or (meta.court.value if meta.court else None),
                "year": meta.year,
                "snippet": rc.chunk.content[:200],
                "source_url": meta.source_url,
                "page_number": rc.chunk.page_number,
                "verified": False,
                "relevance_score": round(rc.final_score, 4),
            }
            if meta.source_url:
                page_anchor = f"#page={rc.chunk.page_number}" if rc.chunk.page_number else ""
                item["deep_link"] = f"{meta.source_url}{page_anchor}"
            precedents.append(item)
            if len(precedents) >= 5:
                break

        return precedents

    def _build_procedural(self, state: AgentState) -> List[str]:
        """Extract procedural steps from the LLM response for procedure_query intent."""
        qu = state.query_understanding
        if not qu or qu.intent != LegalIntentType.PROCEDURE_QUERY:
            return []
        if not state.raw_llm_response:
            return []
        import re
        # Extract numbered or bulleted steps from LLM output
        steps = re.findall(
            r"(?:^|\n)\s*(?:\d+[\.\)]\s+|[-•]\s+)(.+?)(?=\n\s*(?:\d+[\.\)]|[-•])|\Z)",
            state.raw_llm_response,
            re.MULTILINE | re.DOTALL,
        )
        clean = [s.strip().replace("\n", " ") for s in steps if len(s.strip()) > 10]
        return clean[:10]

    def _compute_confidence(self, state: AgentState) -> Tuple[float, Dict[str, Any]]:
        """
        Confidence = weighted combination of:
          - Retrieval quality (avg final_score of top-3 chunks, scaled down
            when fewer than 3 chunks were actually retrieved — thin evidence
            should never score the same as well-corroborated evidence)
          - Source diversity (how many distinct documents back the answer —
            five chunks from one document is weaker corroboration than
            chunks spread across multiple independent sources)
          - Verification ratio (verified / total claims extracted) — this is
            now the ONLY place hallucination flags affect the score; they
            were previously subtracted a second time via a separate penalty
            term on top of already lowering this ratio, which silently
            double-counted the same evidence and made flagged answers score
            lower than the underlying signal actually justified.

        Returns (score, breakdown) — the breakdown is surfaced to the user
        via LegalResponse.confidence_breakdown so "73% confident" is never
        an opaque number with no way to see what it's based on.
        """
        top3 = state.reranked_chunks[:3]
        if top3:
            avg_retrieval = sum(rc.final_score for rc in top3) / len(top3)
            avg_retrieval = min(avg_retrieval, 1.0)
            # Penalize thin evidence: 1 chunk retrieved is materially less
            # trustworthy than 3+, even if that single chunk scored well.
            breadth_factor = min(len(top3) / 3, 1.0)
            retrieval_score = avg_retrieval * (0.7 + 0.3 * breadth_factor)
        else:
            retrieval_score = 0.15  # no retrieval at all is a low-confidence signal, not a neutral one

        unique_docs = len({
            rc.chunk.document_id for rc in state.reranked_chunks[:5]
        }) if state.reranked_chunks else 0
        # 1 source = no diversity bonus, 3+ independent sources = full bonus
        diversity_factor = min(unique_docs / 3, 1.0) if unique_docs else 0.0

        total_flags = len(state.hallucination_flags)
        total_claims_checked = len(state.verified_citations) + total_flags
        if total_claims_checked > 0:
            verify_ratio = len(state.verified_citations) / total_claims_checked
        else:
            # No checkable claims were found at all (e.g. a purely
            # conversational reply, or retrieval came back empty). This is
            # NOT the same as "moderate confidence" — there is nothing here
            # that was actually verified against a source, which is exactly
            # the case a legal research tool should be most cautious about.
            verify_ratio = 0.35

        confidence = (
            0.45 * retrieval_score +
            0.15 * diversity_factor +
            0.40 * verify_ratio
        )

        # Hard ceiling, not an additive penalty: if even one hallucinated
        # claim was caught and stripped, the underlying generation had a
        # real defect — cap how confident we present the cleaned-up answer
        # as being, regardless of how good the surviving claims look.
        if total_flags > 0:
            confidence = min(confidence, 0.6 - min(total_flags - 1, 3) * 0.05)

        confidence = round(max(0.0, min(1.0, confidence)), 3)

        if confidence >= 0.75:
            band = "high"
        elif confidence >= 0.45:
            band = "medium"
        else:
            band = "low"

        breakdown = {
            "band": band,
            "retrieval_quality": round(retrieval_score, 3),
            "chunks_used": len(top3),
            "source_diversity": round(diversity_factor, 3),
            "unique_documents": unique_docs,
            "verification_ratio": round(verify_ratio, 3),
            "verified_claims": len(state.verified_citations),
            "flagged_claims": total_flags,
        }
        return confidence, breakdown

    def _build_warnings(self, state: AgentState) -> List[str]:
        warnings: List[str] = []

        if state.hallucination_flags:
            warnings.append(
                f"{len(state.hallucination_flags)} claim(s) in the generated response "
                f"could not be verified against the knowledge base and were removed "
                f"before this answer was shown to you. If the answer below feels "
                f"incomplete, that is likely why — check the relevant sections and "
                f"precedents directly, or rephrase your question."
            )

        top_score = state.reranked_chunks[0].final_score if state.reranked_chunks else 0
        if top_score < 0.4:
            warnings.append(
                "The retrieved sources had low relevance scores. "
                "This response may not fully address the specific legal question."
            )

        if state.query_understanding:
            qu = state.query_understanding
            if qu.intent == LegalIntentType.DRAFTING_REQUEST:
                warnings.append(
                    "Legal drafts generated by AI must be reviewed and verified "
                    "by a qualified advocate before filing or use."
                )
            if qu.law_filter and len(state.reranked_chunks) < 3:
                warnings.append(
                    f"Limited sources found for the specified law filter. "
                    f"The knowledge base may not have comprehensive coverage for this query."
                )

        return warnings
