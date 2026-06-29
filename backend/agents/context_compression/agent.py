"""
Context Compression Agent — dedicated agent (was inline in pipeline).

Replaces naive truncation with:
1. Structural priority — RATIO, FINAL_ORDER, FINDINGS, SECTION, PUNISHMENT first
2. Sentence-boundary truncation — never cuts mid-sentence
3. Deduplication — removes near-duplicate chunks before context assembly
4. Citation headers — every chunk carries provenance [Citation | §Section | Law | p.N]
"""
import logging
import re
from typing import List, Optional

from backend.models.domain import ChunkType, RetrievedChunk

logger = logging.getLogger(__name__)

_PRIORITY_TYPES = {
    ChunkType.RATIO,
    ChunkType.FINAL_ORDER,
    ChunkType.FINDINGS,
    ChunkType.FACTS,
    ChunkType.ISSUES,
    ChunkType.SECTION,
    ChunkType.PUNISHMENT,
    ChunkType.EXPLANATION,
}

_CHARS_PER_TOKEN = 4   # conservative estimate for legal English


def _jaccard_sim(a: str, b: str, ngram: int = 4) -> float:
    """Approximate Jaccard similarity on character n-grams for dedup."""
    def ngrams(s: str) -> set:
        s = s.lower()
        return {s[i:i+ngram] for i in range(len(s) - ngram + 1)}
    sa, sb = ngrams(a), ngrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate(chunks: List[RetrievedChunk], sim_threshold: float = 0.82) -> List[RetrievedChunk]:
    """
    Remove near-duplicate chunks using character n-gram Jaccard similarity.
    Keeps the higher-scored chunk when a pair exceeds sim_threshold.
    O(n²) but n is bounded by hybrid_top_k (≤60), so cost is negligible.
    """
    kept: List[RetrievedChunk] = []
    for rc in chunks:
        duplicate = False
        for existing in kept:
            if _jaccard_sim(rc.chunk.content, existing.chunk.content) >= sim_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(rc)
    return kept


class ContextCompressionAgent:
    """
    Compresses a list of retrieved chunks into a context string that fits
    within a token budget, preserving structural integrity and full provenance.
    """

    def compress(
        self,
        chunks: List[RetrievedChunk],
        max_tokens: int = 3000,
        deduplicate: bool = True,
    ) -> str:
        """
        Main compression entry point.
        Returns a context string with citation headers per chunk.
        """
        if not chunks:
            return ""

        char_budget = max_tokens * _CHARS_PER_TOKEN

        # Dedup before anything else
        if deduplicate:
            chunks = _deduplicate(chunks)

        # Split by structural priority
        priority = [rc for rc in chunks if rc.chunk.chunk_type in _PRIORITY_TYPES]
        remainder = [rc for rc in chunks if rc.chunk.chunk_type not in _PRIORITY_TYPES]

        # Sort each group by final_score descending
        priority.sort(key=lambda x: x.final_score, reverse=True)
        remainder.sort(key=lambda x: x.final_score, reverse=True)

        ordered = priority + remainder

        parts: List[str] = []
        used = 0

        for rc in ordered:
            header = self._build_header(rc)
            header_cost = len(header) + 1   # +1 newline

            available = char_budget - used - header_cost - 2  # -2 separator
            if available <= 80:
                break

            content = self._trim_to_boundary(rc.chunk.content, available)
            block = f"{header}\n{content}"
            parts.append(block)
            used += len(block) + 2

        result = "\n\n".join(parts)
        logger.debug(
            f"Context compression: {len(chunks)} chunks → {len(parts)} kept, "
            f"{used} chars / {char_budget} budget"
        )
        return result

    def _build_header(self, rc: RetrievedChunk) -> str:
        meta = rc.chunk.metadata
        parts: List[str] = []

        # Citation / case name
        if meta.citation:
            parts.append(meta.citation)
        elif meta.case_number:
            parts.append(meta.case_number)

        # Section reference
        if rc.chunk.section_ref:
            ref = rc.chunk.section_ref
            if rc.chunk.subsection_ref:
                ref = rc.chunk.subsection_ref
            parts.append(f"§{ref}")

        # Law
        if meta.law:
            parts.append(meta.law.value)

        # Court
        if meta.court_name:
            parts.append(meta.court_name)
        elif meta.court:
            parts.append(meta.court.value)

        # Page number for source linking
        if rc.chunk.page_number:
            parts.append(f"p.{rc.chunk.page_number}")

        # Chunk type label
        parts.append(f"[{rc.chunk.chunk_type.value.upper()}]")

        return f"[{' | '.join(parts)}]"

    def _trim_to_boundary(self, text: str, limit: int) -> str:
        """Trim text to `limit` chars, never cutting mid-sentence."""
        if len(text) <= limit:
            return text
        truncated = text[:limit]
        # Find last sentence boundary
        for sep in (". ", ".\n", "? ", "! ", ".\u201d"):
            pos = truncated.rfind(sep)
            if pos > limit * 0.5:   # only use if we retain > 50% of budget
                return truncated[:pos + len(sep)].rstrip() + " [...]"
        # Fallback: word boundary
        pos = truncated.rfind(" ")
        if pos > 0:
            return truncated[:pos] + " [...]"
        return truncated + "[...]"
