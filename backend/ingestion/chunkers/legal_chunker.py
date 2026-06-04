"""
Semantic Legal Chunker.

Strategy 1 — Structural (preferred):
  Judgments: split by facts/issues/arguments/findings/ratio/order sections
  Statutes: split by chapter/section/subsection/explanation/punishment

Strategy 2 — Heading-based fallback:
  Detect headers and split on them when structural markers aren't clear

Strategy 3 — Sentence-window fallback:
  For poorly-structured OCR output, use sentence-aware sliding window

Each chunk retains full context (section ref, document metadata).
Minimum chunk size: 100 chars. Maximum: 1200 chars.
"""
import logging
import re
from typing import List, Optional, Tuple

from backend.models.domain import (
    ChunkType, DocumentMetadata, DocumentType, LawCategory, LegalChunk
)

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150  # For sentence-window fallback


# ─────────────────────────────────────────────
# Structural Markers
# ─────────────────────────────────────────────

JUDGMENT_SECTION_PATTERNS = [
    (ChunkType.FACTS, [
        r"(?:^|\n)(?:FACTS|STATEMENT\s+OF\s+FACTS|FACTUAL\s+BACKGROUND|BACKGROUND|BRIEF\s+FACTS)",
    ]),
    (ChunkType.ISSUES, [
        r"(?:^|\n)(?:ISSUES?|QUESTIONS?\s+OF\s+LAW|POINTS?\s+FOR\s+DETERMINATION|ISSUES?\s+RAISED)",
    ]),
    (ChunkType.ARGUMENTS, [
        r"(?:^|\n)(?:ARGUMENTS?|SUBMISSIONS?|CONTENTIONS?|ARGUMENTS?\s+(?:OF|BY|FOR))",
    ]),
    (ChunkType.FINDINGS, [
        r"(?:^|\n)(?:FINDINGS?|OBSERVATIONS?|ANALYSIS|DISCUSSION|HELD|THE\s+COURT\s+HELD)",
    ]),
    (ChunkType.RATIO, [
        r"(?:^|\n)(?:RATIO|RATIO\s+DECIDENDI|PRINCIPLE|LEGAL\s+PRINCIPLE)",
    ]),
    (ChunkType.FINAL_ORDER, [
        r"(?:^|\n)(?:ORDER|JUDGMENT|OPERATIVE\s+ORDER|DIRECTIONS?|CONCLUSION|DISPOSED?\s+OF)",
    ]),
]

STATUTE_SECTION_PATTERNS = [
    r"^(?:CHAPTER|PART)\s+[IVXLCDM\d]+",          # Chapter/Part headings
    r"^\s*(\d+[A-Z]?)\.\s+[A-Z]",                  # Section numbers like "318. Cheating"
    r"^\s*\(([a-z]|\d+)\)\s",                       # Subsection markers
    r"^\s*Explanation[.—:\s]",                      # Explanations
    r"^\s*Punishment",                               # Punishment clauses
    r"^\s*Provided that",                            # Provisos
]

# Section number extraction for statutes
SECTION_NUMBER_RE = re.compile(
    r"^[\s]*(?:Section\s+)?(\d+[A-Z]?(?:\([a-z0-9]\))?)\s*[.—:]?\s*(.+?)$",
    re.MULTILINE
)


class LegalChunker:
    """
    Multi-strategy semantic chunker for Indian legal documents.
    """

    def chunk(
        self,
        text: str,
        metadata: DocumentMetadata,
        structure_hints: Optional[dict] = None,
    ) -> List[LegalChunk]:
        """
        Main entry point. Selects chunking strategy based on document type.
        """
        if metadata.document_type == DocumentType.JUDGMENT:
            chunks = self._chunk_judgment(text, metadata)
        elif metadata.document_type == DocumentType.STATUTE:
            chunks = self._chunk_statute(text, metadata)
        else:
            chunks = self._chunk_generic(text, metadata)

        # Fallback if structural chunking produced too few chunks
        if len(chunks) < 3 and len(text) > 1000:
            logger.info(
                f"Structural chunking produced {len(chunks)} chunks — "
                f"falling back to sentence window for doc {metadata.document_id[:8]}"
            )
            chunks = self._chunk_sentence_window(text, metadata)

        # Final validation: remove too-short chunks, split too-long ones
        chunks = self._validate_and_fix(chunks, metadata)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        logger.debug(
            f"Chunked document {metadata.document_id[:8]}: "
            f"{len(chunks)} chunks, type={metadata.document_type.value}"
        )
        return chunks

    # ──────────────────────────────────────────────
    # Judgment Chunker
    # ──────────────────────────────────────────────

    def _chunk_judgment(self, text: str, metadata: DocumentMetadata) -> List[LegalChunk]:
        """
        Split judgment by structural sections.
        Matches FACTS, ISSUES, ARGUMENTS, FINDINGS, RATIO, ORDER markers.
        """
        # Build combined pattern
        combined_patterns = []
        for chunk_type, patterns in JUDGMENT_SECTION_PATTERNS:
            for p in patterns:
                combined_patterns.append((chunk_type, re.compile(p, re.IGNORECASE | re.MULTILINE)))

        # Find all section boundaries
        boundaries: List[Tuple[int, ChunkType]] = []
        for chunk_type, pattern in combined_patterns:
            for match in pattern.finditer(text):
                boundaries.append((match.start(), chunk_type))

        if len(boundaries) < 2:
            return []

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        chunks = []
        for i, (start, ctype) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            content = text[start:end].strip()
            if len(content) >= MIN_CHUNK_CHARS:
                chunks.append(self._make_chunk(content, ctype, metadata))

        return chunks

    # ──────────────────────────────────────────────
    # Statute Chunker
    # ──────────────────────────────────────────────

    def _chunk_statute(self, text: str, metadata: DocumentMetadata) -> List[LegalChunk]:
        """
        Split statute by chapter/section/subsection structure.
        Handles BNS/BNSS/BSA format.
        """
        chunks = []
        lines = text.split("\n")
        current_section: Optional[str] = None
        current_type = ChunkType.SECTION
        current_lines: List[str] = []

        chapter_re = re.compile(r"^(?:CHAPTER|PART)\s+[IVXLCDM\d]+", re.IGNORECASE)
        section_re = re.compile(r"^(\d+[A-Z]?)\.\s+(.+?)\.?\s*$")
        punishment_re = re.compile(r"^Punishment\s*[.—:]", re.IGNORECASE)
        explanation_re = re.compile(r"^Explanation\s*[.—\d:]", re.IGNORECASE)

        def flush_current():
            nonlocal current_lines, current_section
            content = "\n".join(current_lines).strip()
            if len(content) >= MIN_CHUNK_CHARS:
                chunk = self._make_chunk(content, current_type, metadata)
                chunk.section_ref = current_section
                chunks.append(chunk)
            current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append(line)
                continue

            if chapter_re.match(stripped):
                flush_current()
                current_type = ChunkType.CHAPTER
                current_section = None
                current_lines = [line]
            elif section_re.match(stripped):
                flush_current()
                m = section_re.match(stripped)
                current_section = m.group(1) if m else None
                current_type = ChunkType.SECTION
                current_lines = [line]
            elif punishment_re.match(stripped):
                flush_current()
                current_type = ChunkType.PUNISHMENT
                current_lines = [line]
            elif explanation_re.match(stripped):
                flush_current()
                current_type = ChunkType.EXPLANATION
                current_lines = [line]
            else:
                current_lines.append(line)

        flush_current()
        return chunks

    # ──────────────────────────────────────────────
    # Generic Chunker
    # ──────────────────────────────────────────────

    def _chunk_generic(self, text: str, metadata: DocumentMetadata) -> List[LegalChunk]:
        """Heading-based splitting for notifications, circulars, uploads."""
        heading_re = re.compile(
            r"(?:^|\n)([A-Z][A-Z\s]{4,50}:?)\s*\n",
            re.MULTILINE
        )
        matches = list(heading_re.finditer(text))
        if len(matches) < 2:
            return []

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if len(content) >= MIN_CHUNK_CHARS:
                chunks.append(self._make_chunk(content, ChunkType.PASSAGE, metadata))
        return chunks

    # ──────────────────────────────────────────────
    # Sentence-Window Fallback
    # ──────────────────────────────────────────────

    def _chunk_sentence_window(
        self, text: str, metadata: DocumentMetadata
    ) -> List[LegalChunk]:
        """
        Sentence-aware sliding window chunking.
        Used when structural/heading detection fails.
        Splits on sentence boundaries, builds windows of ~800 chars.
        """
        sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_re.split(text)

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > MAX_CHUNK_CHARS and current:
                content = " ".join(current).strip()
                if len(content) >= MIN_CHUNK_CHARS:
                    chunks.append(self._make_chunk(content, ChunkType.PASSAGE, metadata))
                # Keep last sentence as overlap context
                current = [current[-1]] if current else []
                current_len = len(current[0]) if current else 0
            current.append(sentence)
            current_len += sentence_len

        if current:
            content = " ".join(current).strip()
            if len(content) >= MIN_CHUNK_CHARS:
                chunks.append(self._make_chunk(content, ChunkType.PASSAGE, metadata))

        return chunks

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _make_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        metadata: DocumentMetadata,
        section_ref: Optional[str] = None,
    ) -> LegalChunk:
        return LegalChunk(
            document_id=metadata.document_id,
            chunk_type=chunk_type,
            content=content,
            section_ref=section_ref or metadata.section,
            metadata=metadata,
        )

    def _validate_and_fix(
        self, chunks: List[LegalChunk], metadata: DocumentMetadata
    ) -> List[LegalChunk]:
        """
        Post-processing:
        - Remove chunks below minimum size
        - Split chunks that exceed maximum size
        """
        valid = []
        for chunk in chunks:
            if len(chunk.content) < MIN_CHUNK_CHARS:
                continue
            if len(chunk.content) <= MAX_CHUNK_CHARS:
                valid.append(chunk)
            else:
                # Split oversized chunk at sentence boundary
                sub_chunks = self._split_oversized(chunk, metadata)
                valid.extend(sub_chunks)
        return valid

    def _split_oversized(
        self, chunk: LegalChunk, metadata: DocumentMetadata
    ) -> List[LegalChunk]:
        """Split a chunk that's too large, preserving section reference."""
        sentence_re = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_re.split(chunk.content)

        result = []
        current = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > MAX_CHUNK_CHARS and current:
                content = " ".join(current).strip()
                sub = self._make_chunk(content, chunk.chunk_type, metadata, chunk.section_ref)
                result.append(sub)
                current = [current[-1]] if current else []
                current_len = len(current[0]) if current else 0
            current.append(sent)
            current_len += len(sent)

        if current:
            content = " ".join(current).strip()
            if len(content) >= MIN_CHUNK_CHARS:
                sub = self._make_chunk(content, chunk.chunk_type, metadata, chunk.section_ref)
                result.append(sub)

        return result if result else [chunk]
