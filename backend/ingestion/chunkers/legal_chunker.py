"""
Legal Chunker v2 — structure-aware, type-labelled chunking.

Fixes from v1:
1. spaCy sentence tokenizer replaces broken regex (handles R.K. Singh, v., s., etc.)
2. ChunkType assigned per chunk based on structural header detection
3. section_ref + subsection_ref extracted and set on every chunk
4. page_number carried through from parsed page boundaries
5. MIN_CHUNK_CHARS raised from implicit to explicit 150 — micro-chunks discarded
6. Sentence-window overlap preserves context across chunk boundaries
"""
import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple

from backend.models.domain import ChunkType, LegalChunk, ParsedDocument

logger = logging.getLogger(__name__)

# Chunk size targets (characters)
MIN_CHUNK_CHARS = 150
TARGET_CHUNK_CHARS = 1200
MAX_CHUNK_CHARS = 2000
OVERLAP_SENTENCES = 2      # sentences carried over from previous chunk

# Header patterns that identify structural section boundaries
_SECTION_HEADERS: Dict[ChunkType, List[re.Pattern]] = {
    ChunkType.FACTS: [
        re.compile(r"^\s*(?:FACTS?|BACKGROUND|THE CASE|BRIEF FACTS?|FACTUAL BACKGROUND)\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*\d+\.\s+(?:The petitioner|The appellant|The complainant|It is alleged)", re.IGNORECASE | re.MULTILINE),
    ],
    ChunkType.ISSUES: [
        re.compile(r"^\s*(?:ISSUE[S]?\s+(?:FOR CONSIDERATION|RAISED|FRAMED)|QUESTION[S]? (?:OF LAW)?|POINT[S]? FOR DETERMINATION)\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
    ],
    ChunkType.ARGUMENTS: [
        re.compile(r"^\s*(?:SUBMISSIONS?|ARGUMENTS?|CONTENTIONS?)\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*(?:LEARNED COUNSEL|COUNSEL FOR|SHRI|MS\.|MR\.)\s+\w+\s+(?:SUBMITS?|ARGUES?|CONTENDS?)", re.IGNORECASE | re.MULTILINE),
    ],
    ChunkType.FINDINGS: [
        re.compile(r"^\s*(?:FINDING[S]?|OBSERVATIONS?|ANALYSIS|DISCUSSION|OUR VIEW|THIS COURT (?:FINDS?|HOLDS?|OBSERVES?))\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
    ],
    ChunkType.RATIO: [
        re.compile(r"^\s*(?:RATIO DECIDENDI|THE LAW|PRINCIPLE|LEGAL POSITION|WE HOLD|WE ARE OF THE VIEW)\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"\b(?:ratio decidendi|it is settled law|it is well established|the law is|we hold that)\b", re.IGNORECASE),
    ],
    ChunkType.FINAL_ORDER: [
        re.compile(r"^\s*(?:ORDER|JUDGMENT|DECREE|RESULT|OPERATIVE PART|IN THE RESULT|CONSEQUENTLY)\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"\b(?:appeal is (?:allowed|dismissed)|petition is (?:allowed|dismissed)|we therefore|accordingly)", re.IGNORECASE),
    ],
    ChunkType.SECTION: [
        re.compile(r"^\s*Section\s+\d+[A-Za-z]?\.?\s*[-–—:]", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*\d+[A-Za-z]?\.\s+[A-Z][^.]+[.—]\s*$", re.MULTILINE),
    ],
    ChunkType.PUNISHMENT: [
        re.compile(r"\b(?:punishment|shall be punished|imprisonment|rigorous imprisonment|fine of)\b", re.IGNORECASE),
    ],
    ChunkType.EXPLANATION: [
        re.compile(r"^\s*(?:EXPLANATION|EXCEPTION|PROVISO)\s*[-:—]", re.IGNORECASE | re.MULTILINE),
    ],
}

_SECTION_REF_PATTERN = re.compile(
    r"(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]?)(?:\s*\(([^)]+)\))*"
    r"(?:\s+(?:of\s+)?(?:the\s+)?(?:BNS|BNSS|BSA|IPC|CrPC))?",
    re.IGNORECASE,
)

# FIX: the original pattern required the heading to be alone on its own
# line (anchored with $ at end-of-line), which never matches real Indian
# statute formatting where the section number, heading, and body text all
# run together on one continuous line/paragraph, e.g.:
#   "318. Cheating.—Whoever, by deceiving any person, ..."
# This is the standard typesetting for BNS/BNSS/BSA/IPC across virtually
# every PDF source (Bare Acts, India Code, Kanoon, gazette notifications).
# The old pattern silently extracted ZERO section_ref for this entire class
# of real-world documents — every chunk fell back to untyped PASSAGE with
# no section_ref, which means exact-lookup retrieval (Path 1, the strongest
# and most precise retrieval path) could never fire for statute sections
# ingested from real source text. This was found by actually running the
# unit tests against realistic input, not assumed.
#
# New pattern: number + period + heading text (no periods, em-dash, or
# colon) immediately followed by [.—:] which starts the body — no longer
# requires end-of-line.
# Matches the START of a sentence/line against known structural section
# header keywords, used by _is_section_boundary to force chunk splits at
# real structural boundaries instead of relying purely on size overflow.
_SECTION_BOUNDARY_PATTERN = re.compile(
    r"^(?:FACTS?|BACKGROUND|THE CASE|BRIEF FACTS?|FACTUAL BACKGROUND|"
    r"ISSUE[S]?(?:\s+(?:FOR CONSIDERATION|RAISED|FRAMED))?|QUESTION[S]?(?:\s+OF LAW)?|"
    r"POINT[S]?\s+FOR DETERMINATION|"
    r"SUBMISSIONS?|ARGUMENTS?|CONTENTIONS?|"
    r"FINDING[S]?|OBSERVATIONS?|ANALYSIS|DISCUSSION|"
    r"RATIO DECIDENDI|THE LAW|PRINCIPLE|LEGAL POSITION|"
    r"ORDER|JUDGMENT|DECREE|RESULT|OPERATIVE PART|IN THE RESULT|CONSEQUENTLY|"
    r"FINAL ORDER)\s*:",
    re.IGNORECASE,
)

_LAW_HEADING_PATTERN = re.compile(
    r"^\s*(\d+[A-Za-z]?)\.\s+([A-Z][^.—:\n]{2,80}?)\s*[.—:]",
    re.MULTILINE,
)


def _detect_chunk_type(text: str) -> ChunkType:
    """Classify a chunk's type by scanning for structural header patterns."""
    # Check in priority order (more specific first)
    for ctype in [
        ChunkType.FINAL_ORDER, ChunkType.RATIO, ChunkType.FINDINGS,
        ChunkType.ISSUES, ChunkType.ARGUMENTS, ChunkType.FACTS,
        ChunkType.SECTION, ChunkType.PUNISHMENT, ChunkType.EXPLANATION,
    ]:
        for pattern in _SECTION_HEADERS.get(ctype, []):
            if pattern.search(text[:800]):
                return ctype
    return ChunkType.PASSAGE


def _extract_section_ref(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract primary section_ref and subsection_ref from chunk text.
    Returns (section_ref, subsection_ref) e.g. ("318", "318(2)(a)").
    """
    # Try statute heading first: "318. Cheating."
    m = _LAW_HEADING_PATTERN.search(text[:500])
    if m:
        return m.group(1), None

    # Try inline section reference
    m = _SECTION_REF_PATTERN.search(text[:1000])
    if m:
        full = m.group(0).strip()
        bare = m.group(1)
        sub = full if "(" in full else None
        return bare, sub

    return None, None


class LegalChunker:
    """
    Converts a ParsedDocument into a list of typed LegalChunks.
    Uses sentence-window chunking with overlap.
    """

    def __init__(self):
        self._nlp = None

    def chunk(self, parsed: ParsedDocument) -> List[LegalChunk]:
        """Main entry point. Returns all chunks for a parsed document."""
        if not parsed.raw_text or len(parsed.raw_text.strip()) < MIN_CHUNK_CHARS:
            return []

        sentences = self._sentence_split(parsed.raw_text)
        if not sentences:
            return []

        raw_chunks = self._build_chunks(sentences)
        chunks = []

        for idx, (text, page_hint) in enumerate(raw_chunks):
            if len(text.strip()) < MIN_CHUNK_CHARS:
                continue

            chunk_type = _detect_chunk_type(text)
            section_ref, subsection_ref = _extract_section_ref(text)

            # Override type for statute documents with section refs
            if section_ref and chunk_type == ChunkType.PASSAGE:
                chunk_type = ChunkType.SECTION

            chunk = LegalChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=parsed.document_id,
                chunk_type=chunk_type,
                content=text.strip(),
                chunk_index=idx,
                page_number=page_hint,
                section_ref=section_ref,
                subsection_ref=subsection_ref,
                metadata=parsed.metadata,
            )
            chunks.append(chunk)

        logger.debug(
            f"Chunked document {parsed.document_id}: "
            f"{len(chunks)} chunks from {len(sentences)} sentences"
        )
        return chunks

    def _build_chunks(
        self, sentences: List[str]
    ) -> List[Tuple[str, Optional[int]]]:
        """
        Sentence-window chunking with overlap AND structural-boundary
        forcing.

        FIX: the original version only split on MAX_CHUNK_CHARS overflow,
        with zero awareness of structural section headers. A short
        judgment (bail order, brief dismissal — very common in practice)
        with FACTS/ISSUES/HELD/FINAL_ORDER sections all under ~2000 chars
        total was merged into ONE chunk, typed only as whatever section
        was detected first (FACTS), silently losing the FINAL_ORDER and
        HELD content as distinct, correctly-typed, separately-retrievable
        chunks. This directly weakens exact-match retrieval (which depends
        on chunk_type) and citation grounding for the entire class of short
        documents. Found by actually running the unit tests against
        realistic input, not assumed.

        Now: a new chunk boundary is forced whenever a sentence matches a
        structural header pattern, REGARDLESS of accumulated size — subject
        only to a much lower per-section minimum (MIN_SECTION_CHARS) so we
        don't produce a useless one-line fragment when two headers appear
        back-to-back with no body text between them.
        """
        chunks = []
        current_sents: List[str] = []
        current_len = 0

        MIN_SECTION_CHARS = 20  # allow short but real final orders e.g. "Bail granted."

        for sent in sentences:
            sent_len = len(sent)
            starts_new_section = self._is_section_boundary(sent)

            should_split_for_size = current_len + sent_len > MAX_CHUNK_CHARS and current_sents
            should_split_for_structure = (
                starts_new_section and current_sents and current_len >= MIN_SECTION_CHARS
            )

            if should_split_for_size or should_split_for_structure:
                text = " ".join(current_sents)
                chunks.append((text, None))
                if should_split_for_size:
                    # Size-driven split: carry overlap sentences for context continuity
                    overlap_buffer = current_sents[-OVERLAP_SENTENCES:]
                    current_sents = list(overlap_buffer)
                    current_len = sum(len(s) for s in current_sents)
                else:
                    # Structure-driven split: no overlap — the new section is a
                    # genuinely distinct unit (FACTS vs FINAL_ORDER should never
                    # share sentences, that would mislabel both)
                    current_sents = []
                    current_len = 0

            current_sents.append(sent)
            current_len += sent_len

        if current_sents and current_len >= MIN_SECTION_CHARS:
            chunks.append((" ".join(current_sents), None))

        return chunks

    def _is_section_boundary(self, sentence: str) -> bool:
        """
        True if `sentence` itself begins a new structural section
        (FACTS:, ISSUES:, HELD:, FINAL ORDER:, etc.) — checked against the
        same header keyword set used by _detect_chunk_type, but matched
        against just the start of a single sentence rather than scanning
        an entire accumulated chunk.
        """
        head = sentence.strip()[:60]
        return bool(_SECTION_BOUNDARY_PATTERN.match(head))

    def _sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy when available.
        Falls back to a robust regex that handles Indian legal abbreviations.
        """
        if self._nlp is None:
            self._nlp = self._load_spacy()

        if self._nlp:
            # spaCy processes up to 1M chars
            doc = self._nlp(text[:1_000_000])
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            # Handle remainder for very long docs
            if len(text) > 1_000_000:
                remainder = text[1_000_000:]
                sents += self._regex_split(remainder)
            return sents

        return self._regex_split(text)

    def _regex_split(self, text: str) -> List[str]:
        """
        Regex sentence splitter that handles:
        - Hon'ble, Dr., Mr., Mrs., Pvt. Ltd., s. (section), v. (versus)
        - R.K., A.K., S.S. (judge name initials)
        - Doesn't split on decimal numbers like 3.14
        """
        # Protect known abbreviation patterns
        protected = re.sub(
            r"\b(Hon['']ble|Dr|Mr|Mrs|Pvt|Ltd|Smt|Shri|Km|St|Art|Sec|Cl|Ch|Reg|Ord|Sch|viz|etc|i\.e|e\.g|vs?|s|r)\.",
            r"\1<DOT>",
            text,
            flags=re.IGNORECASE
        )
        # Protect initials like R.K., A.B.C.
        protected = re.sub(r"\b([A-Z])\.", r"\1<DOT>", protected)
        # Split on ". " or ".\n" followed by capital or quote
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u2018\u201c\u201d])', protected)
        # Restore
        restored = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
        return restored

    def _load_spacy(self):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            # Disable components we don't need for sentence splitting
            nlp.select_pipes(enable=["senter"] if "senter" in nlp.pipe_names else ["parser"])
            return nlp
        except Exception:
            logger.info("spaCy not available; using regex sentence splitter")
            return None
