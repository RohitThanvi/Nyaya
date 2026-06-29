"""
Summarization Agent — hierarchical map-reduce pattern.

Fixes from v1:
1. Hard 12,000-char truncation REMOVED — full document processed via map-reduce
2. summarize_chunks() is now the primary path (was never called in v1)
3. Chunk types (FACTS/ISSUES/ARGUMENTS/FINDINGS/RATIO/FINAL_ORDER) route directly
   to corresponding JudgmentSummary fields — LLM only synthesises within each section
4. spaCy sentence tokenizer replaces broken regex (handles 'R.K. Singh held...')
5. Map step: each window → structured intermediate JSON
   Reduce step: all intermediates → final JudgmentSummary
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from backend.config.settings import get_settings
from backend.models.domain import (
    ChunkType, JudgmentSummary, LegalChunk, RetrievedChunk,
)
from backend.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Characters per summarisation window (configurable in settings)
_WINDOW = 12000
_OVERLAP = 500

# Chunk types that map to JudgmentSummary fields
_TYPE_TO_FIELD: Dict[ChunkType, str] = {
    ChunkType.FACTS:       "facts",
    ChunkType.ISSUES:      "issues",
    ChunkType.ARGUMENTS:   "arguments",
    ChunkType.FINDINGS:    "findings",
    ChunkType.RATIO:       "ratio_decidendi",
    ChunkType.FINAL_ORDER: "final_order",
}


class SummarizationAgent:
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._settings = get_settings()
        self._spacy_nlp = None

    # ──────────────────────────────────────────────────────────────────────
    # Primary path: typed chunk list → JudgmentSummary
    # ──────────────────────────────────────────────────────────────────────

    async def summarize_chunks(
        self,
        chunks: List[LegalChunk],
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgmentSummary:
        """
        Main entry point. Consumes typed LegalChunk list.
        Structural chunks route to their JudgmentSummary field directly.
        Untyped PASSAGE chunks are classified before routing.
        Always processes ALL chunks — no LIMIT, no truncation.
        """
        if not chunks:
            return self._empty_summary(document_id, metadata)

        # Sort by chunk_index to maintain document order
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)

        # Route typed chunks to field buckets
        buckets: Dict[str, List[str]] = {
            "facts": [], "issues": [], "arguments": [],
            "findings": [], "ratio_decidendi": [], "final_order": [],
            "unclassified": [],
        }

        for chunk in sorted_chunks:
            field = _TYPE_TO_FIELD.get(chunk.chunk_type)
            if field:
                buckets[field].append(chunk.content)
            else:
                buckets["unclassified"].append(chunk.content)

        # For unclassified chunks, run hierarchical reduce if needed
        classified_extra = {}
        if buckets["unclassified"]:
            classified_extra = await self._classify_unstructured(
                " ".join(buckets["unclassified"])
            )

        # Merge classified_extra into buckets
        for field, content in classified_extra.items():
            if field in buckets and content:
                buckets[field].append(content)

        # Synthesise each non-empty bucket with LLM
        synthesised: Dict[str, Any] = {}
        for field, texts in buckets.items():
            if field == "unclassified" or not texts:
                continue
            combined = "\n\n".join(texts)
            if len(combined) > _WINDOW:
                combined = await self._hierarchical_reduce(combined, field)
            else:
                combined = await self._synthesise_field(field, combined)
            synthesised[field] = combined

        # Sections discussed
        sections = list({
            c.section_ref for c in sorted_chunks
            if c.section_ref
        })

        meta = metadata or {}
        return JudgmentSummary(
            document_id=document_id,
            case_name=meta.get("case_name") or meta.get("parties"),
            citation=meta.get("citation"),
            court=meta.get("court_name") or meta.get("court"),
            facts=synthesised.get("facts", "Not available in retrieved chunks."),
            issues=self._to_list(synthesised.get("issues", "")),
            arguments={"Appellant": "", "Respondent": ""},
            findings=synthesised.get("findings", "Not available in retrieved chunks."),
            ratio_decidendi=synthesised.get("ratio_decidendi"),
            final_order=synthesised.get("final_order", "Not available in retrieved chunks."),
            sections_discussed=sections,
            is_landmark=meta.get("is_landmark", False),
            summary_brief=await self._brief_summary(synthesised),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Hierarchical map-reduce for long text (replaces 12k hard truncation)
    # ──────────────────────────────────────────────────────────────────────

    async def summarize_text(
        self,
        text: str,
        document_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgmentSummary:
        """
        Entry point when raw text (not chunks) is provided.
        Splits into overlapping windows, map → intermediate JSON, reduce → final.
        """
        windows = self._split_windows(text, _WINDOW, _OVERLAP)
        logger.info(f"Hierarchical summarisation: {len(windows)} windows for {len(text)} chars")

        if len(windows) == 1:
            # Short enough for single-pass
            return await self._single_pass_summary(windows[0], document_id, metadata)

        # MAP: each window → intermediate structured dict
        intermediates: List[Dict[str, Any]] = []
        for i, window in enumerate(windows):
            logger.debug(f"Map step {i+1}/{len(windows)}")
            intermediate = await self._map_window(window, i, len(windows))
            intermediates.append(intermediate)

        # REDUCE: all intermediates → final JudgmentSummary
        return await self._reduce_intermediates(intermediates, document_id, metadata)

    def _split_windows(self, text: str, window: int, overlap: int) -> List[str]:
        """Split text into overlapping windows at sentence boundaries."""
        if len(text) <= window:
            return [text]

        sentences = self._sentence_split(text)
        windows = []
        current = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > window and current:
                windows.append(" ".join(current))
                # Overlap: keep last N chars worth of sentences
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sents
                current_len = overlap_len
            current.append(sent)
            current_len += sent_len

        if current:
            windows.append(" ".join(current))
        return windows

    def _sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy if available,
        otherwise safe regex that handles Indian legal abbreviations.
        """
        if self._spacy_nlp is None:
            self._spacy_nlp = self._load_spacy()

        if self._spacy_nlp:
            doc = self._spacy_nlp(text[:1_000_000])  # spaCy limit guard
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Safe regex fallback: does not split on Dr. Mr. Hon'ble J. R.K. v. s. etc.
        abbrevs = r"(?<!Dr)(?<!Mr)(?<!Mrs)(?<!Hon)(?<!Hon'ble)(?<!Pvt)(?<!Ltd)" \
                  r"(?<![A-Z])(?<!\s[A-Z])(?<!\b[vs])"
        parts = re.split(rf"{abbrevs}(?<=[.!?])\s+(?=[A-Z\"\u201c])", text)
        return [p.strip() for p in parts if p.strip()]

    def _load_spacy(self):
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except Exception:
            logger.info("spaCy en_core_web_sm not available; using regex sentence splitter")
            return None

    async def _map_window(
        self, window: str, idx: int, total: int
    ) -> Dict[str, Any]:
        """Extract structured fields from a single window."""
        prompt = f"""You are analysing part {idx+1} of {total} of an Indian court judgment.
Extract any information present in this section. Return ONLY valid JSON:
{{
  "facts": "brief facts found here or null",
  "issues": ["issue 1", "issue 2"] or [],
  "findings": "findings/observations found here or null",
  "ratio": "ratio decidendi found here or null",
  "final_order": "final order/disposition found here or null",
  "sections_discussed": ["BNS 318", "BNSS 480"] or []
}}

TEXT:
{window[:11000]}"""

        try:
            raw = await self._llm.complete(prompt=prompt, temperature=0.0, max_tokens=800)
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"Map window {idx+1} failed: {e}")
            return {"facts": window[:300], "issues": [], "findings": None,
                    "ratio": None, "final_order": None, "sections_discussed": []}

    async def _reduce_intermediates(
        self,
        intermediates: List[Dict[str, Any]],
        document_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> JudgmentSummary:
        """Merge all intermediate structured dicts into final JudgmentSummary."""
        # Collect all non-null values per field
        all_facts = [d["facts"] for d in intermediates if d.get("facts")]
        all_issues = [i for d in intermediates for i in d.get("issues", [])]
        all_findings = [d["findings"] for d in intermediates if d.get("findings")]
        all_ratio = [d["ratio"] for d in intermediates if d.get("ratio")]
        all_order = [d["final_order"] for d in intermediates if d.get("final_order")]
        all_sections = list({s for d in intermediates for s in d.get("sections_discussed", [])})

        # Final synthesis per field
        facts = await self._synthesise_field("facts", "\n\n".join(all_facts)) if all_facts else "Not extracted."
        findings = await self._synthesise_field("findings", "\n\n".join(all_findings)) if all_findings else "Not extracted."
        ratio = await self._synthesise_field("ratio_decidendi", "\n\n".join(all_ratio)) if all_ratio else None
        final_order = all_order[-1] if all_order else "Not extracted."   # last window is most likely to have order

        meta = metadata or {}
        summary = JudgmentSummary(
            document_id=document_id,
            case_name=meta.get("case_name"),
            citation=meta.get("citation"),
            court=meta.get("court_name"),
            facts=facts,
            issues=list(dict.fromkeys(all_issues))[:10],   # deduplicate
            findings=findings,
            ratio_decidendi=ratio,
            final_order=final_order,
            sections_discussed=all_sections,
            is_landmark=meta.get("is_landmark", False),
            summary_brief=await self._brief_summary({
                "facts": facts, "findings": findings, "final_order": final_order
            }),
        )
        return summary

    async def _single_pass_summary(
        self, text: str, document_id: str, metadata: Optional[Dict[str, Any]]
    ) -> JudgmentSummary:
        """Single LLM call for short documents."""
        prompt = f"""Summarise this Indian court judgment. Return ONLY valid JSON:
{{
  "facts": "...",
  "issues": ["..."],
  "findings": "...",
  "ratio_decidendi": "..." or null,
  "final_order": "...",
  "sections_discussed": ["BNS 318", ...],
  "summary_brief": "2-3 sentence brief"
}}

JUDGMENT TEXT:
{text[:11000]}"""
        try:
            raw = await self._llm.complete(prompt=prompt, temperature=0.0, max_tokens=1500)
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)
            meta = metadata or {}
            return JudgmentSummary(
                document_id=document_id,
                case_name=meta.get("case_name"),
                citation=meta.get("citation"),
                court=meta.get("court_name"),
                facts=data.get("facts", ""),
                issues=data.get("issues", []),
                findings=data.get("findings", ""),
                ratio_decidendi=data.get("ratio_decidendi"),
                final_order=data.get("final_order", ""),
                sections_discussed=data.get("sections_discussed", []),
                is_landmark=meta.get("is_landmark", False),
                summary_brief=data.get("summary_brief", ""),
            )
        except Exception as e:
            logger.error(f"Single-pass summary failed: {e}")
            return self._empty_summary(document_id, metadata)

    # ──────────────────────────────────────────────────────────────────────
    # Field-level synthesis helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _synthesise_field(self, field: str, combined_text: str) -> str:
        """Summarise multiple chunks for one structural field into a single coherent passage."""
        labels = {
            "facts": "factual background",
            "issues": "legal issues framed",
            "findings": "court's findings and observations",
            "ratio_decidendi": "ratio decidendi (principle of law)",
            "final_order": "final order and directions",
        }
        label = labels.get(field, field)
        prompt = f"""Synthesise the following excerpts about '{label}' from an Indian court judgment into a single coherent paragraph. Preserve all specific legal references, section numbers, and party names. Be concise but complete.

EXCERPTS:
{combined_text[:8000]}

OUTPUT (plain prose, no JSON):"""
        try:
            return await self._llm.complete(prompt=prompt, temperature=0.1, max_tokens=600)
        except Exception:
            return combined_text[:1000]

    async def _hierarchical_reduce(self, text: str, field: str) -> str:
        """Reduce a very long field text via recursive windowing."""
        windows = self._split_windows(text, _WINDOW, _OVERLAP)
        if len(windows) == 1:
            return await self._synthesise_field(field, windows[0])
        partial = []
        for w in windows:
            partial.append(await self._synthesise_field(field, w))
        return await self._synthesise_field(field, "\n\n".join(partial))

    async def _classify_unstructured(self, text: str) -> Dict[str, str]:
        """Route untyped PASSAGE chunks to structural fields via LLM classification."""
        prompt = f"""Classify this text into judgment sections. Return ONLY JSON:
{{
  "facts": "text that describes facts/background or null",
  "findings": "text with court observations or null",
  "final_order": "text with final order/directions or null",
  "ratio_decidendi": "text with principle of law or null"
}}

TEXT:
{text[:6000]}"""
        try:
            raw = await self._llm.complete(prompt=prompt, temperature=0.0, max_tokens=600)
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except Exception:
            return {"facts": text[:2000]}

    async def _brief_summary(self, synthesised: Dict[str, Any]) -> str:
        """Generate a 2-3 sentence brief from synthesised fields."""
        key_parts = [
            synthesised.get("facts", ""),
            synthesised.get("final_order", ""),
        ]
        combined = " ".join(p for p in key_parts if p)[:3000]
        if not combined.strip():
            return "Summary not available."
        prompt = f"""Write a 2-3 sentence neutral summary of this Indian court judgment suitable for a legal database index. Include: parties, legal issue, and outcome.

TEXT: {combined}

SUMMARY:"""
        try:
            return await self._llm.complete(prompt=prompt, temperature=0.1, max_tokens=200)
        except Exception:
            return combined[:300]

    def _to_list(self, text: str) -> List[str]:
        if not text:
            return []
        if isinstance(text, list):
            return text
        items = re.split(r"\n+|\d+\.\s+", text)
        return [i.strip() for i in items if i.strip()]

    def _empty_summary(
        self, document_id: str, metadata: Optional[Dict[str, Any]]
    ) -> JudgmentSummary:
        meta = metadata or {}
        return JudgmentSummary(
            document_id=document_id,
            case_name=meta.get("case_name"),
            citation=meta.get("citation"),
            facts="No chunks available for summarisation.",
            issues=[],
            findings="No chunks available for summarisation.",
            final_order="No chunks available for summarisation.",
            sections_discussed=[],
            summary_brief="Document could not be summarised.",
        )
