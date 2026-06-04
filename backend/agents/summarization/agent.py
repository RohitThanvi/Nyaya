"""
Summarization Agent.

Produces structured judgment summaries with:
- Facts, Issues, Arguments, Findings, Ratio, Final Order
- Executive summary
- Key sections discussed
"""
import logging
from typing import Dict, List, Optional

from backend.models.domain import JudgmentSummary, RetrievedChunk
from backend.utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)

SUMMARIZATION_SYSTEM = """You are a legal research assistant specializing in Indian law.

Summarize the provided judgment or legal text into structured components.

Return ONLY valid JSON:
{
  "facts": "<detailed statement of facts>",
  "issues": ["<legal issue 1>", "<legal issue 2>"],
  "arguments": {
    "petitioner": "<petitioner arguments>",
    "respondent": "<respondent arguments>"
  },
  "findings": "<court findings on each issue>",
  "ratio_decidendi": "<the binding legal principle>",
  "final_order": "<the operative order>",
  "sections_discussed": ["BNS 318", "BNSS 173"],
  "summary_brief": "<2-3 sentence executive summary>",
  "is_landmark": <true|false>
}

RULES:
- Be precise and factual. Do not add information not in the text.
- ratio_decidendi: the principle that is binding on future courts.
- sections_discussed: ONLY sections explicitly mentioned in the text.
- summary_brief: must mention the core legal issue and outcome."""


class SummarizationAgent:
    """
    Structured judgment summarization.
    Works on both full documents and retrieved chunk sets.
    """

    def __init__(self):
        self._llm = get_llm_client()

    async def summarize_chunks(
        self,
        chunks: List[RetrievedChunk],
        document_id: str,
        metadata: Optional[Dict] = None,
    ) -> JudgmentSummary:
        """Summarize a judgment from its retrieved chunks."""
        # Order chunks by index for coherent reading
        ordered = sorted(chunks, key=lambda rc: rc.chunk.chunk_index)
        full_text = "\n\n".join(rc.chunk.content for rc in ordered)
        return await self.summarize_text(full_text, document_id, metadata)

    async def summarize_text(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict] = None,
    ) -> JudgmentSummary:
        """Summarize from raw text."""
        # Truncate to fit context window (approx 12k chars = ~3k tokens)
        MAX_CHARS = 12000
        truncated = text[:MAX_CHARS] + ("\n[TEXT TRUNCATED]" if len(text) > MAX_CHARS else "")

        meta_context = ""
        if metadata:
            parts = []
            if metadata.get("citation"):
                parts.append(f"Citation: {metadata['citation']}")
            if metadata.get("court"):
                parts.append(f"Court: {metadata['court']}")
            if metadata.get("year"):
                parts.append(f"Year: {metadata['year']}")
            if parts:
                meta_context = "DOCUMENT INFO: " + " | ".join(parts) + "\n\n"

        messages = [
            {"role": "system", "content": SUMMARIZATION_SYSTEM},
            {"role": "user", "content": f"{meta_context}TEXT TO SUMMARIZE:\n{truncated}"},
        ]

        try:
            result = await self._llm.complete_with_json(messages)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            result = {
                "facts": "Summarization unavailable.",
                "issues": [],
                "arguments": {},
                "findings": "Summarization unavailable.",
                "ratio_decidendi": None,
                "final_order": "Summarization unavailable.",
                "sections_discussed": [],
                "summary_brief": "Could not generate summary.",
                "is_landmark": False,
            }

        return JudgmentSummary(
            document_id=document_id,
            case_name=metadata.get("case_name") if metadata else None,
            citation=metadata.get("citation") if metadata else None,
            court=metadata.get("court") if metadata else None,
            facts=result.get("facts", ""),
            issues=result.get("issues", []),
            arguments=result.get("arguments", {}),
            findings=result.get("findings", ""),
            ratio_decidendi=result.get("ratio_decidendi"),
            final_order=result.get("final_order", ""),
            sections_discussed=result.get("sections_discussed", []),
            is_landmark=result.get("is_landmark", False),
            summary_brief=result.get("summary_brief", ""),
        )
