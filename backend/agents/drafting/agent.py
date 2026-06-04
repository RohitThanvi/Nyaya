"""
Legal Drafting Agent.

Generates structured legal documents using:
1. Template scaffolding (ensures correct legal format)
2. Retrieved context (relevant sections and precedents)
3. LLM filling (natural language for facts/arguments)

Never pure hallucination — always template + context grounded.
"""
import logging
from typing import Dict, List, Optional

from backend.models.domain import DraftType, RetrievedChunk
from backend.utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Document Templates
# ─────────────────────────────────────────────

TEMPLATES: Dict[DraftType, str] = {
    DraftType.BAIL_APPLICATION: """IN THE COURT OF {court}
AT {location}

BAIL APPLICATION NO. _____ OF {year}

IN THE MATTER OF:
{accused_name}                                    ... APPLICANT/ACCUSED

VERSUS

STATE OF {state}                                  ... RESPONDENT

APPLICATION FOR BAIL UNDER SECTION 480/481 BNSS (Formerly Section 437/439 CrPC)

MOST RESPECTFULLY SHOWETH:

1. BRIEF FACTS:
{facts_section}

2. GROUNDS FOR BAIL:
{grounds_section}

3. RELEVANT LEGAL PROVISIONS:
{legal_provisions_section}

4. PRECEDENTS:
{precedents_section}

5. UNDERTAKING:
The applicant undertakes to:
(a) Appear before the court on all dates of hearing;
(b) Not tamper with evidence or influence witnesses;
(c) Not leave the jurisdiction without prior permission of the court;
(d) Surrender passport/travel documents if directed.

PRAYER:
It is therefore most respectfully prayed that this Hon'ble Court may be pleased to:
(a) Release the applicant on bail on such terms and conditions as this Hon'ble Court may deem fit;
(b) Pass any other order(s) as this Hon'ble Court may deem appropriate.

                                                  Respectfully submitted,
Date: {date}
Place: {location}
                                                  Advocate for Applicant
                                                  Enrolment No.: ____________""",

    DraftType.ANTICIPATORY_BAIL: """IN THE HON'BLE {court}
AT {location}

ANTICIPATORY BAIL APPLICATION NO. _____ OF {year}
(Under Section 482 BNSS — Formerly Section 438 CrPC)

IN THE MATTER OF:
{applicant_name}                                  ... APPLICANT

APPLICATION FOR ANTICIPATORY BAIL

MOST RESPECTFULLY SHOWETH:

1. The applicant apprehends arrest in connection with {matter_description}.

2. BRIEF FACTS OF THE CASE:
{facts_section}

3. GROUNDS FOR ANTICIPATORY BAIL:
{grounds_section}

4. RELEVANT PROVISIONS:
{legal_provisions_section}

5. RELEVANT PRECEDENTS:
{precedents_section}

6. The applicant has no prior criminal record and is a person of standing in society.

7. The applicant is ready and willing to cooperate with the investigation.

PRAYER:
{prayer_section}

                                                  Respectfully submitted,
Date: {date}
Place: {location}
                                                  Advocate for Applicant""",

    DraftType.LEGAL_NOTICE: """LEGAL NOTICE

Sent Via: Registered Post / Speed Post

Date: {date}

To,
{recipient_name}
{recipient_address}

Subject: Legal Notice for {subject}

Sir/Madam,

Under instructions from and on behalf of my client, {sender_name}, residing at {sender_address}, I hereby serve upon you the following legal notice:

1. BACKGROUND:
{background_section}

2. ACTS/OMISSIONS:
{acts_omissions_section}

3. LEGAL PROVISIONS ATTRACTED:
{legal_provisions_section}

4. DEMAND:
{demand_section}

5. You are hereby called upon to {action_required} within {deadline} days of receipt of this notice, failing which my client shall be constrained to initiate appropriate legal proceedings against you before the competent court of law, both civil and criminal, without further notice, at your risk, cost, and consequences.

This notice is without prejudice to any other legal rights and remedies available to my client.

{sender_advocate_name}
Advocate
Bar Council Enrolment No.: {enrollment}
Address: {advocate_address}""",

    DraftType.AFFIDAVIT: """AFFIDAVIT

I, {deponent_name}, {age} years, {occupation}, resident of {address}, do hereby solemnly affirm and state on oath as follows:

1. I am the deponent herein and I am fully conversant with the facts stated hereunder.

{numbered_statements}

VERIFICATION:

I, {deponent_name}, do hereby verify that the contents of the above affidavit are true and correct to my knowledge and belief. No part of it is false and nothing material has been concealed therefrom.

Verified at {place} on this {date}.

                                                  DEPONENT

Sworn before me on {date}
at {place}

                                                  NOTARY / MAGISTRATE / OATH COMMISSIONER
                                                  Seal and Signature""",

    DraftType.COMPLAINT: """COMPLAINT UNDER SECTION 173 BNSS / SECTION 200 BNSS
(Formerly Section 190/200 CrPC)

IN THE COURT OF {court}
{location}

Date: {date}

IN THE MATTER OF:
{complainant_name}                                ... COMPLAINANT

VERSUS

{accused_names}                                   ... ACCUSED

COMPLAINT

RESPECTFULLY SHOWETH:

1. THE COMPLAINANT:
{complainant_details}

2. THE ACCUSED:
{accused_details}

3. FACTS OF THE COMPLAINT:
{facts_section}

4. OFFENCES COMMITTED:
{offences_section}

5. EVIDENCE:
{evidence_section}

6. PREVIOUS COMPLAINT/FIR (if any):
{previous_complaint}

PRAYER:
It is therefore most humbly prayed that:
{prayer_section}

                                                  Complainant
                                                  Signature: ________________
Date: {date}
Place: {location}""",
}

DRAFTING_SYSTEM = """You are an expert Indian legal drafter specializing in criminal law.

Your task: Fill in the template placeholders with legally precise, well-argued content based on:
1. The provided facts
2. The retrieved legal provisions and precedents
3. Standard legal drafting conventions

RULES:
1. Use formal legal language appropriate for court documents
2. ONLY cite sections that appear in the provided context
3. Arguments must flow logically from facts to law to relief
4. Never fabricate citations, case names, or section numbers
5. If context is insufficient, write [ADVOCATE TO VERIFY: {specific item}]
6. Include all elements needed for the specific document type

Return valid JSON:
{
  "filled_template": "<complete filled document text>",
  "sections_cited": ["BNS 318", "BNSS 173"],
  "key_arguments": ["<argument 1>", "<argument 2>"],
  "drafting_notes": ["<note for advocate 1>"],
  "confidence": 0.85
}"""


class DraftingAgent:
    """
    Legal document drafting with template + context grounding.
    """

    def __init__(self):
        self._llm = get_llm_client()

    def _get_template(self, draft_type: DraftType) -> str:
        return TEMPLATES.get(draft_type, "")

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context from relevant retrieved provisions."""
        parts = []
        for rc in chunks[:6]:
            chunk = rc.chunk
            meta = chunk.metadata
            ref = f"{meta.law.value if meta.law else ''} {chunk.section_ref or ''}".strip()
            citation = meta.citation or ""
            parts.append(f"[{ref}] [{citation}]\n{chunk.content}")
        return "\n\n".join(parts)

    async def draft(
        self,
        draft_type: DraftType,
        facts: str,
        parties: Dict[str, str],
        retrieved_chunks: List[RetrievedChunk],
        court: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> Dict:
        """
        Generate a legal document.
        Returns filled template + metadata.
        """
        template = self._get_template(draft_type)
        if not template:
            raise ValueError(f"No template for draft type: {draft_type}")

        context = self._build_context(retrieved_chunks)

        parties_str = "\n".join(f"{role}: {name}" for role, name in parties.items())
        additional = f"\nADDITIONAL CONTEXT:\n{additional_context}" if additional_context else ""

        messages = [
            {"role": "system", "content": DRAFTING_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"DOCUMENT TYPE: {draft_type.value}\n\n"
                    f"TEMPLATE:\n{template}\n\n"
                    f"FACTS:\n{facts}\n\n"
                    f"PARTIES:\n{parties_str}\n\n"
                    f"COURT: {court or 'Not specified'}\n\n"
                    f"RELEVANT LEGAL PROVISIONS:\n{context}"
                    f"{additional}"
                ),
            },
        ]

        try:
            result = await self._llm.complete_with_json(messages)
            return {
                "draft_type": draft_type.value,
                "content": result.get("filled_template", ""),
                "sections_cited": result.get("sections_cited", []),
                "key_arguments": result.get("key_arguments", []),
                "drafting_notes": result.get("drafting_notes", []),
                "confidence": float(result.get("confidence", 0.7)),
                "template_used": draft_type.value,
            }
        except Exception as e:
            logger.error(f"Drafting failed: {e}")
            return {
                "draft_type": draft_type.value,
                "content": template,  # Return bare template on failure
                "sections_cited": [],
                "key_arguments": [],
                "drafting_notes": ["Automated drafting failed. Please fill manually."],
                "confidence": 0.0,
                "template_used": draft_type.value,
            }
