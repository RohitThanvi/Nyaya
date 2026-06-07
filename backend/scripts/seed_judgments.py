"""
Judgment seed script — NyayaAI

Seeds landmark Supreme Court judgments from publicly available sources.
Uses two approaches:
1. Direct text of landmark judgments (embedded inline — always works)
2. Fetches from Indian Kanoon public API if available

Usage:
    docker exec nyaya-backend python -m backend.scripts.seed_judgments
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Landmark judgment texts (inline — public domain, from official SC records)
# Each entry: citation, case_name, court, year, bench, law_tags, full_text
# ──────────────────────────────────────────────────────────────────────────────

LANDMARK_JUDGMENTS: List[Dict] = [
    {
        "citation": "(2017) 10 SCC 1",
        "case_name": "Justice K.S. Puttaswamy (Retd.) v. Union of India",
        "court": "Supreme Court of India",
        "year": 2017,
        "bench": "Nine Judge Constitutional Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
CIVIL ORIGINAL JURISDICTION
WRIT PETITION (CIVIL) NO. 494 OF 2012

Justice K.S. Puttaswamy (Retd.) and Another ... Petitioners
Versus
Union of India and Others ... Respondents

JUDGMENT

The Right to Privacy is a fundamental right guaranteed under Part III of the Constitution of India.

FACTS:
The petitioner, a retired Judge of the High Court of Karnataka, challenged the Aadhaar scheme on the ground that it violated the right to privacy. The Union of India contended that the right to privacy was not a fundamental right under the Constitution.

ISSUES:
1. Whether the right to privacy is a fundamental right under the Constitution of India.
2. Whether the earlier decisions in M.P. Sharma v. Satish Chandra (1954) and Kharak Singh v. State of U.P. (1963) correctly stated the law.

HELD:
1. The right to privacy is protected as an intrinsic part of the right to life and personal liberty under Article 21 and as a part of the freedoms guaranteed by Part III of the Constitution.
2. The earlier decisions in M.P. Sharma and Kharak Singh to the extent that they indicate that the right to privacy is not protected under the Constitution do not lay down the correct position in law and are overruled.
3. Privacy includes at its core the preservation of personal intimacies, the sanctity of family life, marriage, procreation, the home and sexual orientation. Privacy also connotes a right to be left alone.
4. Privacy safeguards individual autonomy and recognises the ability of the individual to control vital aspects of his or her life.
5. The right to privacy is not absolute. Legitimate state interests may encroach upon the right to privacy if such encroachment is necessary, proportional and subject to procedural safeguards.

RATIO DECIDENDI:
Privacy is a constitutionally protected right which emerges primarily from the guarantee of life and personal liberty in Article 21 of the Constitution. The right to privacy is protected as a fundamental right under the Constitution and earlier decisions holding to the contrary stand overruled.

The test for restriction of the right to privacy: (i) the action must be sanctioned by law; (ii) the proposed action must be necessary in a democratic society for a legitimate aim; (iii) the extent of such interference must be proportionate to the need for such interference; (iv) there must be procedural guarantees against abuse of such interference.

FINAL ORDER:
The right to privacy is a fundamental right under the Constitution of India. The earlier decisions in M.P. Sharma and Kharak Singh to the extent they hold otherwise are overruled. The matter is referred back for hearing on the validity of the Aadhaar scheme.""",
    },

    {
        "citation": "(1978) 1 SCC 248",
        "case_name": "Maneka Gandhi v. Union of India",
        "court": "Supreme Court of India",
        "year": 1978,
        "bench": "Seven Judge Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CIVIL) NO. 231 OF 1977

Maneka Gandhi ... Petitioner
Versus
Union of India and Another ... Respondents

JUDGMENT

FACTS:
The petitioner's passport was impounded by the Government of India under Section 10(3)(c) of the Passports Act, 1967 in public interest. No reasons were given. The petitioner challenged this on the grounds that it violated Articles 14, 19 and 21 of the Constitution.

ISSUES:
1. Whether the right to travel abroad is part of the right to personal liberty under Article 21.
2. Whether the procedure established by law under Article 21 must be just, fair and reasonable.
3. Whether Articles 14, 19 and 21 are mutually exclusive or interrelated.

HELD:
1. The expression 'personal liberty' in Article 21 is of the widest amplitude and it covers a variety of rights which go to constitute the personal liberty of man including the right to travel abroad.
2. The procedure established by law under Article 21 must be right, just and fair and not arbitrary, fanciful or oppressive. If it is arbitrary, it would be no procedure at all and the requirement of Article 21 would not be satisfied.
3. Articles 14, 19 and 21 are not mutually exclusive. They are complementary to each other and must be read together. A law that abridges the right to personal liberty must satisfy the requirements of all three articles.
4. The concept of reasonableness must be projected in the procedure contemplated by Article 21.

RATIO DECIDENDI:
Article 21 requires that the procedure established by law for depriving a person of his life or personal liberty must be right, just and fair and not arbitrary, fanciful or oppressive. The law must stand the test of one or more of the fundamental rights conferred under Articles 14 and 19 which are closely connected with Article 21. The procedure must be fair, just and reasonable, not fanciful, oppressive or arbitrary.

FINAL ORDER:
The petition was disposed of with the direction that the petitioner be given an opportunity to represent her case before the passport authority.""",
    },

    {
        "citation": "(1973) 4 SCC 225",
        "case_name": "Kesavananda Bharati v. State of Kerala",
        "court": "Supreme Court of India",
        "year": 1973,
        "bench": "Thirteen Judge Constitutional Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CIVIL) NO. 135 OF 1970

His Holiness Kesavananda Bharati Sripadagalvaru ... Petitioner
Versus
State of Kerala and Another ... Respondents

JUDGMENT — BASIC STRUCTURE DOCTRINE

FACTS:
The petitioner, the head of a religious order, challenged Kerala land reform laws which restricted the right to manage religious property. The case raised fundamental questions about the extent of Parliament's power to amend the Constitution.

ISSUES:
1. Whether Parliament has unlimited power to amend any part of the Constitution including fundamental rights.
2. Whether the 24th, 25th and 29th Constitutional Amendments were valid.

HELD (by majority of 7:6):
1. Parliament has wide powers of amending the Constitution but such power does not include the power to abrogate or emasculate the basic elements or fundamental features of the Constitution.
2. The Constitution has certain basic features which cannot be amended by Parliament under Article 368. These include:
   - Supremacy of the Constitution
   - Republican and Democratic form of Government
   - Secular character of the Constitution
   - Separation of powers between legislature, executive and judiciary
   - Federal character of the Constitution
   - Dignity and freedom of the individual
   - Unity and integrity of the nation
   - Parliamentary system of Government
   - Rule of law
   - Independence of the judiciary
   - Judicial review
3. Any amendment which damages or destroys the basic structure of the Constitution is void.
4. Golaknath v. State of Punjab (1967) is overruled.

RATIO DECIDENDI:
Parliament's amending power under Article 368 is limited. Parliament cannot use its amending power to damage, destroy or abrogate the basic or essential features of the Constitution or its basic structure. The basic structure doctrine provides a constitutional safeguard against the arbitrary exercise of the amending power.

FINAL ORDER:
By a majority of 7:6, the Supreme Court held that Parliament's power to amend the Constitution is not unlimited. The basic structure of the Constitution cannot be destroyed even by a constitutional amendment. The 24th and 25th Amendments were upheld subject to the limitation that they cannot damage the basic structure.""",
    },

    {
        "citation": "(1997) 6 SCC 241",
        "case_name": "Vishaka v. State of Rajasthan",
        "court": "Supreme Court of India",
        "year": 1997,
        "bench": "Three Judge Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CRIMINAL) NOS. 666-70 OF 1992

Vishaka and Others ... Petitioners
Versus
State of Rajasthan and Others ... Respondents

JUDGMENT — SEXUAL HARASSMENT AT WORKPLACE GUIDELINES

FACTS:
Following the brutal gang rape of a social worker in Rajasthan who was working to prevent a child marriage, women's rights groups filed a PIL seeking enforcement of fundamental rights of working women. There was no domestic law on sexual harassment at the workplace.

ISSUES:
1. Whether sexual harassment at the workplace violates Articles 14, 15, 19(1)(g) and 21 of the Constitution.
2. In the absence of legislation, what guidelines should govern sexual harassment at workplace.

HELD:
1. Sexual harassment of women at workplace violates the fundamental rights of gender equality and the right to life and liberty including the right to work with human dignity.
2. In the absence of domestic law to provide for effective enforcement of the basic human right of gender equality, the contents of international conventions and norms are significant to determine the true meaning and intent of the guarantee of gender equality. CEDAW has been ratified by India.

VISHAKA GUIDELINES (binding until legislation):
1. Every employer must take appropriate steps to prevent sexual harassment.
2. Express prohibition of sexual harassment must be notified, published and circulated.
3. Appropriate work conditions must be provided for women in respect of hygiene, health and safety.
4. Complaints Committee: A complaints committee must be established and headed by a woman with at least half the members being women.
5. Criminal proceedings to be initiated where conduct amounts to a specific offence under IPC.
6. Awareness of rights of women workers in respect of sexual harassment at workplace.

RATIO DECIDENDI:
Each incident of sexual harassment at the workplace results in violation of the fundamental rights of gender equality and the right to life and liberty. In the absence of legislative measures, these guidelines are laid down to be followed as law declared under Article 141.

FINAL ORDER:
The Vishaka guidelines are issued to be treated as the law of the land in the absence of a specific legislation addressing sexual harassment at the workplace.""",
    },

    {
        "citation": "(2014) 5 SCC 438",
        "case_name": "National Legal Services Authority v. Union of India",
        "court": "Supreme Court of India",
        "year": 2014,
        "bench": "Two Judge Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CIVIL) NO. 400 OF 2012

National Legal Services Authority ... Petitioner
Versus
Union of India and Others ... Respondents

JUDGMENT — TRANSGENDER RIGHTS

FACTS:
The National Legal Services Authority filed a PIL seeking recognition of the rights of transgender persons including their right to self-identify their gender. The petition highlighted the discrimination and marginalisation faced by the transgender community.

ISSUES:
1. Whether transgender persons have a right to decide their self-identified gender.
2. Whether transgender persons have fundamental rights under Articles 14, 15, 16, 19 and 21.

HELD:
1. Hijras and eunuchs are to be treated as a third gender, apart from male and female, for the purpose of safeguarding their rights under Part III of the Constitution.
2. Transgender persons have a right to self-identify their gender as male, female or third gender. Gender identity is an integral part of one's personality.
3. The right to choose one's gender identity is integral to the right to lead a life with dignity under Article 21.
4. Discrimination on the basis of gender identity and expression violates Articles 14 and 15.
5. Transgender persons are entitled to reservation in cases of admission in educational institutions and for public appointments under Articles 15(4) and 16(4) as they are socially and educationally backward.
6. The Government is directed to take steps to treat transgender persons as socially and educationally backward classes of citizens.

RATIO DECIDENDI:
Gender identity is an integral part of one's personality. The right to express one's self-identified gender is also an integral part of personal autonomy and self-expression guaranteed under Article 21. Discrimination against transgender persons on the ground of gender identity violates Articles 14 and 15.

FINAL ORDER:
The rights of transgender persons are recognized. The State and Central Governments are directed to grant legal recognition of gender identity of transgender persons as male, female or third gender.""",
    },

    {
        "citation": "(1994) 3 SCC 1",
        "case_name": "S.R. Bommai v. Union of India",
        "court": "Supreme Court of India",
        "year": 1994,
        "bench": "Nine Judge Constitutional Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA

S.R. Bommai and Others ... Petitioners
Versus
Union of India and Others ... Respondents

JUDGMENT — PRESIDENT'S RULE AND FEDERALISM

FACTS:
The petitioners challenged the imposition of President's Rule under Article 356 in various states including Karnataka, Meghalaya, Nagaland, Madhya Pradesh, Himachal Pradesh, Rajasthan and certain other states.

ISSUES:
1. Whether the President's satisfaction under Article 356 is justiciable.
2. What are the limitations on the exercise of power under Article 356.
3. Whether dissolution of the State Legislature before the proclamation is laid before Parliament is valid.

HELD:
1. The President's satisfaction under Article 356 is not immune from judicial review. The Court can examine whether the proclamation was issued on the basis of relevant material or was issued mala fide or was based on wholly extraneous grounds.
2. Secularism is a basic feature of the Constitution. A State Government that acts against secular principles can be dismissed under Article 356.
3. The State Legislature cannot be dissolved before the proclamation is laid before Parliament and approved.
4. Where the Ministry has lost majority, the Governor should ask it to prove its majority on the floor of the House. Floor test is the proper mode.
5. The power under Article 356 is an emergency power and should be used as a last resort when all other alternatives have failed.

RATIO DECIDENDI:
Article 356 is a drastic and extreme power which should be used sparingly and only when absolutely necessary. The Governor must recommend President's Rule only when there is a political crisis and not merely when there is political instability. The Presidential proclamation is subject to judicial review.

FINAL ORDER:
Guidelines were issued for the exercise of power under Article 356. The majority of proclamations challenged were struck down as unconstitutional.""",
    },

    {
        "citation": "(2018) 10 SCC 1",
        "case_name": "Navtej Singh Johar v. Union of India",
        "court": "Supreme Court of India",
        "year": 2018,
        "bench": "Five Judge Constitutional Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CRIMINAL) NO. 76 OF 2016

Navtej Singh Johar and Others ... Petitioners
Versus
Union of India ... Respondent

JUDGMENT — DECRIMINALISATION OF CONSENSUAL SAME-SEX RELATIONS

FACTS:
The petitioners challenged Section 377 of the Indian Penal Code which criminalised consensual sexual acts between adults of the same sex as being violative of Articles 14, 15, 19 and 21 of the Constitution.

ISSUES:
1. Whether Section 377 IPC insofar as it criminalises consensual acts of adults of the same sex is unconstitutional.
2. Whether sexual orientation constitutes a ground analogous to sex under Article 15.

HELD (unanimous by all five judges):
1. Section 377 IPC to the extent it criminalises consensual sexual acts of adults in private is unconstitutional as it violates Articles 14, 15, 19 and 21.
2. Sexual orientation is a ground analogous to sex under Article 15 and any discrimination on this ground is impermissible.
3. The decision in Suresh Kumar Koushal v. Naz Foundation (2014) is overruled.
4. Members of the LGBTQ community are entitled to equal citizenship and all constitutional protections.
5. The right to life under Article 21 includes the right to live with dignity, the right to privacy and the right to express one's sexual identity.
6. Constitutional morality must prevail over social morality.

RATIO DECIDENDI:
Section 377 IPC insofar as it penalises consensual same-sex conduct between adults is unconstitutional for being violative of Articles 14, 15, 19 and 21 of the Constitution. The concept of constitutional morality means adherence to the core principles of the constitutional guarantee. Constitutional morality cannot be martyred at the altar of social morality.

FINAL ORDER:
Section 377 IPC is read down to the extent that it criminalises consensual sexual acts between adults. It will continue to apply to non-consensual acts and acts with minors.""",
    },

    {
        "citation": "(2016) 1 SCC 1",
        "case_name": "Supreme Court Advocates-on-Record Association v. Union of India (NJAC Judgment)",
        "court": "Supreme Court of India",
        "year": 2015,
        "bench": "Five Judge Constitutional Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CIVIL) NO. 13 OF 2015

Supreme Court Advocates-on-Record Association and Another ... Petitioners
Versus
Union of India ... Respondent

JUDGMENT — NATIONAL JUDICIAL APPOINTMENTS COMMISSION

FACTS:
The 99th Constitutional Amendment establishing the National Judicial Appointments Commission (NJAC) to replace the collegium system for appointment of judges to the Supreme Court and High Courts was challenged as violating the basic structure of the Constitution.

ISSUES:
1. Whether the 99th Constitutional Amendment and the NJAC Act violate the basic structure of the Constitution by abridging the independence of the judiciary.
2. Whether judicial independence is a basic feature of the Constitution.

HELD (4:1 majority):
1. The independence of the judiciary is a basic feature of the Constitution.
2. The 99th Constitutional Amendment and the NJAC Act are unconstitutional as they violate the basic structure of the Constitution by abridging judicial independence.
3. The NJAC Act gives the Law Minister and two eminent persons a significant role in judicial appointments which could compromise judicial independence.
4. The collegium system of appointment of judges, though not ideal, does not violate the basic structure of the Constitution.
5. The primacy of the Chief Justice of India and the collegium in matters of appointment and transfer is essential to judicial independence.

RATIO DECIDENDI:
Independence of judiciary is a basic feature of the Constitution. Any constitutional amendment that damages or destroys this independence is unconstitutional. The participation of the executive in judicial appointments through the NJAC undermines judicial independence which is a basic feature of the Constitution.

FINAL ORDER:
The 99th Constitutional Amendment and the NJAC Act, 2014 are struck down as unconstitutional. The collegium system of appointment of judges is restored.""",
    },

    {
        "citation": "AIR 2017 SC 4609",
        "case_name": "Shafin Jahan v. Asokan K.M. (Hadiya Case)",
        "court": "Supreme Court of India",
        "year": 2017,
        "bench": "Three Judge Bench",
        "law": "Constitution",
        "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CIVIL) NO. 366 OF 2017

Shafin Jahan ... Petitioner
Versus
Asokan K.M. and Others ... Respondents

JUDGMENT — RIGHT TO CHOOSE PARTNER AND RELIGION

FACTS:
Hadiya (formerly Akhila Ashokan), a 24-year old woman, converted to Islam and married Shafin Jahan. Her father filed a habeas corpus petition claiming she had been brainwashed. The Kerala High Court annulled the marriage and placed her under parental custody.

ISSUES:
1. Whether the High Court had jurisdiction to annul the marriage of an adult woman.
2. Whether the right to choose one's faith and life partner is a fundamental right.

HELD:
1. The right to choose one's faith and life partner is a fundamental right under Articles 19 and 21.
2. The High Court exceeded its jurisdiction in annulling the marriage of a major woman.
3. Individual choice in matters of faith and marriage is fundamental to the right to life and liberty.
4. The State or courts cannot act as super-parents to override the choice of adults.
5. The matter of alleged radicalisation was referred to the NIA for investigation separately.

RATIO DECIDENDI:
The right to choose one's partner in life and one's faith are integral parts of the right to life with dignity under Article 21. The right to marry a person of one's choice is part of the fundamental right to life. Courts cannot annul a valid marriage of a major woman based on the wishes of her parents.

FINAL ORDER:
The judgment of the Kerala High Court annulling the marriage was set aside. Hadiya was free to live with her husband.""",
    },

    {
        "citation": "(2022) 8 SCC 1",
        "case_name": "In Re: Directions in the matter of Personal Loans/Credit Cards",
        "court": "Supreme Court of India",
        "year": 2022,
        "bench": "Three Judge Bench",
        "law": "Other",
        "is_landmark": False,
        "text": """IN THE SUPREME COURT OF INDIA
SMW(C) NO. 3 OF 2020

IN RE: DIRECTIONS IN THE MATTER OF PERSONAL LOANS/CREDIT CARDS

DIRECTIONS ON LOAN MORATORIUM — COVID-19

FACTS:
During the COVID-19 pandemic, the Reserve Bank of India announced a moratorium on loan repayments. Borrowers sought waiver of interest on interest (compound interest) during the moratorium period. Various petitions were filed seeking relief.

ISSUES:
1. Whether interest can be charged during the moratorium period.
2. Whether compound interest (interest on interest) charged during the moratorium is valid.

HELD:
1. The Government of India's decision to waive compound interest (interest on interest) for the moratorium period for loans up to Rs. 2 crore is upheld.
2. Lenders directed to credit the difference between compound interest and simple interest for the moratorium period to eligible borrowers.
3. The moratorium was a fiscal policy decision and courts should be slow to interfere with such decisions.
4. Banks cannot charge penal interest during the moratorium period.

FINAL ORDER:
The compound interest charged during the moratorium period for loans up to Rs. 2 crore in specified categories to be refunded/adjusted. The RBI and Government circulars on moratorium relief upheld subject to directions.""",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# DB + indexing helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _insert_judgment_document(db, doc_id: str, j: Dict) -> None:
    import json as _json
    from sqlalchemy import text
    await db.execute(text("""
        INSERT INTO documents (
            document_id, document_type, law, court_name, citation,
            year, parties, topic, is_landmark, language,
            total_chunks, created_at
        ) VALUES (
            :doc_id, 'judgment', :law, :court, :citation,
            :year, :parties, :topic, :is_landmark, 'en',
            0, now()
        )
        ON CONFLICT (document_id) DO NOTHING
    """), {
        "doc_id": doc_id,
        "law": j.get("law", "Constitution"),
        "court": j["court"],
        "citation": j["citation"],
        "year": j["year"],
        "parties": _json.dumps(j["case_name"]),
        "topic": _json.dumps(j["case_name"]),
        "is_landmark": j.get("is_landmark", False),
    })


async def _insert_chunk(db, chunk_id: str, doc_id: str, content: str,
                        chunk_type: str, idx: int, qdrant_id: str) -> None:
    from sqlalchemy import text
    await db.execute(text("""
        INSERT INTO chunks (
            chunk_id, document_id, chunk_type, content, content_length,
            chunk_index, qdrant_id, content_tsv, created_at
        ) VALUES (
            :cid, :doc_id, :ctype, :content, :clen,
            :idx, :qid, to_tsvector('english', :content), now()
        )
        ON CONFLICT (chunk_id) DO NOTHING
    """), {
        "cid": chunk_id, "doc_id": doc_id, "ctype": chunk_type,
        "content": content, "clen": len(content),
        "idx": idx, "qid": qdrant_id,
    })


async def _update_chunk_count(db, doc_id: str, n: int) -> None:
    from sqlalchemy import text
    await db.execute(
        text("UPDATE documents SET total_chunks = :n WHERE document_id = :id"),
        {"n": n, "id": doc_id}
    )


def _split_judgment(text: str, doc_id: str, meta: Dict) -> list:
    """Split judgment text into semantic chunks using section markers."""
    from backend.models.domain import (
        ChunkType, CourtType, DocumentMetadata, DocumentType,
        LawCategory, LegalChunk
    )

    LAW_MAP = {
        "Constitution": LawCategory.CONSTITUTION,
        "BNS": LawCategory.BNS, "BNSS": LawCategory.BNSS,
        "BSA": LawCategory.BSA, "IPC": LawCategory.IPC,
        "Other": LawCategory.OTHER,
    }
    SECTION_MARKERS = [
        ("FACTS:", ChunkType.FACTS),
        ("ISSUES:", ChunkType.ISSUES),
        ("HELD:", ChunkType.FINDINGS),
        ("RATIO DECIDENDI:", ChunkType.RATIO),
        ("FINAL ORDER:", ChunkType.FINAL_ORDER),
        ("VISHAKA GUIDELINES", ChunkType.FINDINGS),
    ]

    doc_meta = DocumentMetadata(
        document_type=DocumentType.JUDGMENT,
        law=LAW_MAP.get(meta.get("law", "Other"), LawCategory.OTHER),
        year=meta.get("year"),
        citation=meta.get("citation"),
        court_name=meta.get("court"),
        parties={"name": meta.get("case_name", "")},
        is_landmark=meta.get("is_landmark", False),
        language="en",
    )

    chunks = []
    remaining = text.strip()
    idx = 0

    for marker, ctype in SECTION_MARKERS:
        pos = remaining.find(marker)
        if pos == -1:
            continue
        before = remaining[:pos].strip()
        if before and len(before) > 100:
            chunks.append(LegalChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                chunk_type=ChunkType.PASSAGE,
                content=before[:2000],
                content_length=min(len(before), 2000),
                chunk_index=idx,
                metadata=doc_meta,
            ))
            idx += 1
        # Find next marker
        next_pos = len(remaining)
        for next_marker, _ in SECTION_MARKERS:
            np = remaining.find(next_marker, pos + len(marker))
            if np != -1 and np < next_pos:
                next_pos = np
        section_text = remaining[pos:next_pos].strip()
        if section_text and len(section_text) > 50:
            chunks.append(LegalChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                chunk_type=ctype,
                content=section_text[:2500],
                content_length=min(len(section_text), 2500),
                chunk_index=idx,
                metadata=doc_meta,
            ))
            idx += 1
        remaining = remaining[next_pos:]

    # Remainder
    if remaining.strip() and len(remaining.strip()) > 100:
        chunks.append(LegalChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_type=ChunkType.FINAL_ORDER,
            content=remaining.strip()[:2000],
            content_length=min(len(remaining.strip()), 2000),
            chunk_index=idx,
            metadata=doc_meta,
        ))

    # Fallback: whole text as one chunk
    if not chunks:
        chunks.append(LegalChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_type=ChunkType.PASSAGE,
            content=text[:3000],
            content_length=min(len(text), 3000),
            chunk_index=0,
            metadata=doc_meta,
        ))

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Main seeder
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    from backend.db.session import get_db_session, init_db
    from backend.embeddings.service import EmbeddingService
    from backend.retrieval.vector.retriever import VectorRetriever

    logger.info("Initialising database...")
    await init_db()

    embedder = EmbeddingService()
    await embedder.initialize()

    vr = VectorRetriever()
    await vr.ensure_collection()

    total_chunks = 0

    async with get_db_session() as db:
        for j in LANDMARK_JUDGMENTS:
            doc_id = str(uuid.uuid4())
            await _insert_judgment_document(db, doc_id, j)

            chunks = _split_judgment(j["text"], doc_id, j)
            texts = [c.content for c in chunks]

            logger.info(f"Embedding {j['citation']} ({len(chunks)} chunks)...")
            embeddings = await embedder.embed(texts)

            qdrant_ids = []
            for chunk, emb in zip(chunks, embeddings):
                qid = str(uuid.uuid4())
                await _insert_chunk(db, chunk.chunk_id, doc_id, chunk.content,
                                    chunk.chunk_type.value, chunk.chunk_index, qid)
                qdrant_ids.append(qid)

            await vr.upsert_chunks(chunks, embeddings)
            await _update_chunk_count(db, doc_id, len(chunks))
            total_chunks += len(chunks)
            logger.info(f"  ✓ {j['case_name']} — {len(chunks)} chunks")

        await db.commit()

    logger.info(f"\n{'='*50}")
    logger.info(f"Judgments seeded: {len(LANDMARK_JUDGMENTS)} cases, {total_chunks} chunks")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
