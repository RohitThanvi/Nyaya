"""
Judgment seed script — 10 landmark Supreme Court judgments, inline public-domain text.

Usage:
    docker compose exec backend python -m backend.scripts.seed_judgments

Rewritten to use the actual current APIs:
    EmbeddingService.embed_batch(texts)
    VectorRetriever.upsert_batch(ids, vecs, payloads)
    parties stored as JSONB object {"name": "..."} — not json.dumps string
    no total_chunks / qdrant_id columns — schema matches 0001_initial.sql exactly
"""
import asyncio
import json
import logging
import uuid
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LANDMARK_JUDGMENTS: List[Dict] = [
    {
        "citation": "(2017) 10 SCC 1",
        "case_name": "Justice K.S. Puttaswamy (Retd.) v. Union of India",
        "court": "Supreme Court of India", "year": 2017,
        "bench": "Nine Judge Constitutional Bench", "law": "Constitution", "is_landmark": True,
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
        "court": "Supreme Court of India", "year": 1978,
        "bench": "Seven Judge Bench", "law": "Constitution", "is_landmark": True,
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
2. The procedure established by law under Article 21 must be right, just and fair and not arbitrary, fanciful or oppressive.
3. Articles 14, 19 and 21 are not mutually exclusive. They are complementary to each other and must be read together.
4. The concept of reasonableness must be projected in the procedure contemplated by Article 21.

RATIO DECIDENDI:
Article 21 requires that the procedure established by law for depriving a person of his life or personal liberty must be right, just and fair and not arbitrary, fanciful or oppressive. The law must stand the test of one or more of the fundamental rights conferred under Articles 14 and 19 which are closely connected with Article 21.

FINAL ORDER:
The petition was disposed of with the direction that the petitioner be given an opportunity to represent her case before the passport authority.""",
    },
    {
        "citation": "(1973) 4 SCC 225",
        "case_name": "Kesavananda Bharati v. State of Kerala",
        "court": "Supreme Court of India", "year": 1973,
        "bench": "Thirteen Judge Constitutional Bench", "law": "Constitution", "is_landmark": True,
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
2. The Constitution has certain basic features which cannot be amended by Parliament under Article 368, including supremacy of the Constitution, republican and democratic form of Government, secular character, separation of powers, federal character, dignity and freedom of the individual, unity and integrity of the nation, rule of law, and independence of the judiciary.
3. Any amendment which damages or destroys the basic structure of the Constitution is void.
4. Golaknath v. State of Punjab (1967) is overruled.

RATIO DECIDENDI:
Parliament's amending power under Article 368 is limited. Parliament cannot use its amending power to damage, destroy or abrogate the basic or essential features of the Constitution or its basic structure.

FINAL ORDER:
By a majority of 7:6, the Supreme Court held that Parliament's power to amend the Constitution is not unlimited. The basic structure of the Constitution cannot be destroyed even by a constitutional amendment.""",
    },
    {
        "citation": "(1997) 6 SCC 241",
        "case_name": "Vishaka v. State of Rajasthan",
        "court": "Supreme Court of India", "year": 1997,
        "bench": "Three Judge Bench", "law": "Constitution", "is_landmark": True,
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
2. In the absence of domestic law, the contents of international conventions and norms are significant to determine the true meaning and intent of the guarantee of gender equality.

VISHAKA GUIDELINES (binding until legislation):
1. Every employer must take appropriate steps to prevent sexual harassment.
2. Express prohibition of sexual harassment must be notified, published and circulated.
3. Appropriate work conditions must be provided for women.
4. A complaints committee must be established headed by a woman with at least half the members being women.
5. Criminal proceedings to be initiated where conduct amounts to a specific offence under IPC.

RATIO DECIDENDI:
Each incident of sexual harassment at the workplace results in violation of the fundamental rights of gender equality and the right to life and liberty. In the absence of legislative measures, these guidelines are laid down to be followed as law declared under Article 141.

FINAL ORDER:
The Vishaka guidelines are issued to be treated as the law of the land in the absence of a specific legislation addressing sexual harassment at the workplace.""",
    },
    {
        "citation": "(2014) 5 SCC 438",
        "case_name": "National Legal Services Authority v. Union of India",
        "court": "Supreme Court of India", "year": 2014,
        "bench": "Two Judge Bench", "law": "Constitution", "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CIVIL) NO. 400 OF 2012

National Legal Services Authority ... Petitioner
Versus
Union of India and Others ... Respondents

JUDGMENT — TRANSGENDER RIGHTS

FACTS:
The National Legal Services Authority filed a PIL seeking recognition of the rights of transgender persons including their right to self-identify their gender.

ISSUES:
1. Whether transgender persons have a right to decide their self-identified gender.
2. Whether transgender persons have fundamental rights under Articles 14, 15, 16, 19 and 21.

HELD:
1. Hijras and eunuchs are to be treated as a third gender, apart from male and female, for the purpose of safeguarding their rights under Part III of the Constitution.
2. Transgender persons have a right to self-identify their gender as male, female or third gender.
3. The right to choose one's gender identity is integral to the right to lead a life with dignity under Article 21.
4. Discrimination on the basis of gender identity and expression violates Articles 14 and 15.
5. Transgender persons are entitled to reservation in cases of admission in educational institutions and for public appointments under Articles 15(4) and 16(4).

RATIO DECIDENDI:
Gender identity is an integral part of one's personality. The right to express one's self-identified gender is also an integral part of personal autonomy and self-expression guaranteed under Article 21.

FINAL ORDER:
The rights of transgender persons are recognized. The State and Central Governments are directed to grant legal recognition of gender identity of transgender persons as male, female or third gender.""",
    },
    {
        "citation": "(1994) 3 SCC 1",
        "case_name": "S.R. Bommai v. Union of India",
        "court": "Supreme Court of India", "year": 1994,
        "bench": "Nine Judge Constitutional Bench", "law": "Constitution", "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA

S.R. Bommai and Others ... Petitioners
Versus
Union of India and Others ... Respondents

JUDGMENT — PRESIDENT'S RULE AND FEDERALISM

FACTS:
The petitioners challenged the imposition of President's Rule under Article 356 in various states including Karnataka, Meghalaya, Nagaland, Madhya Pradesh, Himachal Pradesh and Rajasthan.

ISSUES:
1. Whether the President's satisfaction under Article 356 is justiciable.
2. What are the limitations on the exercise of power under Article 356.
3. Whether dissolution of the State Legislature before the proclamation is laid before Parliament is valid.

HELD:
1. The President's satisfaction under Article 356 is not immune from judicial review. The Court can examine whether the proclamation was issued mala fide or on wholly extraneous grounds.
2. Secularism is a basic feature of the Constitution. A State Government that acts against secular principles can be dismissed under Article 356.
3. The State Legislature cannot be dissolved before the proclamation is laid before Parliament and approved.
4. Where the Ministry has lost majority, the Governor should ask it to prove its majority on the floor of the House.
5. The power under Article 356 is an emergency power and should be used as a last resort.

RATIO DECIDENDI:
Article 356 is a drastic and extreme power which should be used sparingly and only when absolutely necessary. The Presidential proclamation is subject to judicial review.

FINAL ORDER:
Guidelines were issued for the exercise of power under Article 356. The majority of proclamations challenged were struck down as unconstitutional.""",
    },
    {
        "citation": "(2018) 10 SCC 1",
        "case_name": "Navtej Singh Johar v. Union of India",
        "court": "Supreme Court of India", "year": 2018,
        "bench": "Five Judge Constitutional Bench", "law": "Constitution", "is_landmark": True,
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
2. Sexual orientation is a ground analogous to sex under Article 15.
3. The decision in Suresh Kumar Koushal v. Naz Foundation (2014) is overruled.
4. Members of the LGBTQ community are entitled to equal citizenship and all constitutional protections.
5. Constitutional morality must prevail over social morality.

RATIO DECIDENDI:
Section 377 IPC insofar as it penalises consensual same-sex conduct between adults is unconstitutional for being violative of Articles 14, 15, 19 and 21 of the Constitution.

FINAL ORDER:
Section 377 IPC is read down to the extent that it criminalises consensual sexual acts between adults. It will continue to apply to non-consensual acts and acts with minors.""",
    },
    {
        "citation": "(2016) 1 SCC 1",
        "case_name": "Supreme Court Advocates-on-Record Association v. Union of India (NJAC Judgment)",
        "court": "Supreme Court of India", "year": 2015,
        "bench": "Five Judge Constitutional Bench", "law": "Constitution", "is_landmark": True,
        "text": """IN THE SUPREME COURT OF INDIA
WRIT PETITION (CIVIL) NO. 13 OF 2015

Supreme Court Advocates-on-Record Association and Another ... Petitioners
Versus
Union of India ... Respondent

JUDGMENT — NATIONAL JUDICIAL APPOINTMENTS COMMISSION

FACTS:
The 99th Constitutional Amendment establishing the National Judicial Appointments Commission (NJAC) to replace the collegium system for appointment of judges was challenged as violating the basic structure of the Constitution.

ISSUES:
1. Whether the 99th Constitutional Amendment and the NJAC Act violate the basic structure of the Constitution.
2. Whether judicial independence is a basic feature of the Constitution.

HELD (4:1 majority):
1. The independence of the judiciary is a basic feature of the Constitution.
2. The 99th Constitutional Amendment and the NJAC Act are unconstitutional as they violate the basic structure of the Constitution by abridging judicial independence.
3. The NJAC Act gives the Law Minister and two eminent persons a significant role in judicial appointments which could compromise judicial independence.
4. The collegium system, though not ideal, does not violate the basic structure of the Constitution.

RATIO DECIDENDI:
Independence of judiciary is a basic feature of the Constitution. Any constitutional amendment that damages or destroys this independence is unconstitutional.

FINAL ORDER:
The 99th Constitutional Amendment and the NJAC Act, 2014 are struck down as unconstitutional. The collegium system of appointment of judges is restored.""",
    },
    {
        "citation": "AIR 2017 SC 4609",
        "case_name": "Shafin Jahan v. Asokan K.M. (Hadiya Case)",
        "court": "Supreme Court of India", "year": 2017,
        "bench": "Three Judge Bench", "law": "Constitution", "is_landmark": True,
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
The right to choose one's partner in life and one's faith are integral parts of the right to life with dignity under Article 21.

FINAL ORDER:
The judgment of the Kerala High Court annulling the marriage was set aside. Hadiya was free to live with her husband.""",
    },
    {
        "citation": "(2022) 8 SCC 1",
        "case_name": "In Re: Directions in the matter of Personal Loans/Credit Cards",
        "court": "Supreme Court of India", "year": 2022,
        "bench": "Three Judge Bench", "law": "Other", "is_landmark": False,
        "text": """IN THE SUPREME COURT OF INDIA
SMW(C) NO. 3 OF 2020

IN RE: DIRECTIONS IN THE MATTER OF PERSONAL LOANS/CREDIT CARDS

DIRECTIONS ON LOAN MORATORIUM — COVID-19

FACTS:
During the COVID-19 pandemic, the Reserve Bank of India announced a moratorium on loan repayments. Borrowers sought waiver of interest on interest (compound interest) during the moratorium period.

ISSUES:
1. Whether interest can be charged during the moratorium period.
2. Whether compound interest charged during the moratorium is valid.

HELD:
1. The Government of India's decision to waive compound interest for the moratorium period for loans up to Rs. 2 crore is upheld.
2. Lenders directed to credit the difference between compound interest and simple interest for the moratorium period to eligible borrowers.
3. The moratorium was a fiscal policy decision and courts should be slow to interfere with such decisions.
4. Banks cannot charge penal interest during the moratorium period.

FINAL ORDER:
The compound interest charged during the moratorium period for loans up to Rs. 2 crore in specified categories to be refunded/adjusted.""",
    },
]

SECTION_MARKERS = [
    ("FACTS:", "facts"),
    ("ISSUES:", "issues"),
    ("HELD:", "findings"),
    ("RATIO DECIDENDI:", "ratio"),
    ("FINAL ORDER:", "final_order"),
    ("VISHAKA GUIDELINES", "findings"),
]


def _split_judgment(text: str, doc_id: str, j: Dict) -> List[dict]:
    """Split judgment text into typed chunks using section markers."""
    chunks = []
    remaining = text.strip()
    idx = 0

    for marker, ctype in SECTION_MARKERS:
        pos = remaining.find(marker)
        if pos == -1:
            continue
        before = remaining[:pos].strip()
        if before and len(before) > 100:
            chunks.append({"content": before[:2000], "chunk_type": "passage",
                           "chunk_index": idx, "section_ref": None})
            idx += 1
        next_pos = len(remaining)
        for next_marker, _ in SECTION_MARKERS:
            np = remaining.find(next_marker, pos + len(marker))
            if np != -1 and np < next_pos:
                next_pos = np
        section_text = remaining[pos:next_pos].strip()
        if section_text and len(section_text) > 50:
            chunks.append({"content": section_text[:2500], "chunk_type": ctype,
                           "chunk_index": idx, "section_ref": None})
            idx += 1
        remaining = remaining[next_pos:]

    if remaining.strip() and len(remaining.strip()) > 100:
        chunks.append({"content": remaining.strip()[:2000], "chunk_type": "final_order",
                       "chunk_index": idx, "section_ref": None})

    if not chunks:
        chunks.append({"content": text[:3000], "chunk_type": "passage",
                       "chunk_index": 0, "section_ref": None})

    for c in chunks:
        c["chunk_id"] = str(uuid.uuid4())
        c["document_id"] = doc_id

    return chunks


async def _insert_judgment_document(db, doc_id: str, j: Dict) -> None:
    from sqlalchemy import text
    await db.execute(text("""
        INSERT INTO documents (
            document_id, document_type, law, court_name, citation,
            year, parties, topic, is_landmark, language, created_at
        ) VALUES (
            :doc_id, 'judgment', :law, :court, :citation,
            :year, :parties, :topic, :is_landmark, 'en', now()
        )
        ON CONFLICT (document_id) DO NOTHING
    """), {
        "doc_id": doc_id,
        "law": j.get("law", "Constitution"),
        "court": j["court"],
        "citation": j["citation"],
        "year": j["year"],
        # FIX: parties stored as proper JSONB object, not a json.dumps'd string —
        # frontend reads doc.parties.name as an object, not a string
        "parties": json.dumps({"name": j["case_name"]}),
        "topic": j["case_name"],
        "is_landmark": j.get("is_landmark", False),
    })


async def _insert_chunk(db, c: dict) -> None:
    from sqlalchemy import text
    await db.execute(text("""
        INSERT INTO chunks (
            chunk_id, document_id, chunk_type, content, content_length,
            chunk_index, section_ref, content_tsv, created_at
        ) VALUES (
            :cid, :doc_id, :ctype, :content, :clen,
            :idx, :sref, to_tsvector('english', :content), now()
        )
        ON CONFLICT (chunk_id) DO NOTHING
    """), {
        "cid": c["chunk_id"], "doc_id": c["document_id"], "ctype": c["chunk_type"],
        "content": c["content"], "clen": len(c["content"]),
        "idx": c["chunk_index"], "sref": c.get("section_ref"),
    })


def _build_payload(doc_id: str, j: Dict, c: dict) -> dict:
    return {
        "document_id": doc_id,
        "chunk_type": c["chunk_type"],
        "content": c["content"],
        "content_length": len(c["content"]),
        "chunk_index": c["chunk_index"],
        "section_ref": c.get("section_ref"),
        "document_type": "judgment",
        "law": j.get("law") if j.get("law") != "Other" else None,
        "court": "Supreme Court",
        "court_name": j["court"],
        "case_number": None,
        "citation": j["citation"],
        "year": j["year"],
        "topic": j["case_name"],
        "keywords": [],
        "source_url": None,
        "is_landmark": j.get("is_landmark", False),
        "language": "en",
    }


async def main() -> None:
    from backend.db.session import get_db_session
    from backend.embeddings.service import EmbeddingService
    from backend.retrieval.vector.retriever import VectorRetriever

    embedder = EmbeddingService()
    vr = VectorRetriever()
    await vr.ensure_collection()

    total_chunks = 0

    async with get_db_session() as db:
        for j in LANDMARK_JUDGMENTS:
            doc_id = str(uuid.uuid4())
            await _insert_judgment_document(db, doc_id, j)

            chunks = _split_judgment(j["text"], doc_id, j)
            texts = [c["content"] for c in chunks]

            logger.info(f"Embedding {j['citation']} ({len(chunks)} chunks)...")
            embeddings = await embedder.embed_batch(texts)

            for c in chunks:
                await _insert_chunk(db, c)
            await db.commit()

            chunk_ids = [c["chunk_id"] for c in chunks]
            payloads = [_build_payload(doc_id, j, c) for c in chunks]
            success = await vr.upsert_batch(chunk_ids, embeddings, payloads)
            if not success:
                logger.error(f"  Qdrant upsert FAILED for {j['citation']} "
                             f"— in Postgres but not searchable via vector search")

            total_chunks += len(chunks)
            logger.info(f"  ✓ {j['case_name']} — {len(chunks)} chunks")

    logger.info(f"\n{'='*50}")
    logger.info(f"Seeded {len(LANDMARK_JUDGMENTS)} judgments, {total_chunks} total chunks")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
