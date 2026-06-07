"""
Seed script — NyayaAI

Downloads real Indian law datasets from HuggingFace (Apache-2.0 / public domain)
and seeds PostgreSQL + Qdrant.

Data sources:
  - nandhakumarg/IPC_and_BNS_transformation  — 563 BNS section texts
  - viber1/indian-law-dataset                — 24k Indian legal Q&A pairs
  - Inline BNSS / BSA statutory text (public domain)

Usage:
    docker exec nyaya-backend python -m backend.scripts.seed_data
    docker exec nyaya-backend python -m backend.scripts.seed_data --clear
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Inline BNSS statutory sections  (public domain)
# ──────────────────────────────────────────────────────────────────────────────
BNSS_SECTIONS: List[Tuple[str, str, str]] = [
    # (section_ref, title, text)
    ("1", "Short title, extent and commencement",
     "This Act may be called the Bharatiya Nagarik Suraksha Sanhita, 2023. It extends to the "
     "whole of India. It shall come into force on such date as the Central Government may, by "
     "notification in the Official Gazette, appoint."),

    ("173", "Information in cognizable cases — FIR",
     "Every information relating to the commission of a cognizable offence, irrespective of "
     "the area where the offence is committed, may be given orally or by electronic communication "
     "to an officer in charge of a police station, and if given orally shall be reduced to writing "
     "by him or under his direction, and be read over to the informant. Every such information, "
     "whether given in writing or reduced to writing, shall be signed by the person giving it, "
     "and the substance thereof shall be entered in a book to be kept by such officer in such "
     "form as the State Government may prescribe. A copy of the information as recorded shall be "
     "given forthwith, free of cost, to the informant."),

    ("187", "Arrest by police officer without warrant",
     "A police officer may, without an order from a Magistrate and without a warrant, arrest any "
     "person who has been concerned in any cognizable offence, or against whom a reasonable "
     "complaint has been made, or credible information has been received, or a reasonable "
     "suspicion exists, of his having been so concerned. A police officer shall not arrest a "
     "person except in accordance with the procedure specified in this Sanhita. Where a police "
     "officer makes an arrest without a warrant, he shall inform the person arrested of the "
     "grounds of arrest and of the right to bail."),

    ("193", "Rights of arrested person",
     "Every person who has been arrested and is being held in custody shall have the right to "
     "be informed, as soon as may be, of the grounds for such arrest or detention. Subject to "
     "any law for the time being in force, every person who is arrested shall have the right to "
     "consult and to be defended by a legal practitioner of his choice. Where any person is "
     "arrested, he shall be entitled to have one friend, relative or other person known to him "
     "informed as soon as practicable of his arrest and the place where he is being detained."),

    ("290", "Plea bargaining — application by accused",
     "A person accused of an offence may file an application for plea bargaining in the Court "
     "in which such offence is pending for trial. The application shall be filed within thirty "
     "days from the date of framing of charges. No application shall be filed if the offence "
     "affects the socio-economic conditions of the country, or is committed against a woman or "
     "a child below the age of fourteen years."),

    ("480", "Bail in bailable offence",
     "When any person other than a person accused of a non-bailable offence is arrested or "
     "detained without warrant by an officer in charge of a police station, or appears or is "
     "brought before a Court, and is prepared at any time while in the custody of such officer "
     "or at any stage of the proceeding before such Court to give bail, such person shall be "
     "released on bail."),

    ("481", "Bail in non-bailable offence",
     "When any person accused of, or suspected of, the commission of any non-bailable offence "
     "is arrested or detained without warrant by an officer in charge of a police station or "
     "appears or is brought before a Court other than the High Court or Court of Session, he "
     "may be released on bail, but he shall not be so released if there appear reasonable "
     "grounds for believing that he has been guilty of an offence punishable with death or "
     "imprisonment for life. A person who is under the age of sixteen years or is a woman or "
     "is sick or infirm, may be released on bail if the Court so directs."),

    ("482", "Anticipatory bail",
     "Where any person has reason to believe that he may be arrested on an accusation of "
     "having committed a non-bailable offence, he may apply to the High Court or the Court "
     "of Session for a direction under this section; and that Court may, if it thinks fit, "
     "direct that in the event of such arrest, he shall be released on bail. When the High "
     "Court or the Court of Session makes a direction under sub-section (1), it may include "
     "such conditions in such directions in the light of the facts of the particular case as "
     "it may think fit, including the condition that the person shall make himself available "
     "for interrogation by a police officer as and when required."),

    ("528", "Inherent powers of High Court — quashing proceedings",
     "Nothing in this Sanhita shall be deemed to limit or affect the inherent powers of the "
     "High Court to make such orders as may be necessary to give effect to any order under "
     "this Sanhita, or to prevent abuse of the process of any Court or otherwise to secure "
     "the ends of justice. The High Court may exercise its power under this section to quash "
     "an FIR, charge-sheet, or any criminal proceeding if it is satisfied that allowing the "
     "proceeding to continue would be an abuse of the process of the court or that the ends "
     "of justice require that the proceedings be quashed."),

    ("43", "Use of handcuffs",
     "The person arrested shall not be subjected to more restraint than is necessary to "
     "prevent his escape. Where a police officer has reason to believe that the arrested "
     "person may cause harm to himself or others, he may use handcuffs in accordance with "
     "the guidelines issued by the State Government. Handcuffs shall not be used on a "
     "person accused of a bailable offence except where the officer has reasons to believe "
     "that the person is likely to abscond or resist arrest."),

    ("105", "Trial of offences under Bharatiya Nyaya Sanhita",
     "Every offence under the Bharatiya Nyaya Sanhita, 2023 shall be tried in accordance "
     "with the provisions of this Sanhita. Where an offence is committed outside India by a "
     "citizen of India, proceedings may be taken against such person as if the offence had "
     "been committed at any place within India at which such person may be found."),

    ("258", "Statement of accused",
     "The Magistrate before taking cognizance of an offence may, if he thinks fit, examine "
     "upon oath any person present in court. Before examining the accused, the Magistrate "
     "shall inform the accused of the particulars of the offence alleged against him. The "
     "accused shall not be compelled to give evidence against himself."),
]

# ──────────────────────────────────────────────────────────────────────────────
# Inline BSA statutory sections  (public domain)
# ──────────────────────────────────────────────────────────────────────────────
BSA_SECTIONS: List[Tuple[str, str, str]] = [
    ("1", "Short title, commencement and application",
     "This Act may be called the Bharatiya Sakshya Adhiniyam, 2023. It shall come into force "
     "on such date as the Central Government may, by notification in the Official Gazette, "
     "appoint. It applies to all judicial proceedings in or before any Court including Courts "
     "Martial, but not to affidavits presented to any Court or officer, nor to proceedings "
     "before an arbitrator."),

    ("2", "Definitions — fact, relevant, document, evidence, electronic record",
     "In this Adhiniyam: 'fact' means any thing, state of things, or relation of things "
     "capable of being perceived by the senses, or any mental condition of which any person "
     "is conscious; 'relevant' — one fact is said to be relevant to another when the one is "
     "connected with the other in any of the ways referred to in the provisions of this "
     "Adhiniyam relating to the relevancy of facts; 'document' means any matter expressed "
     "or described upon any substance by means of letters, figures or marks intended to be "
     "used for recording that matter; 'electronic record' means data, record or data "
     "generated, image or sound stored, received or sent in an electronic form or micro film "
     "or computer generated micro fiche; 'evidence' means all statements which the Court "
     "permits or requires to be made before it by witnesses, and all documents including "
     "electronic records produced for inspection."),

    ("57", "Electronic and digital records — primary evidence",
     "Electronic or digital records produced from proper custody shall be primary evidence. "
     "Electronic record stored in any device, including a computer or a server located "
     "anywhere, shall be deemed to be a document and shall be admissible in any proceeding "
     "without further proof or production of the original as evidence of any contents of the "
     "original or of any fact stated therein, if the electronic record is certified by a "
     "person occupying a responsible official position in relation to the operation of the "
     "relevant device or management of the relevant activities."),

    ("61", "Admissibility of electronic records — conditions",
     "All other conditions being satisfied, information contained in an electronic record "
     "which is printed on paper, stored, recorded or copied in optical or magnetic media "
     "produced by a computer shall be deemed to also be a document if the conditions "
     "mentioned in this section are satisfied. The computer output containing the information "
     "was produced during the period over which the computer was used regularly to store or "
     "process information for the purposes of any activities regularly carried on over that "
     "period by the person having lawful control over the use of the computer."),

    ("63", "Secondary evidence",
     "Secondary evidence means and includes certified copies given under the provisions "
     "hereinafter contained; copies made from the original by mechanical processes which in "
     "themselves ensure the accuracy of the copy, and copies compared with such copies; "
     "copies made from or compared with the original; counterparts of documents as against "
     "the parties who did not execute them; oral accounts of the contents of a document "
     "given by some person who has himself seen it. A document may be proved by primary or "
     "secondary evidence."),

    ("106", "Burden of proof",
     "The burden of proof in a suit or proceeding lies on that person who would fail if no "
     "evidence at all were given on either side. When a person is accused of any offence, "
     "the burden of proving the existence of circumstances bringing the case within any of "
     "the General Exceptions in the Bharatiya Nyaya Sanhita, 2023, or within any special "
     "exception or proviso contained in any other part of the said Sanhita, or in any law "
     "defining the offence, is upon him, and the Court shall presume the absence of such "
     "circumstances."),

    ("111", "Burden of proof in cases of dowry death",
     "When the question is whether a person has committed the dowry death of a woman, and "
     "it is shown that soon before her death such woman had been subjected by such person to "
     "cruelty or harassment for, or in connection with, any demand for dowry, the Court "
     "shall presume that such person had caused the dowry death. In this section 'dowry "
     "death' shall have the same meaning as in section 80 of the Bharatiya Nyaya Sanhita, "
     "2023."),

    ("113", "Presumption as to absence of consent in rape cases",
     "When the prosecutrix states in her evidence before the Court that she did not consent "
     "to the act of sexual intercourse and the question is whether she consented to it or "
     "not, and the act of sexual intercourse by the accused is proved, the Court shall "
     "presume that she did not consent."),

    ("123", "Examination of witnesses",
     "Witnesses shall be first examined-in-chief, then, if the adverse party so desires, "
     "cross-examined, and then, if the party calling them so desires, re-examined. The "
     "examination and cross-examination must relate to relevant facts but the "
     "cross-examination need not be confined to the facts to which the witness testified "
     "on his examination-in-chief."),

    ("151", "Confession to police officer not provable",
     "No confession made to a police officer shall be proved as against a person accused of "
     "any offence. A confession made by an accused person is admissible in evidence if it "
     "is made voluntarily and before a Judicial Magistrate. The Magistrate shall, before "
     "recording the confession, explain to the person making it that he is not bound to "
     "make a confession and that if he does so it may be used as evidence against him."),

    ("23", "Admissions",
     "An admission is a statement, oral or documentary or contained in electronic form, "
     "which suggests any inference as to any fact in issue or relevant fact, and which is "
     "made by any of the persons, and under the circumstances, hereinafter mentioned. "
     "Admissions are relevant and may be proved as against the person who makes them or "
     "his representative in interest."),

    ("24", "Admissions by party to proceeding or his agent",
     "Statements made by a party to the proceeding, or by an agent to any party whom the "
     "Court regards, under the circumstances of the case, as expressly or impliedly "
     "authorised by him to make them, are admissions. Statements made by parties to suits "
     "suing or sued in a representative character are not admissions, unless the party "
     "making the statement made them while holding that character."),
]


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace download helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_hf_parquet(url: str) -> List[dict]:
    """Download a HuggingFace parquet file and return rows as list of dicts."""
    try:
        import io, urllib.request
        import pyarrow.parquet as pq
        logger.info(f"Downloading: {url}")
        with urllib.request.urlopen(url, timeout=90) as resp:
            data = resp.read()
        table = pq.read_table(io.BytesIO(data))
        return table.to_pylist()
    except ImportError:
        logger.warning("pyarrow not installed — skipping. Install with: pip install pyarrow")
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Domain model builder
# ──────────────────────────────────────────────────────────────────────────────

def _make_chunk(doc_id: str, content: str, section_ref: str,
                chunk_index: int, law_str: str, topic: str):
    """Create a LegalChunk domain object."""
    from backend.models.domain import (
        ChunkType, DocumentMetadata, DocumentType, LawCategory, LegalChunk
    )
    LAW_MAP = {
        "BNS": LawCategory.BNS, "BNSS": LawCategory.BNSS,
        "BSA": LawCategory.BSA, "IPC": LawCategory.IPC,
        "Other": LawCategory.OTHER,
    }
    meta = DocumentMetadata(
        document_type=DocumentType.STATUTE,
        law=LAW_MAP.get(law_str, LawCategory.OTHER),
        year=2023,
        section=section_ref,
        topic=topic,
        language="en",
    )
    return LegalChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=doc_id,
        chunk_type=ChunkType.SECTION,
        content=content,
        content_length=len(content),
        chunk_index=chunk_index,
        section_ref=section_ref,
        metadata=meta,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _insert_document(db, doc_id: str, law: str, topic: str) -> None:
    from sqlalchemy import text
    await db.execute(text("""
        INSERT INTO documents
            (document_id, document_type, law, topic, year, language, total_chunks, created_at)
        VALUES
            (:doc_id, 'statute', :law, :topic, 2023, 'en', 0, now())
        ON CONFLICT (document_id) DO NOTHING
    """), {"doc_id": doc_id, "law": law, "topic": topic})


async def _insert_chunk(db, chunk, qdrant_id: str) -> None:
    from sqlalchemy import text
    await db.execute(text("""
        INSERT INTO chunks
            (chunk_id, document_id, chunk_type, content, content_length,
             chunk_index, section_ref, qdrant_id, content_tsv, created_at)
        VALUES
            (:cid, :doc_id, :ctype, :content, :clen,
             :cidx, :sref, :qid,
             to_tsvector('english', :content), now())
        ON CONFLICT (chunk_id) DO NOTHING
    """), {
        "cid": chunk.chunk_id,
        "doc_id": chunk.document_id,
        "ctype": chunk.chunk_type.value,
        "content": chunk.content,
        "clen": chunk.content_length,
        "cidx": chunk.chunk_index,
        "sref": chunk.section_ref,
        "qid": qdrant_id,
    })


async def _update_chunk_count(db, doc_id: str, count: int) -> None:
    from sqlalchemy import text
    await db.execute(
        text("UPDATE documents SET total_chunks = :n WHERE document_id = :id"),
        {"n": count, "id": doc_id},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Embed + index helper
# ──────────────────────────────────────────────────────────────────────────────

async def _embed_and_index(db, chunks: list, embedder, vector_retriever) -> int:
    """Embed chunks and upsert to both Postgres and Qdrant."""
    if not chunks:
        return 0

    texts = [c.content for c in chunks]
    logger.info(f"  Embedding {len(texts)} chunks...")
    embeddings = await embedder.embed(texts)

    qdrant_ids = []
    for chunk in chunks:
        qid = str(uuid.uuid4())
        await _insert_chunk(db, chunk, qid)
        qdrant_ids.append(qid)

    # Upsert to Qdrant
    await vector_retriever.upsert_chunks(chunks, embeddings)
    return len(chunks)


# ──────────────────────────────────────────────────────────────────────────────
# Seeders
# ──────────────────────────────────────────────────────────────────────────────

async def seed_inline_law(db, law: str, sections: List[Tuple], embedder, vr) -> int:
    """Seed inline BNSS or BSA sections."""
    doc_id = str(uuid.uuid4())
    topic = {
        "BNSS": "Bharatiya Nagarik Suraksha Sanhita 2023",
        "BSA": "Bharatiya Sakshya Adhiniyam 2023",
    }.get(law, law)

    await _insert_document(db, doc_id, law, topic)

    chunks = []
    for idx, (section_ref, title, text_body) in enumerate(sections):
        content = f"{law} Section {section_ref} — {title}\n\n{text_body}"
        chunks.append(_make_chunk(doc_id, content, section_ref, idx, law, title))

    n = await _embed_and_index(db, chunks, embedder, vr)
    await _update_chunk_count(db, doc_id, n)
    logger.info(f"✓ {law}: {n} sections seeded")
    return n


async def seed_bns_from_hf(db, embedder, vr) -> int:
    """Seed BNS sections from nandhakumarg/IPC_and_BNS_transformation on HuggingFace."""
    URL = (
        "https://huggingface.co/datasets/nandhakumarg/IPC_and_BNS_transformation"
        "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )
    rows = _fetch_hf_parquet(URL)
    if not rows:
        logger.warning("BNS HF dataset unavailable — skipping")
        return 0

    doc_id = str(uuid.uuid4())
    await _insert_document(db, doc_id, "BNS",
                           "Bharatiya Nyaya Sanhita 2023 — Full Sections")

    chunks = []
    idx = 0
    for row in rows:
        response = row.get("response", "")
        if isinstance(response, str):
            try:
                d = eval(response, {"__builtins__": {}})
            except Exception:
                continue
        else:
            d = response if isinstance(response, dict) else {}

        sec = str(d.get("BNS Section", "")).strip()
        heading = str(d.get("BNS Heading", "")).strip()
        body = str(d.get("BNS description", "")).strip()

        if not body or sec in ("", "N/A", "Repealed"):
            continue

        content = f"BNS Section {sec} — {heading}\n\n{body}"[:4000]
        chunks.append(_make_chunk(doc_id, content, sec, idx, "BNS", heading))
        idx += 1

    if not chunks:
        logger.warning("No usable BNS rows from HF dataset")
        return 0

    n = await _embed_and_index(db, chunks, embedder, vr)
    await _update_chunk_count(db, doc_id, n)
    logger.info(f"✓ BNS (HuggingFace): {n} sections seeded")
    return n


async def seed_qa_from_hf(db, embedder, vr) -> int:
    """Seed viber1/indian-law-dataset Q&A pairs (Apache-2.0)."""
    URL = (
        "https://huggingface.co/datasets/viber1/indian-law-dataset"
        "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )
    rows = _fetch_hf_parquet(URL)
    if not rows:
        logger.warning("viber1 Q&A dataset unavailable — skipping")
        return 0

    doc_id = str(uuid.uuid4())
    await _insert_document(db, doc_id, "Other",
                           "Indian Legal Q&A Corpus (viber1/indian-law-dataset)")

    seen: set = set()
    chunks = []
    for row in rows:
        q = str(row.get("Instruction", "")).strip()
        a = str(row.get("Response", "")).strip()
        if not q or not a or q in seen or len(a) < 100:
            continue
        seen.add(q)
        content = f"Q: {q}\n\nA: {a}"[:3000]
        chunks.append(_make_chunk(doc_id, content, "", len(chunks),
                                  "Other", q[:120]))
        if len(chunks) >= 5000:
            break

    logger.info(f"Seeding {len(chunks)} Q&A pairs...")
    n = await _embed_and_index(db, chunks, embedder, vr)
    await _update_chunk_count(db, doc_id, n)
    logger.info(f"✓ Indian Law Q&A: {n} chunks seeded")
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Clear
# ──────────────────────────────────────────────────────────────────────────────

async def clear_seed_data(db) -> None:
    from sqlalchemy import text
    from backend.retrieval.vector.retriever import VectorRetriever, get_qdrant_client
    from backend.config.settings import get_settings

    logger.info("Clearing seeded data...")
    result = await db.execute(text(
        "SELECT document_id FROM documents WHERE document_type = 'statute' "
        "OR (topic LIKE '%Q&A%' OR topic LIKE '%indian-law%')"
    ))
    doc_ids = [str(r[0]) for r in result.fetchall()]
    if doc_ids:
        await db.execute(text(
            "DELETE FROM documents WHERE document_id = ANY(:ids)"
        ), {"ids": doc_ids})
        await db.commit()
        logger.info(f"Deleted {len(doc_ids)} documents from PostgreSQL")

    settings = get_settings()
    client = get_qdrant_client()
    try:
        await client.delete_collection(settings.qdrant.collection_name)
        logger.info("Deleted Qdrant collection")
    except Exception as e:
        logger.warning(f"Qdrant clear: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def main(clear: bool = False) -> None:
    from backend.db.session import get_db_session, init_db
    from backend.embeddings.service import EmbeddingService
    from backend.retrieval.vector.retriever import VectorRetriever

    logger.info("Initialising database schema...")
    await init_db()

    embedder = EmbeddingService()
    await embedder.initialize()

    vr = VectorRetriever()
    await vr.ensure_collection()

    async with get_db_session() as db:
        if clear:
            await clear_seed_data(db)

        total = 0

        # 1. BNS from HuggingFace (563 sections, full statutory text)
        total += await seed_bns_from_hf(db, embedder, vr)

        # 2. BNSS — inline key sections
        total += await seed_inline_law(db, "BNSS", BNSS_SECTIONS, embedder, vr)

        # 3. BSA — inline key sections
        total += await seed_inline_law(db, "BSA", BSA_SECTIONS, embedder, vr)

        # 4. Indian Law Q&A corpus (up to 5000 pairs)
        total += await seed_qa_from_hf(db, embedder, vr)

        await db.commit()

    logger.info(f"\n{'='*50}")
    logger.info(f"Seed complete — {total} total chunks indexed")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NyayaAI seed script")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing statute data before seeding")
    args = parser.parse_args()
    asyncio.run(main(clear=args.clear))
