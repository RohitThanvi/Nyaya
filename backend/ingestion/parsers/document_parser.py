"""
Legal document parsers.

Strategy:
1. Try pdfplumber (best for digital PDFs — retains structure)
2. Fall back to PyMuPDF (faster, handles more formats)
3. If text extraction quality is poor → Tesseract OCR
4. HTML parsing for India Code web pages

Quality detection: if text has too many non-ASCII chars or short lines → OCR needed.
"""
import hashlib
import io
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _estimate_ocr_quality(text: str) -> float:
    """
    Estimate text extraction quality.
    Returns 0.0 (garbage) to 1.0 (clean text).
    """
    if not text or len(text) < 100:
        return 0.0
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0.0
    # Ratio of lines with at least 5 chars
    decent_lines = sum(1 for l in non_empty if len(l.strip()) >= 5)
    line_ratio = decent_lines / len(non_empty)
    # Ratio of ASCII-printable chars
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    ascii_ratio = printable / len(text)
    return (line_ratio * 0.5 + ascii_ratio * 0.5)


class PDFParser:
    """
    Multi-strategy PDF parser.
    Preserves page structure for accurate citation referencing.
    """

    def __init__(self, ocr_threshold: float = 0.6):
        self._ocr_threshold = ocr_threshold

    def parse(self, file_path: str) -> Tuple[str, Dict, int]:
        """
        Parse PDF and return (text, structure_hints, page_count).

        structure_hints: {page_num: {"headings": [...], "sections": [...]}}
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        # Try pdfplumber first
        text, structure, pages = self._try_pdfplumber(file_path)
        quality = _estimate_ocr_quality(text)

        if quality < self._ocr_threshold:
            logger.info(f"PDF quality {quality:.2f} below threshold, trying PyMuPDF: {path.name}")
            text2, structure2, pages2 = self._try_pymupdf(file_path)
            quality2 = _estimate_ocr_quality(text2)
            if quality2 > quality:
                text, structure, pages = text2, structure2, pages2
                quality = quality2

        if quality < self._ocr_threshold:
            logger.info(f"PDF quality {quality:.2f} still low, attempting OCR: {path.name}")
            text, structure, pages = self._try_tesseract(file_path, pages)

        logger.info(
            f"Parsed {path.name}: {pages} pages, quality={quality:.2f}, chars={len(text)}"
        )
        return text, structure, pages

    def _try_pdfplumber(self, file_path: str) -> Tuple[str, Dict, int]:
        try:
            import pdfplumber
            texts = []
            structure = {}
            with pdfplumber.open(file_path) as pdf:
                pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    texts.append(page_text)
                    # Extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        structure.setdefault(i, {})["tables"] = tables
            return "\n\n".join(texts), structure, pages
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
            return "", {}, 0

    def _try_pymupdf(self, file_path: str) -> Tuple[str, Dict, int]:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            texts = []
            structure = {}
            for i, page in enumerate(doc, 1):
                blocks = page.get_text("dict")["blocks"]
                page_text = page.get_text("text")
                texts.append(page_text)
                # Extract headings by font size
                headings = []
                for block in blocks:
                    if block.get("type") == 0:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("size", 0) > 12 and span.get("text", "").strip():
                                    headings.append(span["text"].strip())
                if headings:
                    structure.setdefault(i, {})["headings"] = headings
            doc.close()
            return "\n\n".join(texts), structure, len(doc)
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
            return "", {}, 0

    def _try_tesseract(self, file_path: str, page_count: int) -> Tuple[str, Dict, int]:
        try:
            import fitz
            from PIL import Image
            import pytesseract
            doc = fitz.open(file_path)
            texts = []
            for page in doc:
                # Render at 300 DPI for good OCR quality
                mat = fitz.Matrix(300 / 72, 300 / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(
                    img,
                    lang="eng",
                    config="--psm 6 --oem 3",
                )
                texts.append(page_text)
            doc.close()
            return "\n\n".join(texts), {}, len(doc)
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return "", {}, page_count


class HTMLParser:
    """
    HTML parser for India Code and judiciary websites.
    Strips navigation, footers, and extracts main legal content.
    """

    def parse(self, html_content: str, source_url: str = "") -> Tuple[str, Dict]:
        """Parse HTML and return (text, metadata)."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed")
            return html_content, {}

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "iframe", "noscript", "form"]):
            tag.decompose()

        # India Code specific selectors
        main_content = (
            soup.find("div", class_="act-content") or
            soup.find("div", class_="main-content") or
            soup.find("div", {"id": "content"}) or
            soup.find("article") or
            soup.find("main") or
            soup.find("body")
        )

        if not main_content:
            return soup.get_text(separator="\n", strip=True), {}

        text = main_content.get_text(separator="\n", strip=True)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Extract metadata from India Code patterns
        metadata = {}
        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Extract act/section numbers from India Code URL patterns
        if "indiacode.nic.in" in source_url:
            act_match = re.search(r"actid=(\d+)", source_url)
            if act_match:
                metadata["act_id"] = act_match.group(1)

        return text, metadata


class MetadataExtractor:
    """
    Extracts structured metadata from raw document text.
    Handles Supreme Court judgment headers, India Code headers.
    """

    # Judgment metadata patterns
    CITATION_PATTERNS = [
        r"AIR\s+\d{4}\s+SC\s+\d+",
        r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
        r"\d{4}\s+SCR\s+\d+",
        r"MANU/SC/\d+/\d{4}",
    ]

    CASE_NUMBER_PATTERNS = [
        r"(Writ\s+Petition\s+(?:Civil|Criminal)?\s+No\.?\s*\d+\s+of\s+\d{4})",
        r"(Criminal\s+Appeal\s+No\.?\s*\d+\s+of\s+\d{4})",
        r"(Special\s+Leave\s+Petition\s+No\.?\s*\d+\s+of\s+\d{4})",
        r"(Transfer\s+Petition\s+No\.?\s*\d+\s+of\s+\d{4})",
        r"(Civil\s+Appeal\s+No\.?\s*\d+\s+of\s+\d{4})",
    ]

    JUDGE_PATTERN = r"(?:Justice|J\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
    YEAR_PATTERN = r"\b(19[0-9]{2}|20[0-2][0-9])\b"

    def extract_judgment_metadata(self, text: str, filename: str = "") -> Dict:
        """Extract metadata from judgment text."""
        meta = {}
        first_2000 = text[:2000]  # Metadata usually in header

        # Citation
        for pat in self.CITATION_PATTERNS:
            m = re.search(pat, first_2000)
            if m:
                meta["citation"] = m.group(0)
                break

        # Case number
        for pat in self.CASE_NUMBER_PATTERNS:
            m = re.search(pat, first_2000, re.IGNORECASE)
            if m:
                meta["case_number"] = m.group(1)
                break

        # Year
        years = re.findall(self.YEAR_PATTERN, first_2000)
        if years:
            meta["year"] = int(years[0])

        # Judges
        judges = re.findall(self.JUDGE_PATTERN, first_2000)
        if judges:
            meta["bench"] = list(set(judges))

        # Petitioner vs Respondent
        vs_match = re.search(
            r"([A-Z][A-Za-z\s]+?)\s+(?:v\.|vs\.?|versus)\s+([A-Z][A-Za-z\s]+?)(?:\n|$)",
            first_2000
        )
        if vs_match:
            meta["parties"] = {
                "petitioner": vs_match.group(1).strip(),
                "respondent": vs_match.group(2).strip(),
            }

        return meta

    def extract_statute_metadata(self, text: str, filename: str = "") -> Dict:
        """Extract metadata from statute/act text."""
        meta = {}
        first_500 = text[:500]

        # Detect law type
        law_patterns = {
            "BNS": r"Bharatiya\s+Nyaya\s+Sanhita",
            "BNSS": r"Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita",
            "BSA": r"Bharatiya\s+Sakshya\s+Adhiniyam",
        }
        for law, pattern in law_patterns.items():
            if re.search(pattern, text[:1000], re.IGNORECASE):
                meta["law"] = law
                break

        # Year of enactment
        year_match = re.search(r"Act(?:,\s+|\s+)(\d{4})", first_500)
        if year_match:
            meta["year"] = int(year_match.group(1))

        return meta
