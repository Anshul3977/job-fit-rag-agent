"""Pull plain text out of PDF, DOCX, and text uploads."""

from __future__ import annotations

import io
from dataclasses import dataclass

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.errors import ExtractError

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class ExtractedDocument:
    text: str
    page_spans: list[tuple[int, int, int]] | None


def _docx_blocks(document: Document):
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def extract_docx(data: bytes) -> ExtractedDocument:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises several package-specific types
        raise ExtractError("Could not read this DOCX file.") from exc
    lines: list[str] = []
    for block in _docx_blocks(document):
        if isinstance(block, Paragraph):
            style = block.style.name if block.style is not None else ""
            raw = block.text.strip()
            if not raw:
                continue
            if style.startswith("Heading"):
                lines.append(raw.upper())
            else:
                lines.append(raw)
        else:
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    lines.append(" | ".join(dict.fromkeys(cells)))
    return ExtractedDocument("\n".join(lines), None)


def extract_pdf(data: bytes) -> ExtractedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ExtractError("Could not read this PDF file.") from exc
    if reader.is_encrypted:
        raise ExtractError("Encrypted PDFs are not supported.")
    pieces: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise ExtractError(f"Could not extract text from page {index}.") from exc
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(page_text)
        cursor += len(page_text)
        spans.append((start, cursor, index))
    return ExtractedDocument("".join(pieces), spans)


def extract_text_bytes(data: bytes) -> ExtractedDocument:
    if b"\x00" in data[:4096]:
        raise ExtractError("This file looks binary, not plain text.")
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return ExtractedDocument(data.decode(encoding), None)
        except UnicodeDecodeError:
            continue
    raise ExtractError("Could not decode this text file.")


def extract_upload(filename: str, data: bytes) -> ExtractedDocument:
    suffix = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == ".pdf":
        return extract_pdf(data)
    if suffix == ".docx":
        return extract_docx(data)
    if suffix in {".txt", ".md"}:
        return extract_text_bytes(data)
    raise ExtractError("Unsupported file type.")


def usable_text(text: str) -> str:
    return " ".join(text.split())
