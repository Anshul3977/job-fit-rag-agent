"""Section-aware resume and job chunking.

Short sections stay intact. Longer sections are split with overlap so a
requirement near a boundary still shares text with the next chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 220

# Longer aliases first so "work experience" wins over "experience".
_HEADER_SPECS: list[tuple[str, str]] = [
    (r"professional summary", "summary"),
    (r"professional experience", "experience"),
    (r"work experience", "experience"),
    (r"technical skills", "skills"),
    (r"minimum qualifications", "requirements"),
    (r"required qualifications", "requirements"),
    (r"preferred qualifications", "preferred"),
    (r"what you(?:'|’)ll (?:bring|need)", "requirements"),
    (r"what you(?:'|’)ll do", "responsibilities"),
    (r"about the role", "about"),
    (r"nice to have", "preferred"),
    (r"must[- ]haves?", "requirements"),
    (r"tech stack", "skills"),
    (r"our stack", "skills"),
    (r"qualifications", "requirements"),
    (r"requirements", "requirements"),
    (r"responsibilities", "responsibilities"),
    (r"certifications?", "certifications"),
    (r"publications", "publications"),
    (r"experience", "experience"),
    (r"projects?", "projects"),
    (r"education", "education"),
    (r"summary", "summary"),
    (r"objective", "summary"),
    (r"skills", "skills"),
    (r"about", "about"),
    (r"preferred", "preferred"),
    (r"bonus", "preferred"),
    (r"benefits", "benefits"),
    (r"awards", "awards"),
]

_HEADER_RES = [(re.compile(rf"^{pattern}\s*:?\s*$", re.I), name) for pattern, name in _HEADER_SPECS]

SECTION_WEIGHTS = {
    "requirements": 1.0,
    "skills": 0.9,
    "preferred": 0.85,
    "experience": 0.75,
    "responsibilities": 0.7,
    "projects": 0.7,
    "full": 0.55,
    "summary": 0.45,
    "about": 0.4,
    "preamble": 0.35,
    "education": 0.3,
    "benefits": 0.1,
}


@dataclass
class Section:
    name: str
    text: str
    start: int
    end: int


@dataclass
class ChunkDraft:
    section: str
    text: str
    char_start: int
    char_end: int
    page: int | None
    token_estimate: int


def match_header(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    for pattern, name in _HEADER_RES:
        if pattern.match(stripped):
            return name
    return None


def split_sections(text: str) -> list[Section]:
    """Split on known headings. The preamble is everything before the first one."""
    matches: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        name = match_header(line)
        if name:
            matches.append((offset, offset + len(line), name))
        offset += len(line)

    if not matches:
        body = text.strip()
        if not body:
            return []
        start = text.find(body)
        return [Section("preamble", body, start, start + len(body))]

    sections: list[Section] = []
    if matches[0][0] > 0:
        preamble = text[: matches[0][0]].strip()
        if preamble:
            start = text.find(preamble)
            sections.append(Section("preamble", preamble, start, start + len(preamble)))

    for index, (_start, header_end, name) in enumerate(matches):
        body_end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        body = text[header_end:body_end].strip()
        if not body:
            continue
        body_start = text.find(body, header_end)
        if body_start < 0:
            body_start = header_end
        sections.append(Section(name, body, body_start, body_start + len(body)))
    return sections


def _page_for(page_spans: list[tuple[int, int, int]] | None, char_index: int) -> int | None:
    if not page_spans:
        return None
    for start, end, page in page_spans:
        if start <= char_index < end:
            return page
    return page_spans[-1][2]


def chunk_text(
    text: str,
    page_spans: list[tuple[int, int, int]] | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    drafts: list[ChunkDraft] = []
    for section in split_sections(text):
        parts = splitter.split_text(section.text) if len(section.text) > chunk_size else [section.text]
        cursor = section.start
        for part in parts:
            part = part.strip()
            if not part:
                continue
            found = text.find(part, cursor)
            if found < 0:
                found = text.find(part)
            start = found if found >= 0 else cursor
            end = start + len(part)
            cursor = max(start + 1, end - chunk_overlap)
            drafts.append(
                ChunkDraft(
                    section=section.name,
                    text=part,
                    char_start=start,
                    char_end=end,
                    page=_page_for(page_spans, start),
                    token_estimate=max(1, round(len(part) / 4)),
                )
            )
    return drafts
