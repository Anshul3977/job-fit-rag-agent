from app.ingest.chunking import chunk_text, split_sections


def test_sections_and_offsets():
    text = """
Avery Chen
SUMMARY
Ships Python APIs.
EXPERIENCE
Built FastAPI services for billing.
PROJECTS
A retrieval-augmented FAQ bot.
SKILLS
Python, FastAPI, Docker
""".strip()
    sections = {section.name: section.text for section in split_sections(text)}
    assert "FastAPI" in sections["experience"]
    assert "FAQ" in sections["projects"]
    assert "Docker" in sections["skills"]
    drafts = chunk_text(text)
    assert drafts
    assert {draft.section for draft in drafts} >= {"experience", "projects", "skills"}
    for draft in drafts:
        assert draft.char_end > draft.char_start
        assert draft.text == text[draft.char_start : draft.char_end] or draft.text in text


def test_long_section_splits_with_overlap():
    body = " ".join(["FastAPI billing endpoint"] * 400)
    text = f"EXPERIENCE\n{body}\n"
    drafts = chunk_text(text, chunk_size=500, chunk_overlap=80)
    assert len(drafts) >= 2
    assert all(draft.section == "experience" for draft in drafts)
    assert drafts[1].text[:40] in drafts[0].text or drafts[0].text[-40:] in drafts[1].text
