"""Skill aliases used by the deterministic judge.

The table is intentionally explicit. A hit means the resume text contains the
tool, not a guessed synonym like "API framework" for FastAPI.
"""

from __future__ import annotations

import re

# (canonical name, aliases). Aliases are matched with a word boundary when they
# are a single token, and as a casefold substring when they contain punctuation.
SKILLS: list[tuple[str, list[str]]] = [
    ("Python", ["python"]),
    ("Java", ["java"]),
    ("Go", ["golang", "go"]),
    ("TypeScript", ["typescript"]),
    ("JavaScript", ["javascript", "js"]),
    ("SQL", ["sql", "postgresql", "postgres", "mysql", "sqlite"]),
    ("PostgreSQL", ["postgresql", "postgres", "psql"]),
    ("FastAPI", ["fastapi", "fast api"]),
    ("Flask", ["flask"]),
    ("Django", ["django"]),
    ("Spring", ["spring boot", "springframework", "spring mvc"]),
    ("React", ["react.js", "reactjs", "react"]),
    ("Next.js", ["next.js", "nextjs", "next js"]),
    ("CSS", ["css"]),
    ("HTML", ["html"]),
    ("Docker", ["docker-compose", "docker compose", "dockerfile", "docker"]),
    ("Kubernetes", ["kubernetes", "k8s"]),
    ("Terraform", ["terraform"]),
    ("AWS", ["aws", "amazon web services"]),
    ("GCP", ["gcp", "google cloud"]),
    ("Azure", ["azure"]),
    ("pytest", ["pytest"]),
    ("CI/CD", ["ci/cd", "continuous integration", "github actions", "gitlab ci"]),
    ("git", ["git"]),
    ("Linux", ["linux", "ubuntu"]),
    ("gRPC", ["grpc"]),
    ("REST", ["restful", "rest"]),
    ("RAG", ["retrieval-augmented", "retrieval augmented", "rag"]),
    ("embeddings", ["sentence-transformers", "minilm", "embeddings", "embedding"]),
    ("evaluation", ["evaluation", "eval set", "human eval"]),
    ("PyTorch", ["pytorch"]),
    ("TensorFlow", ["tensorflow"]),
    ("Hugging Face", ["hugging face", "huggingface"]),
    ("LangGraph", ["langgraph"]),
    ("LangChain", ["langchain"]),
    ("pandas", ["pandas"]),
    ("Pydantic", ["pydantic"]),
    ("SQLAlchemy", ["sqlalchemy"]),
    ("accessibility", ["accessibility", "a11y"]),
    ("Redis", ["redis"]),
    ("Kafka", ["kafka"]),
    ("Spark", ["spark"]),
]

ADJACENT: dict[str, list[str]] = {
    "FastAPI": ["Flask", "Django"],
    "Flask": ["FastAPI", "Django"],
    "Django": ["FastAPI", "Flask"],
    "React": ["Next.js"],
    "Next.js": ["React"],
    "Kubernetes": ["Docker"],
    "PostgreSQL": ["SQL"],
    "PyTorch": ["TensorFlow"],
    "TensorFlow": ["PyTorch"],
    "LangGraph": ["LangChain"],
    "Spring": ["Java"],
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "with",
    "your",
    "you",
    "our",
    "we",
    "that",
    "this",
    "from",
    "into",
    "using",
    "use",
    "used",
    "able",
    "ability",
    "experience",
    "strong",
    "plus",
    "years",
    "year",
    "including",
    "across",
    "such",
    "as",
    "be",
    "is",
    "are",
    "will",
    "can",
    "have",
    "has",
    "who",
    "their",
    "about",
    "role",
    "team",
    "work",
    "working",
    "build",
    "building",
    "built",
    "other",
    "etc",
    "not",
    "but",
    "at",
    "by",
    "it",
    "its",
}


def term_in_text(alias: str, text: str) -> bool:
    if not alias or not text:
        return False
    # "go" is a common verb. Count it only as the language token or as Golang.
    if alias == "go":
        return re.search(r"\bgolang\b", text, re.I) is not None or re.search(r"\bGo\b", text) is not None
    if re.search(r"[^a-z0-9]", alias, re.I):
        return alias.casefold() in text.casefold()
    return re.search(rf"\b{re.escape(alias)}\b", text, re.I) is not None


def skills_in(text: str) -> list[str]:
    found: list[str] = []
    for name, aliases in SKILLS:
        if any(term_in_text(alias, text) for alias in aliases):
            found.append(name)
    return found


def skill_in_text(skill: str, text: str) -> bool:
    aliases = dict(SKILLS).get(skill, [skill])
    return any(term_in_text(alias, text) for alias in aliases)


_NEGATION = re.compile(r"\b(not|no|never|without|nor|n't)\b", re.I)
_SENTENCE = re.compile(r"[^\n.!?]+")


def _alias_span(skill: str, text: str) -> tuple[int, int] | None:
    for alias in aliases_for(skill):
        if re.search(r"[^a-z0-9]", alias, re.I):
            match = re.search(re.escape(alias), text, re.I)
        else:
            match = re.search(rf"\b{re.escape(alias)}\b", text, re.I)
        if match:
            return match.start(), match.end()
    return None


def _sentence_is_negated(sentence: str, skill: str) -> bool:
    span = _alias_span(skill, sentence)
    if span is None:
        return False
    return _NEGATION.search(sentence[: span[0]]) is not None


def skill_is_claimed(skill: str, text: str) -> bool:
    """True when a sentence names the skill and does not negate it first.

    "I have not shipped FastAPI" is a mention, not experience.
    """
    if not skill_in_text(skill, text):
        return False
    for match in _SENTENCE.finditer(text):
        sentence = match.group(0)
        if skill_in_text(skill, sentence) and not _sentence_is_negated(sentence, skill):
            return True
    return False


def requirement_skills(requirement: str) -> list[str]:
    """Skills the bullet is asking for. Drop the contrast after 'rather than'."""
    head = re.split(r"\brather than\b|\binstead of\b", requirement, maxsplit=1, flags=re.I)[0]
    return skills_in(head)


def aliases_for(skill: str) -> list[str]:
    return dict(SKILLS).get(skill, [skill])


def significant_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{2,}", text)
    kept: list[str] = []
    seen: set[str] = set()
    for word in words:
        key = word.casefold()
        if key in STOPWORDS or key in seen:
            continue
        seen.add(key)
        kept.append(word)
    return kept[:12]
