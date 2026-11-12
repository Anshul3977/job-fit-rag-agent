"""Engine and session factory. Created lazily so tests can set env first."""

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models import Base

_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _Session
    if _engine is not None:
        return _engine
    settings = get_settings()
    url = settings.sqlalchemy_url
    connect_args: dict = {}
    if url.startswith("sqlite"):
        if url not in {"sqlite://", "sqlite:///:memory:"}:
            raw = url[len("sqlite:///") :]
            if raw and raw != ":memory:":
                Path(raw).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False}
    _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    _Session = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    get_engine()
    assert _Session is not None
    return _Session


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    if engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            conn.execute(text("PRAGMA foreign_keys=ON"))


def get_db() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine. Tests use this after pointing DATABASE_URL at a temp file."""
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _Session = None


def ping_db() -> None:
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
