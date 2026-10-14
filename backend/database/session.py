"""Database engine, session factory, and initialization helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine = None
_session_factory: sessionmaker[Session] | None = None


def _build_engine():
    settings = get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": 30}
    return create_engine(settings.database_url, connect_args=connect_args)


def engine():
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def SessionLocal() -> Session:
    """Create a new database session (used outside request scope, e.g. background tasks)."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=engine(), autoflush=False, expire_on_commit=False)
    return _session_factory()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables and ensure the experiments output directory exists."""
    from backend.database import models  # noqa: F401 - register ORM models

    Base.metadata.create_all(bind=engine())
    Path(get_settings().experiments_dir).mkdir(parents=True, exist_ok=True)
