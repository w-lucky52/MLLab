"""Shared fixtures: in-memory database and API test client."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import database
from backend.core.config import get_settings
from backend.database import session as db_session
from backend.database.models import Base
from backend.main import app


@pytest.fixture()
def client(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(db_session, "SessionLocal", testing_session_factory)
    monkeypatch.setattr(database, "init_db", lambda: None)
    monkeypatch.setattr(get_settings(), "experiments_dir", str(tmp_path / "experiments"))

    with TestClient(app) as test_client:
        yield test_client
