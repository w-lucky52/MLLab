"""ORM models."""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.session import Base


class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Experiment(Base):
    """A single machine-learning experiment run."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    dataset_name: Mapped[str] = mapped_column(String(100))
    model_name: Mapped[str] = mapped_column(String(100))
    task_type: Mapped[str | None] = mapped_column(String(20), default=None)
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    test_size: Mapped[float] = mapped_column(Float, default=0.2)
    random_state: Mapped[int] = mapped_column(Integer, default=42)

    status: Mapped[str] = mapped_column(String(20), default=ExperimentStatus.PENDING.value)
    metrics: Mapped[dict | None] = mapped_column(JSON, default=None)
    confusion_matrix: Mapped[list | None] = mapped_column(JSON, default=None)
    class_labels: Mapped[list | None] = mapped_column(JSON, default=None)
    # Regression only: {"y_true": [...], "y_pred": [...]} for scatter/residual plots.
    predictions: Mapped[dict | None] = mapped_column(JSON, default=None)
    training_info: Mapped[dict | None] = mapped_column(JSON, default=None)
    charts: Mapped[list | None] = mapped_column(JSON, default=None)
    artifact_dir: Mapped[str | None] = mapped_column(String(255), default=None)
    error: Mapped[str | None] = mapped_column(String(2000), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
