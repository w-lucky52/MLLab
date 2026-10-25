"""Request/response schemas.

All response models here are the public contract between the backend and the
frontend teammate. The corresponding OpenAPI schema is auto-generated at
``/docs`` and is the source of truth for API consumers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

TaskType = Literal["classification", "regression"]


# --- generic error response --------------------------------------------------


class ErrorResponse(BaseModel):
    """Uniform error body returned for all non-2xx responses."""

    detail: str


# --- datasets ----------------------------------------------------------------


class DatasetInfo(BaseModel):
    """Lightweight dataset metadata (list view).

    Includes sample/feature counts but never the raw ``X`` / ``y`` arrays.
    """

    name: str
    description: str
    task_type: TaskType
    n_samples: int
    n_features: int


class DatasetDetail(DatasetInfo):
    """Full dataset metadata with a small preview table."""

    preview_columns: list[str]
    preview_rows: list[list[float | str | int]]


# --- models ------------------------------------------------------------------


class ParameterSpec(BaseModel):
    """Hyperparameter metadata for dynamic form rendering.

    ``type`` is one of int/float/bool/select. Numeric parameters may carry
    min/max/step; select parameters carry options.
    """

    name: str
    label: str
    type: Literal["int", "float", "bool", "select"]
    default: Any
    description: str = ""
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    options: list[str] | None = None


class ModelInfo(BaseModel):
    """Model catalog entry."""

    name: str
    description: str
    task_type: TaskType
    default_parameters: dict[str, Any]
    parameters: list[ParameterSpec] = Field(default_factory=list)


# --- experiments -------------------------------------------------------------


class ExperimentCreate(BaseModel):
    dataset_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    name: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    # Allowed splits: 8:2 (0.2), 7:3 (0.3, standard), 6:4 (0.4)
    test_size: float = Field(default=0.3)
    random_state: int = 42

    @field_validator("test_size")
    @classmethod
    def _validate_test_size(cls, value: float) -> float:
        if value not in (0.2, 0.3, 0.4):
            raise ValueError("test_size must be one of 0.2, 0.3, 0.4")
        return value


class PerClassMetrics(BaseModel):
    """Precision / recall / f1 / support for one class."""

    model_config = ConfigDict(serialize_by_alias=True, populate_by_name=True)

    class_value: str | int | float = Field(alias="class")
    precision: float
    recall: float
    f1: float
    support: int


class TrainingLogEntry(BaseModel):
    """One structured entry of the experiment ``process_log``.

    ``elapsed_ms`` is the pipeline-relative time in milliseconds; it is
    ``None`` for ``iteration_checkpoint`` entries because per-iteration
    timings are not recorded by the models. ``details`` carries the
    structured payload — the frontend renders display text from it.
    """

    sequence: int
    event: str
    elapsed_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TrainingInfo(BaseModel):
    """Training metadata returned by iterative models.

    All fields are optional because not every model exposes them. The
    frontend should render only the fields that are present. Records written
    before these fields existed simply leave them as ``None``.
    """

    loss_history: list[float] | None = None
    loss_name: str | None = None
    loss_source: str | None = None
    n_iter: int | None = None
    stop_reason: str | None = None
    coefficients: list[list[float]] | list[float] | None = None
    intercept: list[float] | float | None = None
    feature_importances: list[float] | None = None
    n_features_in: int | None = None
    feature_names: list[str] | None = None
    fit_duration_ms: float | None = None
    resolved_parameters: dict[str, Any] | None = None
    standardize: bool | None = None
    train_sample_count: int | None = None
    test_sample_count: int | None = None
    # Class labels in their original ``classes_`` order. Regression models
    # and older records leave this as ``None``.
    classes: list[Any] | None = None
    # Per-class feature means / variances (GaussianNB only). Both are stored
    # together or omitted together.
    class_feature_means: list[list[float]] | None = None
    class_feature_variances: list[list[float]] | None = None
    # Real per-model explanation payload (linear contributions, KNN
    # neighbours, GaussianNB likelihoods, CART decision paths or MLP
    # permutation importance). ``None`` when the model has no faithful
    # explanation or for records written before this field existed.
    explanation_data: dict[str, Any] | None = None
    # Structured training-process log. ``None`` for experiments recorded
    # before this field existed — the frontend hides the log panel then.
    process_log: list[TrainingLogEntry] | None = None


# --- charts ------------------------------------------------------------------

ChartType = Literal["bar", "confusion_matrix", "loss_curve", "scatter"]


class ChartBarData(BaseModel):
    categories: list[str]
    values: list[float]


class ChartConfusionMatrixData(BaseModel):
    matrix: list[list[int]]
    labels: list[str]


class ChartLossCurveData(BaseModel):
    epochs: list[int]
    losses: list[float]


class ChartScatterData(BaseModel):
    points: list[list[float]]


class ChartSpec(BaseModel):
    """A single visualization artifact attached to a completed experiment.

    ``file`` is the PNG artifact name inside the experiment artifact directory.
    The image URL is ``/artifacts/{artifact_dir}/{file}``.
    """

    type: ChartType
    title: str
    x_label: str | None = None
    y_label: str | None = None
    file: str | None = None
    data: dict[str, Any]


class EvaluationResult(BaseModel):
    """Raw evaluation data rendered by the frontend (ECharts).

    Classification: ``metrics`` + ``confusion_matrix`` + ``class_labels``
    + ``y_true`` + ``y_pred`` + ``y_score`` + ``score_type``
    + ``score_classes`` + ``per_class_metrics``.
    Regression: ``metrics`` + ``y_true`` + ``y_pred`` (for actual-vs-predicted
    and residual plots). The unused fields are ``null``.
    """

    metrics: dict[str, float]
    confusion_matrix: list[list[int]] | None = None
    class_labels: list[str] | None = None
    y_true: list[float | str | int] | None = None
    y_pred: list[float | str | int] | None = None
    y_score: list[list[float]] | None = None
    score_type: Literal["probability", "decision"] | None = None
    score_classes: list[str | int | float] | None = None
    # Friendly display names aligned with ``score_classes`` (same order).
    # Display only: scoring, ROC and the confusion matrix keep using
    # ``score_classes``. ``None`` for regression or older records.
    score_class_names: list[str] | None = None
    per_class_metrics: list[PerClassMetrics] | None = None


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
 
    id: int
    name: str
    dataset_name: str
    model_name: str
    task_type: TaskType | None
    hyperparameters: dict[str, Any]
    test_size: float
    random_state: int
    status: str
    metrics: dict[str, Any] | None
    confusion_matrix: list[list[int]] | None
    class_labels: list[str] | None
    # Stored but assembled into ``evaluation`` for API responses.
    predictions: dict[str, Any] | None = Field(default=None, exclude=True)
    training_info: dict[str, Any] | None
    charts: list[ChartSpec] | None
    artifact_dir: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @computed_field
    @property
    def evaluation(self) -> EvaluationResult | None:
        """Task-shaped evaluation block the frontend consumes."""
        if self.status != "completed" or self.metrics is None:
            return None
        preds = self.predictions or {}
        if self.task_type == "regression":
            return EvaluationResult(
                metrics=self.metrics,
                y_true=preds.get("y_true"),
                y_pred=preds.get("y_pred"),
            )
        return EvaluationResult(
            metrics=self.metrics,
            confusion_matrix=self.confusion_matrix,
            class_labels=self.class_labels,
            y_true=preds.get("y_true"),
            y_pred=preds.get("y_pred"),
            y_score=preds.get("y_score"),
            score_type=preds.get("score_type"),
            score_classes=preds.get("score_classes"),
            score_class_names=preds.get("score_class_names"),
            per_class_metrics=preds.get("per_class_metrics"),
        )
