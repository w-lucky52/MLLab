"""Formal contracts for the ``ml_engine`` package.

This module defines the Python interfaces (``typing.Protocol``) that the
``ml_engine`` package is expected to provide. The backend only depends on
these contracts (via :mod:`backend.services.ml_bridge`); the actual
implementations live in the ``ml_engine`` package owned by the algorithm team.

To implement a compatible ml_engine module, define classes that satisfy the
protocols below. The function-level contracts (e.g. ``get_dataset``) are
documented in :mod:`backend.services.ml_bridge`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TaskType = str  # one of the constants above


# --- datasets ----------------------------------------------------------------


@runtime_checkable
class DatasetData(Protocol):
    """Object returned by ``DatasetAdapter.load()``.

    All five fields are required so that downstream consumers (backend,
    evaluation, visualization) can rely on a uniform shape.
    """

    X: Any  # numpy array-like of shape (n_samples, n_features)
    y: Any  # numpy array-like of shape (n_samples,)
    task_type: TaskType
    feature_names: Sequence[str]
    target_names: Sequence[str]


@runtime_checkable
class DatasetAdapter(Protocol):
    """A single registered dataset.

    ``n_samples`` / ``n_features`` are optional static class attributes. The
    dataset list endpoint reads them without loading data; when an adapter
    does not provide them, the backend lazily calls ``load()`` once as a
    fallback (cheap for the built-in toy datasets).
    """

    name: str
    description: str
    task_type: TaskType
    n_samples: int | None = None
    n_features: int | None = None

    def load(self) -> DatasetData: ...


class DatasetRegistry(Protocol):
    """Dataset registry contract.

    Module layout as merged by A (PR #2):

    - ``get_dataset`` / ``list_datasets`` live in the ``ml_engine.datasets``
      package (``__init__.py``);
    - :func:`split_dataset` lives in ``ml_engine.datasets.base``.
    """

    @staticmethod
    def get_dataset(name: str) -> DatasetAdapter: ...

    @staticmethod
    def list_datasets() -> Sequence[DatasetAdapter]: ...

    @staticmethod
    def split_dataset(
        X: Any,
        y: Any,
        test_size: float,
        random_state: int = 42,
        task_type: TaskType = "classification",
        standardize: bool = True,
    ) -> tuple[Any, Any, Any, Any]:
        """Split and (optionally) standardize the data.

        Returns ``(X_train, X_test, y_train, y_test)``. ``test_size`` must be
        one of ``0.2``, ``0.3`` or ``0.4``. For classification tasks
        (``task_type="classification"``) the split is stratified on ``y``;
        for regression it is not. ``random_state`` defaults to ``42`` for
        reproducibility.

        When ``standardize=True``, a scaler is fit on ``X_train`` only and
        applied to both ``X_train`` and ``X_test``. ``y`` is never
        standardized.
        """
        ...


# --- models ------------------------------------------------------------------


# Allowed parameter widget types, agreed across backend / algorithms / frontend.
PARAMETER_TYPES = ("int", "float", "bool", "select")


@runtime_checkable
class ParameterSpec(Protocol):
    """Metadata for a single model hyperparameter.

    Used by the frontend to dynamically render parameter input forms.

    ``type`` is one of :data:`PARAMETER_TYPES`. Numeric params (``int`` /
    ``float``) may carry ``min`` / ``max`` / ``step``; a ``select`` param
    carries ``options``. ``description`` is an optional longer explanation
    shown next to the input.
    """

    name: str
    label: str
    type: str  # "int" | "float" | "bool" | "select"
    default: Any
    description: str
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    options: Sequence[str] | None = None  # for type="select"


@runtime_checkable
class ModelAdapter(Protocol):
    """A single registered model/algorithm."""

    name: str
    description: str
    task_type: TaskType
    default_parameters: dict[str, Any]
    parameters: Sequence[ParameterSpec]

    def create_estimator(
        self,
        hyperparameters: dict[str, Any] | None = None,
        random_state: int | None = None,
    ) -> Any:
        """Return a fresh scikit-learn style estimator (``fit``/``predict``).

        ``random_state`` carries the experiment-level seed. Implementations
        pass it to the estimator only when its constructor accepts one
        (e.g. tree ensembles); models without a ``random_state`` parameter
        simply ignore it.
        """
        ...


class ModelRegistry(Protocol):
    """``ml_engine.models.base`` module-level contract."""

    @staticmethod
    def get_model(name: str) -> ModelAdapter: ...

    @staticmethod
    def list_models(task_type: TaskType | None = None) -> Sequence[ModelAdapter]: ...


# --- evaluation --------------------------------------------------------------


# Fixed metric keys so the frontend, evaluation module and report generator
# all use the same vocabulary.
#
# Classification scores are label-weighted (proposed by B, pending E
# confirmation); the confusion matrix is a separate non-scalar field, not a
# metric. AUC is out of scope this phase (clustering removed, no ROC).
CLASSIFICATION_METRICS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
REGRESSION_METRICS = ("mae", "mse", "rmse", "r2")


class EvaluationFn(Protocol):
    """``ml_engine.evaluation.metrics.evaluate`` signature.

    Returns a dict with key ``metrics`` (always) plus, for classification
    tasks, ``confusion_matrix`` and ``class_labels``:

    ``metrics``
        A JSON-serializable dict whose keys are a subset of
        :data:`CLASSIFICATION_METRICS` for classification tasks or
        :data:`REGRESSION_METRICS` for regression tasks.

    ``confusion_matrix``
        A 2-D list of ints (classification only). Row order is true label,
        column order is predicted label. Omit / ``None`` for regression.

    ``class_labels``
        A list of label strings matching the order of ``confusion_matrix``
        rows/columns (classification only). Omit / ``None`` for regression.

    Regression ``y_true`` / ``y_pred`` arrays are NOT this function's
    responsibility: the backend already holds them and attaches them to the
    API ``evaluation`` block itself (for actual-vs-predicted and residual
    plots).

    ``y_score`` is the predicted probability / decision score; it is
    ``None`` when the model does not expose ``predict_proba`` or
    ``decision_function``. It is accepted for forward compatibility (e.g. a
    future ROC/AUC requirement) and currently unused by the metric set.
    """

    def __call__(
        self,
        task_type: TaskType,
        y_true: Any,
        y_pred: Any,
        y_score: Any | None = None,
        class_labels: Sequence[str] | None = None,
    ) -> dict[str, Any]: ...


# --- visualization -----------------------------------------------------------


class ChartsModule(Protocol):
    """``ml_engine.visualization.charts`` module-level contract."""

    @staticmethod
    def build_charts(
        task_type: TaskType, metrics: dict[str, Any], y_true: Any, y_pred: Any
    ) -> list[dict[str, Any]]:
        """Return JSON-serializable chart specifications.

        Each spec must contain at least ``type``, ``title`` and ``data`` keys,
        and may include ``x_label`` / ``y_label``.
        """
        ...

    @staticmethod
    def save_charts(charts: list[dict[str, Any]], output_dir: Path) -> list[str]:
        """Render chart specs to PNG files; return their file names."""
        ...
