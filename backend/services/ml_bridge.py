"""Bridge between the backend and the ``ml_engine`` package.

``ml_engine`` (datasets/models/evaluation/visualization) is implemented
separately. This module is the ONLY place in the backend that knows its
layout, so the rest of the backend only talks to the functions below.

The formal interface contracts are defined in
:mod:`backend.services.ml_contracts` (``Protocol`` classes). The algorithm
team should implement classes/modules that satisfy those protocols.

Module-level contract summary:

``ml_engine.datasets`` (package ``__init__``)
    - ``get_dataset(name) -> DatasetAdapter`` (raise ``KeyError`` if unknown).
    - ``list_datasets() -> sequence[DatasetAdapter]``.

``ml_engine.datasets.base``
    - ``split_dataset(X, y, test_size, random_state, task_type, standardize)
      -> (X_train, X_test, y_train, y_test)`` splits and standardizes the
      data. ``test_size`` in {0.2, 0.3, 0.4}. Classification splits are
      stratified on ``y``.

``ml_engine.models.base``
    - ``get_model(name) -> ModelAdapter`` (raise ``KeyError`` if unknown).
    - ``list_models(task_type: str | None = None) -> sequence[ModelAdapter]``.

``ml_engine.evaluation.metrics``  (D-owned dispatcher)
    - ``evaluate(task_type, y_true, y_pred, y_score=None, class_labels=None)
      -> dict``. Dispatches to ``evaluation.classification`` (A) or
      ``evaluation.regression`` (C). Returns ``metrics`` plus
      ``confusion_matrix`` / ``class_labels`` for classification.

``ml_engine.visualization.charts``
    - ``build_charts(task_type, metrics, y_true, y_pred) -> list[dict]``
      chart specifications (must include ``type``, ``title``, ``data``).
    - ``save_charts(charts, output_dir: Path) -> list[str]`` PNG file names.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from backend.services.ml_contracts import TaskType


class MlEngineUnavailable(RuntimeError):
    """Raised when the ml_engine package (or a submodule) is not implemented yet."""


def _import(module_name: str):
    try:
        return import_module(f"ml_engine.{module_name}")
    except ImportError as exc:
        raise MlEngineUnavailable(
            f"ml_engine submodule 'ml_engine.{module_name}' is not available yet."
        ) from exc


# --- datasets ---------------------------------------------------------------


def list_datasets() -> list[dict[str, Any]]:
    """Return lightweight dataset metadata.

    Uses the adapter's static ``n_samples`` / ``n_features`` when available;
    otherwise lazily loads the dataset once and counts (the built-in datasets
    are small). Raw ``X`` / ``y`` are never returned.
    """
    import numpy as np

    datasets = _import("datasets")
    items: list[dict[str, Any]] = []
    for adapter in datasets.list_datasets():
        n_samples = getattr(adapter, "n_samples", None)
        n_features = getattr(adapter, "n_features", None)
        if n_samples is None or n_features is None:
            data = adapter.load()
            if n_samples is None:
                n_samples = len(data.y)
            if n_features is None:
                n_features = int(np.asarray(data.X).shape[1])
        items.append(
            {
                "name": adapter.name,
                "description": adapter.description,
                "task_type": adapter.task_type,
                "n_samples": int(n_samples),
                "n_features": int(n_features),
            }
        )
    return items


def dataset_detail(name: str) -> dict[str, Any]:
    datasets = _import("datasets")
    adapter = datasets.get_dataset(name)
    data = adapter.load()
    task_type = data.task_type

    feature_names = list(data.feature_names)
    target_names = list(data.target_names)

    columns = [*feature_names, "target"]
    rows: list[list[Any]] = []
    for i in range(min(5, len(data.y))):
        row = [float(v) for v in data.X[i]]
        if task_type == "classification":
            row.append(str(target_names[int(data.y[i])]))
        else:
            row.append(float(data.y[i]))
        rows.append(row)
    return {
        "name": adapter.name,
        "description": adapter.description,
        "task_type": task_type,
        "n_samples": int(len(data.y)),
        "n_features": len(feature_names),
        "preview_columns": columns,
        "preview_rows": rows,
    }


def load_dataset(name: str) -> Any:
    datasets = _import("datasets")
    return datasets.get_dataset(name).load()


def split_dataset(
    X: Any,
    y: Any,
    test_size: float,
    random_state: int = 42,
    task_type: TaskType = "classification",
    standardize: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """Split and standardize the data via ml_engine.

    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    base = _import("datasets.base")
    return base.split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        task_type=task_type,
        standardize=standardize,
    )


# --- models -----------------------------------------------------------------


def list_models(task_type: str | None = None) -> list[dict[str, Any]]:
    base = _import("models.base")
    return [
        {
            "name": a.name,
            "description": a.description,
            "task_type": a.task_type,
            "default_parameters": dict(a.default_parameters),
            "parameters": [_serialize_param(p) for p in getattr(a, "parameters", [])],
        }
        for a in base.list_models(task_type)
    ]


def model_detail(name: str) -> dict[str, Any]:
    base = _import("models.base")
    adapter = base.get_model(name)
    return {
        "name": adapter.name,
        "description": adapter.description,
        "task_type": adapter.task_type,
        "default_parameters": dict(adapter.default_parameters),
        "parameters": [_serialize_param(p) for p in getattr(adapter, "parameters", [])],
    }


def create_estimator(
    name: str,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> Any:
    base = _import("models.base")
    return base.get_model(name).create_estimator(hyperparameters, random_state)


def get_training_info(
    estimator: Any,
    feature_names: list[str] | None = None,
    fit_duration_ms: float | None = None,
) -> dict[str, Any]:
    """Extract training metadata from a fitted estimator.

    Reads sklearn-style fitted attributes (trailing underscore) and maps
    them to JSON-friendly keys without the underscore:

    ``loss_history_`` -> ``loss_history``
    ``n_iter_``       -> ``n_iter``
    ``stop_reason_``  -> ``stop_reason``
    ``coef_``         -> ``coefficients``
    ``intercept_``    -> ``intercept``
    ``feature_importances_`` -> ``feature_importances``
    ``n_features_in_`` -> ``n_features_in``

    ``feature_names`` (when supplied) and ``fit_duration_ms`` are also
    recorded. Fields are omitted when the estimator does not expose them.

    Layer-list weights (``coefs_`` / ``intercepts_`` as exposed by
    ``MLPClassifier``) are intentionally NOT mapped: per-layer shapes would
    break the ``TrainingInfo.coefficients`` contract, and hidden-layer
    weights have no display value. For the MLP the training info is limited
    to ``loss_history`` / ``n_iter`` / ``stop_reason`` / ``n_features_in``.
    """
    import numpy as np

    info: dict[str, Any] = {}
    attr_map = {
        "loss_history_": "loss_history",
        "n_iter_": "n_iter",
        "stop_reason_": "stop_reason",
        "n_features_in_": "n_features_in",
    }
    for src, dst in attr_map.items():
        if hasattr(estimator, src):
            info[dst] = getattr(estimator, src)
    if hasattr(estimator, "coef_"):
        info["coefficients"] = np.asarray(estimator.coef_).tolist()
    if hasattr(estimator, "intercept_"):
        info["intercept"] = np.asarray(estimator.intercept_).tolist()
    if hasattr(estimator, "feature_importances_"):
        info["feature_importances"] = np.asarray(
            estimator.feature_importances_
        ).tolist()
    if feature_names is not None:
        info["feature_names"] = list(feature_names)
    if fit_duration_ms is not None:
        info["fit_duration_ms"] = float(fit_duration_ms)
    return info


def get_predict_scores(estimator: Any, X: Any) -> dict[str, Any] | None:
    """Return structured prediction scores, or ``None``.

    Returns a dict with keys:

    - ``y_score``: 2D array (n_samples, n_classes) or None
    - ``score_type``: "probability" / "decision" / None
    - ``score_classes``: class labels matching columns of y_score, or None

    Rules:

    1. ``predict_proba`` is preferred when available (score_type="probability").
    2. Otherwise ``decision_function`` is used (score_type="decision").
       For binary classification a 1-D output is converted to two columns
       where column 0 = -score and column 1 = +score (so argmax matches
       ``estimator.classes_``).
    3. If neither method exists, returns ``None``.
    4. ``score_classes`` always matches ``estimator.classes_``.
    """
    import numpy as np

    classes = getattr(estimator, "classes_", None)

    if hasattr(estimator, "predict_proba"):
        y_score = np.asarray(estimator.predict_proba(X))
        score_type = "probability"
    elif hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X))
        score_type = "decision"
        # Binary decision_function may return 1-D signed scores; convert to
        # two columns so argmax(score) matches estimator.classes_.
        if raw.ndim == 1:
            y_score = np.stack([-raw, raw], axis=1)
        else:
            y_score = raw
    else:
        return None

    if not np.all(np.isfinite(y_score)):
        raise FloatingPointError(
            "Non-finite prediction scores detected; refusing to write to database."
        )

    result: dict[str, Any] = {
        "y_score": y_score.tolist(),
        "score_type": score_type,
        "score_classes": list(classes) if classes is not None else None,
    }
    return result


# --- evaluation & visualization ----------------------------------------------


def _serialize_param(p: Any) -> dict[str, Any]:
    """Convert a ParameterSpec (dict or object) to a JSON-serializable dict."""
    source = p if isinstance(p, dict) else vars(p)
    spec = {
        "name": source["name"],
        "label": source["label"],
        "type": source["type"],
        "default": source["default"],
        "description": source.get("description", "") or "",
    }
    for key in ("min", "max", "step", "options"):
        value = source.get(key)
        if value is not None:
            spec[key] = value
    return spec


def evaluate(
    task_type: str,
    y_true: Any,
    y_pred: Any,
    y_score: Any = None,
    class_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute evaluation metrics, confusion matrix and class labels.

    Returns a dict with ``metrics``, ``confusion_matrix`` and
    ``class_labels`` keys. ``class_labels`` is passed through so the
    evaluation module can order the confusion matrix consistently.
    """
    metrics = _import("evaluation.metrics")
    return metrics.evaluate(
        task_type, y_true, y_pred, y_score=y_score, class_labels=class_labels
    )


def build_charts(
    task_type: str, metrics: dict[str, Any], y_true: Any, y_pred: Any
) -> list[dict[str, Any]]:
    charts = _import("visualization.charts")
    return charts.build_charts(task_type, metrics, y_true, y_pred)


def save_charts(charts: list[dict[str, Any]], output_dir: Path) -> list[str]:
    module = _import("visualization.charts")
    return list(module.save_charts(charts, output_dir))
