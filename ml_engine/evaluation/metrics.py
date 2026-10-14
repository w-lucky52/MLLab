"""Unified evaluation entry point (owned by D, backend/platform side).

This module only **dispatches** by ``task_type``; the actual metric
implementations live in teammate-owned submodules:

``ml_engine.evaluation.classification``  (owned by A)
    Must expose::

        evaluate(y_true, y_pred, y_score=None, class_labels=None) -> dict

    Returns::

        {
            "metrics": {
                "accuracy": float,
                "precision_weighted": float,
                "recall_weighted": float,
                "f1_weighted": float,
            },
            "confusion_matrix": list[list[int]],  # rows=true, cols=pred
            "class_labels": list[str],            # same order as matrix
        }

``ml_engine.evaluation.regression``  (owned by C)
    Must expose::

        evaluate(y_true, y_pred) -> dict

    Returns::

        {
            "metrics": {
                "mae": float,
                "mse": float,
                "rmse": float,
                "r2": float,
            }
        }

All values must be plain JSON-serializable Python types (convert numpy
scalars explicitly). If an owner module has not been implemented yet, the
dispatcher raises a clear error instead of returning partial results.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any

from backend.services.ml_contracts import (
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)


def _load_owner_module(module_path: str, owner: str) -> ModuleType:
    """Import a teammate-owned evaluation submodule.

    Raises a clear ``RuntimeError`` naming the owner when the module has not
    been implemented yet, but lets any error raised *inside* an existing
    module (bad dependency, syntax error, circular import) propagate loudly.
    """
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        raise RuntimeError(
            f"{module_path} (owned by {owner}) is not available yet; "
            "experiments with this task type cannot be evaluated."
        )
    return importlib.import_module(module_path)


def evaluate(
    task_type: str,
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
    class_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Dispatch evaluation to the task-specific implementation.

    Raises
    ------
    ValueError
        If ``task_type`` is neither classification nor regression.
    RuntimeError
        If the owner submodule (``classification`` / ``regression``) is not
        available yet. The message names the responsible teammate.
    """
    if task_type == TASK_CLASSIFICATION:
        classification = _load_owner_module(
            "ml_engine.evaluation.classification", owner="A"
        )
        return classification.evaluate(
            y_true, y_pred, y_score=y_score, class_labels=class_labels
        )

    if task_type == TASK_REGRESSION:
        regression = _load_owner_module("ml_engine.evaluation.regression", owner="C")
        return regression.evaluate(y_true, y_pred)

    raise ValueError(f"Unknown task_type: {task_type!r}")
