"""Regression evaluation — contract surface (frozen with D).

C's implementation lives in :mod:`ml_engine.metrics.regression` as four
standalone functions. This module adapts them to the frozen dispatcher
contract::

    evaluate(y_true, y_pred) -> {"metrics": {"mae", "mse", "rmse", "r2"}}

matching the shape of ``ml_engine.evaluation.classification.evaluate``.
All values are plain floats (the underlying functions convert numpy
scalars explicitly), so the result is JSON-serializable as-is.
"""

from __future__ import annotations

from typing import Any

from ml_engine.metrics import regression as _impl


def evaluate(y_true: Any, y_pred: Any, y_score: Any | None = None) -> dict[str, Any]:
    """Return the four standard regression metrics for a prediction.

    ``y_score`` is accepted and ignored for signature symmetry with the
    classification contract.
    """
    return {
        "metrics": {
            "mae": _impl.mean_absolute_error(y_true, y_pred),
            "mse": _impl.mean_squared_error(y_true, y_pred),
            "rmse": _impl.root_mean_squared_error(y_true, y_pred),
            "r2": _impl.r2_score(y_true, y_pred),
        }
    }
