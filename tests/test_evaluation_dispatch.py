"""Tests for the D-owned evaluation dispatcher."""

import pytest

from ml_engine.evaluation import metrics as eval_metrics
from ml_engine.models.base import _SklearnModelAdapter


def test_classification_dispatches_to_a_module():
    result = eval_metrics.evaluate(
        "classification",
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        class_labels=["setosa", "versicolor"],
    )

    assert set(result["metrics"]) == {
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    }
    assert result["metrics"]["accuracy"] == 0.75
    # Rows are true labels, columns predicted; order follows class_labels.
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["class_labels"] == ["setosa", "versicolor"]


def test_regression_dispatches_to_c_implementation():
    result = eval_metrics.evaluate("regression", [1.0, 2.0, 3.0], [1.0, 2.0, 5.0])

    assert set(result["metrics"]) == {"mae", "mse", "rmse", "r2"}
    assert result["metrics"]["mae"] == pytest.approx(2.0 / 3.0)
    assert result["metrics"]["mse"] == pytest.approx(4.0 / 3.0)
    assert result["metrics"]["rmse"] == pytest.approx((4.0 / 3.0) ** 0.5)
    assert result["metrics"]["r2"] == pytest.approx(-1.0)
    assert result.get("confusion_matrix") is None


def test_unknown_task_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown task_type"):
        eval_metrics.evaluate("clustering", [], [])


class _DummyEstimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _SeededEstimator:
    def __init__(self, random_state=None, **kwargs):
        self.kwargs = {"random_state": random_state, **kwargs}


def test_adapter_injects_random_state_only_when_supported():
    seeded = _SklearnModelAdapter(
        name="seeded", cls=_SeededEstimator, task_type="regression",
        description="", parameters=[],
    )
    plain = _SklearnModelAdapter(
        name="plain", cls=_DummyEstimator, task_type="regression",
        description="", parameters=[],
    )

    assert seeded.create_estimator({}, random_state=42).kwargs["random_state"] == 42
    # Models without a random_state constructor parameter ignore the seed.
    assert "random_state" not in plain.create_estimator({}, random_state=42).kwargs


def test_adapter_derives_default_parameters_from_parameter_defaults():
    adapter = _SklearnModelAdapter(
        name="dummy",
        cls=_DummyEstimator,
        task_type="regression",
        description="",
        parameters=[
            {"name": "learning_rate", "label": "lr", "type": "float",
             "default": 0.05, "description": ""},
            {"name": "max_iter", "label": "it", "type": "int",
             "default": 1000, "description": ""},
        ],
    )

    assert adapter.default_parameters == {"learning_rate": 0.05, "max_iter": 1000}
    # User overrides take precedence over derived defaults.
    estimator = adapter.create_estimator({"max_iter": 50})
    assert estimator.kwargs == {"learning_rate": 0.05, "max_iter": 50}
