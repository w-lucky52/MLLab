"""Phase 3A — training-process data for the result page.

Covers ``resolved_parameters``, effective-parameter collection, loss-curve
provenance, standardization/sample counts and legacy compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import load_iris

from backend.api.schemas import TrainingInfo
from backend.core.config import get_settings
from backend.services import ml_bridge


def _iris_data():
    iris = load_iris()
    return iris.data[:60], iris.target[:60]


# --- get_effective_parameters ----------------------------------------------


def test_effective_parameters_fill_defaults_when_hyperparameters_empty():
    """An empty hyperparameters dict still resolves to the full default set."""
    estimator = ml_bridge.create_estimator("ridge_regression", {})

    resolved = ml_bridge.get_effective_parameters(estimator)

    assert resolved == {
        "learning_rate": 0.05,
        "max_iter": 1000,
        "tol": 1e-8,
        "alpha": 1.0,
    }


def test_effective_parameters_reflect_user_overrides():
    estimator = ml_bridge.create_estimator(
        "logistic_regression",
        {"learning_rate": 0.2, "max_iter": 123, "alpha": 0.5},
    )

    resolved = ml_bridge.get_effective_parameters(estimator)

    assert resolved["learning_rate"] == 0.2
    assert resolved["max_iter"] == 123
    assert resolved["alpha"] == 0.5
    assert resolved["tol"] == 1e-8  # untouched default


def test_effective_parameters_include_injected_random_state():
    mlp = ml_bridge.create_estimator("mlp_classifier", {}, random_state=7)
    forest = ml_bridge.create_estimator("random_forest_regressor", {}, random_state=11)

    assert ml_bridge.get_effective_parameters(mlp)["random_state"] == 7
    assert ml_bridge.get_effective_parameters(forest)["random_state"] == 11


def test_effective_parameters_do_not_fake_random_state():
    """Models whose constructor has no random_state must not gain one."""
    knn = ml_bridge.create_estimator("knn_classifier", {}, random_state=7)
    tree = ml_bridge.create_estimator("decision_tree_regressor", {}, random_state=7)

    assert "random_state" not in ml_bridge.get_effective_parameters(knn)
    assert "random_state" not in ml_bridge.get_effective_parameters(tree)


def test_effective_parameters_serialize_numpy_values():
    class NumpyEstimator:
        def __init__(self, count, tolerance, weights):
            self.count = np.int64(count)
            self.tolerance = np.float64(tolerance)
            self.weights = np.array(weights)

    estimator = NumpyEstimator(3, 0.5, [1.0, 2.0])
    resolved = ml_bridge.get_effective_parameters(estimator)

    assert resolved == {"count": 3, "tolerance": 0.5, "weights": [1.0, 2.0]}
    assert isinstance(resolved["count"], int)
    assert isinstance(resolved["tolerance"], float)
    json.dumps(resolved)


def test_effective_parameters_recursively_convert_nested_containers():
    """NumPy scalars/arrays nested in list/dict/tuple are converted recursively."""

    class NestedEstimator:
        def __init__(self, payload, weights):
            self.payload = {
                "layers": [np.int32(2), np.float32(0.25)],
                "name": ("hidden", {"size": np.int64(8)}),
                "empty": None,
            }
            self.weights = np.array([[1, 2], [3, 4]], dtype=np.float64)

    resolved = ml_bridge.get_effective_parameters(NestedEstimator(None, None))

    assert resolved == {
        "payload": {
            "layers": [2, 0.25],
            "name": ["hidden", {"size": 8}],
            "empty": None,
        },
        "weights": [[1.0, 2.0], [3.0, 4.0]],
    }
    assert isinstance(resolved["payload"]["layers"][0], int)
    assert isinstance(resolved["payload"]["layers"][1], float)
    json.dumps(resolved)


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf"), np.float64("nan")],
)
def test_effective_parameters_reject_non_finite_floats(bad_value):
    class NonFiniteEstimator:
        def __init__(self, tolerance):
            self.tolerance = bad_value

    with pytest.raises(ValueError) as excinfo:
        ml_bridge.get_effective_parameters(NonFiniteEstimator(None))

    message = str(excinfo.value)
    assert "tolerance" in message
    assert "float" in message


def test_effective_parameters_reject_unknown_objects():
    class SomeCallback:
        pass

    class CallbackEstimator:
        def __init__(self, callback):
            self.callback = SomeCallback()

    with pytest.raises(TypeError) as excinfo:
        ml_bridge.get_effective_parameters(CallbackEstimator(None))

    assert str(excinfo.value) == (
        "Cannot serialize effective parameter 'callback' of type SomeCallback."
    )


def test_effective_parameters_reject_set_values():
    class SetEstimator:
        def __init__(self, tags):
            self.tags = {1, 2}

    with pytest.raises(TypeError) as excinfo:
        ml_bridge.get_effective_parameters(SetEstimator(None))

    message = str(excinfo.value)
    assert "tags" in message
    assert "set" in message


def test_effective_parameters_reject_unknown_objects_nested_in_containers():
    class CallbackEstimator:
        def __init__(self, callbacks):
            self.callbacks = {"after_fit": [object()]}

    with pytest.raises(TypeError) as excinfo:
        ml_bridge.get_effective_parameters(CallbackEstimator(None))

    # The parameter name and its top-level type locate the bad argument; the
    # chained cause pins down the exact unserializable leaf type.
    assert "callbacks" in str(excinfo.value)
    assert "dict" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, TypeError)
    assert "object" in str(excinfo.value.__cause__)


def test_training_info_propagates_unserializable_parameter():
    """get_training_info must not silently drop an unserializable parameter."""

    class OpaqueEstimator:
        def __init__(self, callback):
            self.callback = object()

    with pytest.raises(TypeError, match="callback"):
        ml_bridge.get_training_info(OpaqueEstimator(None))


def test_all_registered_models_export_serializable_parameters():
    models = ml_bridge.list_models()
    names = [model["name"] for model in models]

    assert len(names) == 10

    for name in names:
        estimator = ml_bridge.create_estimator(name, {}, random_state=42)
        resolved = ml_bridge.get_effective_parameters(estimator)

        assert isinstance(resolved, dict), name
        assert resolved, name
        assert all(not key.endswith("_") for key in resolved), name
        json.dumps(resolved)


def test_effective_parameters_skip_trailing_underscore_attributes():
    X, y = _iris_data()
    estimator = ml_bridge.create_estimator(
        "logistic_regression", {"max_iter": 20, "tol": 0.0}
    )
    estimator.fit(X, y)
    before = {name: id(value) for name, value in vars(estimator).items()}

    resolved = ml_bridge.get_effective_parameters(estimator)

    assert resolved is not None
    assert all(not name.endswith("_") for name in resolved)
    # The estimator itself is only read.
    assert {name: id(value) for name, value in vars(estimator).items()} == before


def test_effective_parameters_none_for_object_without_constructor_args():
    assert ml_bridge.get_effective_parameters(object()) is None


# --- loss curve provenance --------------------------------------------------


def test_gd_model_loss_source_is_loss_history():
    from ml_engine.models.logistic_regression import LogisticRegressionGD

    X, y = _iris_data()
    estimator = LogisticRegressionGD(learning_rate=0.1, max_iter=30, tol=0.0)
    estimator.fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert info["loss_source"] == "loss_history_"
    assert info["loss_name"] == "objective"
    assert len(info["loss_history"]) == info["n_iter"]
    assert all(isinstance(value, float) for value in info["loss_history"])
    assert all(np.isfinite(info["loss_history"]))


def test_gbdt_train_score_maps_to_loss_history():
    from ml_engine import GradientBoostingRegressor

    X = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y = 3.0 * X[:, 0] + 1.0
    estimator = GradientBoostingRegressor(
        n_estimators=5, learning_rate=0.1, max_depth=2
    )
    estimator.fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert info["loss_source"] == "train_score_"
    assert info["loss_name"] == "training_loss"
    assert len(info["loss_history"]) == 5
    assert all(np.isfinite(info["loss_history"]))


def test_models_without_curve_get_no_loss_keys():
    from ml_engine import CartRegressionTree, RandomForestRegressor
    from ml_engine.models.knn import KNNModel
    from ml_engine.models.naive_bayes import GaussianNBModel

    X, y = _iris_data()
    X_reg = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y_reg = 2.0 * X_reg[:, 0] - 1.0

    estimators = [
        KNNModel().fit(X, y),
        GaussianNBModel().fit(X, y),
        CartRegressionTree(max_depth=2).fit(X_reg, y_reg),
        RandomForestRegressor(n_estimators=3, random_state=0).fit(X_reg, y_reg),
    ]

    for estimator in estimators:
        info = ml_bridge.get_training_info(estimator)
        assert "loss_history" not in info, type(estimator).__name__
        assert "loss_name" not in info, type(estimator).__name__
        assert "loss_source" not in info, type(estimator).__name__


@pytest.mark.parametrize(
    "curve",
    [
        np.array([1.0, np.nan, 0.5]),
        np.array([1.0, np.inf]),
        np.array([[1.0, 2.0]]),
        np.array([]),
        ["not-a-number", "0.5"],
    ],
)
def test_invalid_loss_curves_are_not_written(curve):
    class InvalidCurve:
        loss_history_ = curve

    info = ml_bridge.get_training_info(InvalidCurve())

    assert "loss_history" not in info
    assert "loss_name" not in info
    assert "loss_source" not in info


def test_invalid_train_score_is_not_written():
    class InvalidTrainScore:
        train_score_ = np.array([0.5, np.nan])

    info = ml_bridge.get_training_info(InvalidTrainScore())

    assert "loss_history" not in info
    assert "loss_source" not in info


# --- experiment pipeline integration ---------------------------------------


def _fake_dataset():
    return SimpleNamespace(
        X=np.tile(np.arange(8, dtype=float).reshape(4, 2), (10, 1)),
        y=np.array([0, 1] * 20),
        task_type="classification",
        feature_names=["f0", "f1"],
        target_names=["class_0", "class_1"],
    )


def _fake_split(X, y, test_size, random_state=42, task_type="classification",
                standardize=True):
    n = len(y)
    n_test = max(1, int(round(n * test_size)))
    return X[n_test:], X[:n_test], y[n_test:], y[:n_test]


def _fake_evaluate(task_type, y_true, y_pred, y_score=None, class_labels=None):
    if task_type != "classification":
        return {"metrics": {"r2": 0.0}, "confusion_matrix": None, "class_labels": None}
    return {
        "metrics": {"accuracy": 1.0},
        "confusion_matrix": [[1, 0], [0, 1]],
        "class_labels": class_labels,
        "per_class_metrics": [
            {"class": "class_0", "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
            {"class": "class_1", "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
        ],
    }


def _fake_model_detail(name):
    return {"name": name, "description": "", "task_type": "classification",
            "default_parameters": {}}


class _ParamEchoClassifier:
    """Mirrors the adapter merge: defaults + user params + injected seed."""

    classes_ = np.array([0, 1])

    def __init__(self, alpha=1.0, max_iter=1000, random_state=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


def _patch_pipeline(monkeypatch, split=None):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail)
    monkeypatch.setattr(
        ml_bridge,
        "create_estimator",
        lambda name, params, random_state=None: _ParamEchoClassifier(
            alpha=params.get("alpha", 1.0),
            max_iter=params.get("max_iter", 1000),
            random_state=random_state,
        ),
    )
    monkeypatch.setattr(ml_bridge, "split_dataset", split or _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)


def test_hyperparameters_keep_user_values_while_resolved_has_defaults(
    client, monkeypatch
):
    _patch_pipeline(monkeypatch)

    created = client.post(
        "/api/v1/experiments",
        json={
            "dataset_name": "iris",
            "model_name": "logistic_regression",
            "hyperparameters": {"alpha": 0.25, "max_iter": 50},
        },
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    assert detail["hyperparameters"] == {"alpha": 0.25, "max_iter": 50}
    assert detail["training_info"]["resolved_parameters"] == {
        "alpha": 0.25,
        "max_iter": 50,
        "random_state": 42,
    }


def test_hyperparameters_stay_user_submitted_after_resolution(client, monkeypatch):
    """resolved_parameters must never leak into the stored hyperparameters."""
    _patch_pipeline(monkeypatch)

    created = client.post(
        "/api/v1/experiments",
        json={
            "dataset_name": "iris",
            "model_name": "logistic_regression",
            "hyperparameters": {"alpha": 0.25},
        },
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    assert detail["hyperparameters"] == {"alpha": 0.25}
    assert detail["training_info"]["resolved_parameters"] == {
        "alpha": 0.25,
        "max_iter": 1000,
        "random_state": 42,
    }
    assert "max_iter" not in detail["hyperparameters"]


def test_empty_hyperparameters_still_records_full_defaults(client, monkeypatch):
    _patch_pipeline(monkeypatch)

    created = client.post(
        "/api/v1/experiments",
        json={"dataset_name": "iris", "model_name": "logistic_regression"},
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    assert detail["hyperparameters"] == {}
    assert detail["training_info"]["resolved_parameters"] == {
        "alpha": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    }


def test_standardize_and_sample_counts_are_recorded(client, monkeypatch):
    captured = {}

    def recording_split(X, y, test_size, random_state=42,
                        task_type="classification", standardize=True):
        captured["standardize"] = standardize
        return _fake_split(X, y, test_size, random_state, task_type, standardize)

    _patch_pipeline(monkeypatch, split=recording_split)

    created = client.post(
        "/api/v1/experiments",
        json={"dataset_name": "iris", "model_name": "logistic_regression"},
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    info = detail["training_info"]
    assert captured["standardize"] is True
    assert info["standardize"] is True
    assert info["train_sample_count"] == 28
    assert info["test_sample_count"] == 12
    assert info["train_sample_count"] + info["test_sample_count"] == 40


def test_summary_json_stores_the_same_training_info(client, monkeypatch):
    _patch_pipeline(monkeypatch)

    created = client.post(
        "/api/v1/experiments",
        json={"dataset_name": "iris", "model_name": "logistic_regression"},
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    summary_path = (
        Path(get_settings().experiments_dir)
        / detail["artifact_dir"]
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["training_info"] == detail["training_info"]
    assert summary["hyperparameters"] == detail["hyperparameters"] == {}


# --- legacy compatibility ---------------------------------------------------


def test_training_info_schema_parses_legacy_payload():
    legacy = TrainingInfo.model_validate(
        {
            "loss_history": [1.0, 0.5],
            "n_iter": 2,
            "stop_reason": "converged",
            "feature_names": ["f0"],
        }
    )

    assert legacy.loss_history == [1.0, 0.5]
    assert legacy.resolved_parameters is None
    assert legacy.standardize is None
    assert legacy.train_sample_count is None
    assert legacy.test_sample_count is None
    assert legacy.loss_name is None
    assert legacy.loss_source is None


def test_training_info_schema_accepts_all_new_fields():
    info = TrainingInfo.model_validate(
        {
            "loss_history": [2.0, 1.0],
            "loss_name": "training_loss",
            "loss_source": "train_score_",
            "resolved_parameters": {"n_estimators": 5, "learning_rate": 0.1},
            "standardize": True,
            "train_sample_count": 28,
            "test_sample_count": 12,
        }
    )

    assert info.resolved_parameters == {"n_estimators": 5, "learning_rate": 0.1}
    assert info.standardize is True
    assert info.train_sample_count == 28
    assert info.test_sample_count == 12
    assert info.loss_name == "training_loss"
    assert info.loss_source == "train_score_"
