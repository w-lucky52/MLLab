"""Phase 4A — model-explanation data added to ``training_info``.

Covers class-label collection (``classes``) and GaussianNB per-class feature
statistics (``class_feature_means`` / ``class_feature_variances``). All new
fields are optional and defensive: when a value cannot be reproduced
faithfully it is omitted instead of fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

from backend.api.schemas import TrainingInfo
from backend.core.config import get_settings
from backend.services import ml_bridge


def _iris_data():
    iris = load_iris()
    return iris.data[:60], iris.target[:60]


def _small_gaussian_data():
    X = np.array(
        [
            [0.0, 1.0],
            [2.0, 2.0],
            [1.0, 0.0],
            [10.0, 10.0],
            [12.0, 10.0],
            [11.0, 11.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# --- classes_ collection ----------------------------------------------------


def test_logistic_regression_classes_follow_classes():
    from ml_engine.models.logistic_regression import LogisticRegressionGD

    X, y = _iris_data()
    labels = np.where(y == 0, "versicolor", "setosa")
    estimator = LogisticRegressionGD(learning_rate=0.1, max_iter=30, tol=0.0)
    estimator.fit(X, labels)

    info = ml_bridge.get_training_info(estimator)

    assert info["classes"] == list(estimator.classes_)
    assert info["classes"] == ["setosa", "versicolor"]
    assert all(isinstance(label, str) for label in info["classes"])


def test_linear_svm_classes_follow_classes():
    from ml_engine.models.linear_svm import LinearSVM

    X, y = _iris_data()
    estimator = LinearSVM(learning_rate=0.01, max_iter=100, tol=0.0).fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert info["classes"] == [0, 1]
    assert info["classes"] == list(estimator.classes_)


def test_knn_classes_are_recorded():
    from ml_engine.models.knn import KNNModel

    X, y = _iris_data()
    labels = np.where(y == 0, "b", "a")
    estimator = KNNModel().fit(X, labels)

    info = ml_bridge.get_training_info(estimator)

    assert info["classes"] == list(estimator.classes_)
    assert set(info["classes"]) == {"a", "b"}


def test_gaussian_nb_classes_are_recorded():
    from ml_engine.models.naive_bayes import GaussianNBModel

    X, y = _iris_data()
    estimator = GaussianNBModel().fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert info["classes"] == [0, 1]
    assert info["classes"] == list(estimator.classes_)


def test_mlp_classes_are_recorded():
    from ml_engine.models.mlp_classifier import MLPClassifier

    X, y = _iris_data()
    estimator = MLPClassifier(hidden_size=4, max_iter=20, tol=0.0).fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert info["classes"] == [0, 1]
    assert info["classes"] == list(estimator.classes_)


def test_string_classes_are_json_serializable():
    from ml_engine.models.knn import KNNModel

    X, y = _iris_data()
    estimator = KNNModel().fit(X, np.where(y == 0, "class_b", "class_a"))

    info = ml_bridge.get_training_info(estimator)

    payload = json.dumps(info)
    assert json.loads(payload)["classes"] == ["class_a", "class_b"]


@pytest.mark.parametrize(
    "classes",
    [
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0, float("-inf")], dtype=object),
    ],
)
def test_non_finite_numeric_classes_are_omitted(classes):
    class NonFiniteClasses:
        def __init__(self):
            self.classes_ = classes

    info = ml_bridge.get_training_info(NonFiniteClasses())

    assert "classes" not in info
    assert json.dumps(info)


def test_regression_model_does_not_record_classes():
    from ml_engine.models.linear_regression import LinearRegressionGD

    X = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y = 3.0 * X[:, 0] + 1.0
    estimator = LinearRegressionGD(learning_rate=0.05, max_iter=50).fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert "classes" not in info
    assert "class_feature_means" not in info
    assert "class_feature_variances" not in info


# --- GaussianNB statistics --------------------------------------------------


def test_gaussian_nb_statistics_match_independent_sklearn_fit():
    from ml_engine.models.naive_bayes import GaussianNBModel

    X, y = _small_gaussian_data()
    estimator = GaussianNBModel(var_smoothing=1e-9).fit(X, y)
    expected = GaussianNB(var_smoothing=1e-9).fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    np.testing.assert_allclose(info["class_feature_means"], expected.theta_)
    np.testing.assert_allclose(info["class_feature_variances"], expected.var_)


def test_gaussian_nb_rows_follow_classes_order():
    from ml_engine.models.naive_bayes import GaussianNBModel

    X, y = _small_gaussian_data()
    estimator = GaussianNBModel().fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert info["classes"] == list(estimator.classes_)
    means = np.asarray(info["class_feature_means"])
    assert means.shape[0] == len(info["classes"])
    for row, label in zip(means, info["classes"]):
        np.testing.assert_allclose(row, X[y == label].mean(axis=0))


def test_gaussian_nb_columns_align_with_feature_names():
    from ml_engine.models.naive_bayes import GaussianNBModel

    X, y = _iris_data()
    estimator = GaussianNBModel().fit(X, y)

    aligned = ml_bridge.get_training_info(estimator, feature_names=["a", "b", "c", "d"])
    mismatched = ml_bridge.get_training_info(estimator, feature_names=["a", "b"])

    assert len(aligned["class_feature_means"][0]) == 4
    assert len(aligned["class_feature_variances"][0]) == 4
    assert "class_feature_means" not in mismatched
    assert "class_feature_variances" not in mismatched


def _statistics_estimator(theta, variances, classes=None, n_features_in=None):
    class Statistics:
        def __init__(self):
            self.classes_ = np.array([0, 1]) if classes is None else classes
            self.theta_ = theta
            self.var_ = variances
            if n_features_in is not None:
                self.n_features_in_ = n_features_in

    return Statistics()


def test_mismatched_theta_and_var_shapes_write_neither_field():
    estimator = _statistics_estimator(np.zeros((2, 3)), np.zeros((2, 2)))

    info = ml_bridge.get_training_info(estimator)

    assert "class_feature_means" not in info
    assert "class_feature_variances" not in info


def test_row_count_must_match_class_count():
    estimator = _statistics_estimator(np.zeros((3, 2)), np.ones((3, 2)))

    info = ml_bridge.get_training_info(estimator)

    assert "class_feature_means" not in info
    assert "class_feature_variances" not in info


def test_column_count_must_match_n_features_in():
    estimator = _statistics_estimator(
        np.zeros((2, 2)), np.ones((2, 2)), n_features_in=3
    )

    info = ml_bridge.get_training_info(estimator)

    assert "class_feature_means" not in info
    assert "class_feature_variances" not in info


@pytest.mark.parametrize(
    "theta, variances",
    [
        (np.array([[0.0, np.nan], [1.0, 2.0]]), np.ones((2, 2))),
        (np.zeros((2, 2)), np.array([[0.0, 1.0], [np.inf, 2.0]])),
    ],
)
def test_non_finite_statistics_write_neither_field(theta, variances):
    estimator = _statistics_estimator(theta, variances)

    info = ml_bridge.get_training_info(estimator)

    assert "class_feature_means" not in info
    assert "class_feature_variances" not in info


@pytest.mark.parametrize("present", ["theta_", "var_"])
def test_missing_one_statistic_attribute_writes_neither_field(present):
    class PartialStatistics:
        def __init__(self):
            self.classes_ = np.array([0, 1])
            if present == "theta_":
                self.theta_ = np.zeros((2, 2))
            else:
                self.var_ = np.ones((2, 2))

    info = ml_bridge.get_training_info(PartialStatistics())

    assert "class_feature_means" not in info
    assert "class_feature_variances" not in info


# --- forbidden expansions ---------------------------------------------------


def test_mlp_does_not_expose_layer_weights():
    from ml_engine.models.mlp_classifier import MLPClassifier

    X, y = _iris_data()
    estimator = MLPClassifier(hidden_size=4, max_iter=20, tol=0.0).fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert "coefs_" not in info
    assert "intercepts_" not in info
    assert "coefficients" not in info
    assert "intercept" not in info
    assert "feature_importances" not in info


def test_knn_and_gaussian_nb_have_no_fabricated_importance():
    from ml_engine.models.knn import KNNModel
    from ml_engine.models.naive_bayes import GaussianNBModel

    X, y = _iris_data()
    for estimator in (KNNModel().fit(X, y), GaussianNBModel().fit(X, y)):
        info = ml_bridge.get_training_info(estimator)
        assert "feature_importances" not in info, type(estimator).__name__


def test_predictions_score_is_not_copied_into_training_info():
    from ml_engine.models.logistic_regression import LogisticRegressionGD

    X, y = _iris_data()
    estimator = LogisticRegressionGD(learning_rate=0.1, max_iter=30, tol=0.0)
    estimator.fit(X, y)

    info = ml_bridge.get_training_info(estimator)

    assert "y_score" not in info
    assert "score_classes" not in info


# --- JSON / schema compatibility --------------------------------------------


def test_all_model_training_info_is_json_serializable():
    from ml_engine import GradientBoostingRegressor
    from ml_engine.models.knn import KNNModel
    from ml_engine.models.linear_regression import LinearRegressionGD
    from ml_engine.models.linear_svm import LinearSVM
    from ml_engine.models.logistic_regression import LogisticRegressionGD
    from ml_engine.models.mlp_classifier import MLPClassifier
    from ml_engine.models.naive_bayes import GaussianNBModel

    X, y = _iris_data()
    X_reg = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y_reg = 2.0 * X_reg[:, 0] - 1.0

    estimators = [
        KNNModel().fit(X, y),
        GaussianNBModel().fit(X, y),
        LogisticRegressionGD(learning_rate=0.1, max_iter=20, tol=0.0).fit(X, y),
        LinearSVM(learning_rate=0.01, max_iter=20, tol=0.0).fit(X, y),
        MLPClassifier(hidden_size=4, max_iter=10, tol=0.0).fit(X, y),
        LinearRegressionGD(learning_rate=0.05, max_iter=20).fit(X_reg, y_reg),
        GradientBoostingRegressor(n_estimators=3, learning_rate=0.1, max_depth=2)
        .fit(X_reg, y_reg),
    ]

    for estimator in estimators:
        info = ml_bridge.get_training_info(
            estimator, feature_names=["f0", "f1", "f2", "f3"], fit_duration_ms=1.5
        )
        payload = json.dumps(info)
        assert json.loads(payload) == info, type(estimator).__name__


def test_training_info_schema_parses_legacy_payload():
    legacy = TrainingInfo.model_validate(
        {
            "loss_history": [1.0, 0.5],
            "n_iter": 2,
            "stop_reason": "converged",
            "feature_names": ["f0"],
        }
    )

    assert legacy.classes is None
    assert legacy.class_feature_means is None
    assert legacy.class_feature_variances is None


def test_training_info_schema_accepts_new_fields():
    info = TrainingInfo.model_validate(
        {
            "classes": ["a", 1, 1.5],
            "class_feature_means": [[0.0, 1.0], [2.0, 3.0]],
            "class_feature_variances": [[1.0, 1.0], [2.0, 2.0]],
        }
    )

    assert info.classes == ["a", 1, 1.5]
    assert info.class_feature_means == [[0.0, 1.0], [2.0, 3.0]]
    assert info.class_feature_variances == [[1.0, 1.0], [2.0, 2.0]]


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
    return {
        "metrics": {"accuracy": 1.0},
        "confusion_matrix": [[1, 0], [0, 1]],
        "class_labels": class_labels,
        "per_class_metrics": [],
    }


class _FakeGaussianClassifier:
    classes_ = np.array([0, 1])
    n_features_in_ = 2
    theta_ = np.array([[0.0, 1.0], [2.0, 3.0]])
    var_ = np.array([[1.0, 1.0], [2.0, 2.0]])

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


def _patch_pipeline(monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(
        ml_bridge,
        "model_detail",
        lambda name: {"name": name, "description": "",
                      "task_type": "classification", "default_parameters": {}},
    )
    monkeypatch.setattr(
        ml_bridge,
        "create_estimator",
        lambda name, params, random_state=None: _FakeGaussianClassifier(),
    )
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)


def test_summary_json_contains_model_explanation_fields(client, monkeypatch):
    _patch_pipeline(monkeypatch)

    created = client.post(
        "/api/v1/experiments",
        json={"dataset_name": "iris", "model_name": "gaussian_nb"},
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    info = detail["training_info"]
    assert info["classes"] == [0, 1]
    assert info["class_feature_means"] == [[0.0, 1.0], [2.0, 3.0]]
    assert info["class_feature_variances"] == [[1.0, 1.0], [2.0, 2.0]]

    summary_path = (
        Path(get_settings().experiments_dir) / detail["artifact_dir"] / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["training_info"] == info
