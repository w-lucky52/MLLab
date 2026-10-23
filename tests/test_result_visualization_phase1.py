"""Phase 1 — result visualization data contract expansion.

Covers per-class classification metrics, model ``predict_proba`` exposure,
structured prediction scores, extended predictions/training_info storage
and API schema additions.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from backend.services import ml_bridge
from ml_engine.evaluation import classification as clf_eval
from ml_engine.models.knn import KNNModel
from ml_engine.models.naive_bayes import GaussianNBModel

# --- classification.per_class_metrics ---------------------------------------


def test_per_class_metrics_values_and_order():
    """Per-class precision/recall/f1/support are correct and ordered by class."""
    y_true = np.array([0, 0, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 2])

    result = clf_eval.evaluate(y_true, y_pred, class_labels=["A", "B", "C"])

    assert "per_class_metrics" in result
    pcm = result["per_class_metrics"]
    assert len(pcm) == 3

    # support must equal class count in y_true
    assert pcm[0]["support"] == 2  # class A
    assert pcm[1]["support"] == 3  # class B
    assert pcm[2]["support"] == 2  # class C

    # class_labels order matches per_class_metrics order
    assert result["class_labels"] == ["A", "B", "C"]
    assert [m["class"] for m in pcm] == ["A", "B", "C"]

    # All NumPy scalars must be plain Python
    for m in pcm:
        assert isinstance(m["precision"], float)
        assert isinstance(m["recall"], float)
        assert isinstance(m["f1"], float)
        assert isinstance(m["support"], int)
        assert isinstance(m["class"], (str, int, float))


def test_per_class_metrics_and_confusion_matrix_order_consistent():
    """Confusion matrix rows/columns and per_class_metrics share class order."""
    y_true = np.array([2, 0, 1, 0, 2, 1, 1])
    y_pred = np.array([2, 0, 1, 0, 1, 2, 1])

    result = clf_eval.evaluate(y_true, y_pred, class_labels=["zero", "one", "two"])

    assert result["class_labels"] == ["zero", "one", "two"]
    assert [m["class"] for m in result["per_class_metrics"]] == ["zero", "one", "two"]
    assert len(result["confusion_matrix"]) == 3
    assert len(result["confusion_matrix"][0]) == 3


def test_per_class_metrics_zero_division_stable():
    """Zero precision/recall for a class never predicted returns 0.0."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])  # everything predicted as 0

    result = clf_eval.evaluate(y_true, y_pred)
    pcm = result["per_class_metrics"]

    # class 1 precision = 0 (never predicted), recall = 0 (never found)
    class_1 = [m for m in pcm if m["class"] == 1][0]
    assert class_1["precision"] == 0.0
    assert class_1["recall"] == 0.0
    assert class_1["f1"] == 0.0


# --- KNN predict_proba ------------------------------------------------------


def test_knn_predict_proba_shape_and_rowsum():
    iris = load_iris()
    knn = KNNModel(n_neighbors=3)
    knn.fit(iris.data, iris.target)

    proba = knn.predict_proba(iris.data)
    assert proba.shape == (len(iris.data), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_knn_predict_proba_columns_follow_classes():
    iris = load_iris()
    knn = KNNModel(n_neighbors=5)
    knn.fit(iris.data, iris.target)

    proba = knn.predict_proba(iris.data)
    # argmax column must match classes_[idx]
    predicted = knn.classes_[np.argmax(proba, axis=1)]
    assert np.array_equal(predicted, knn.predict(iris.data))


def test_knn_predict_proba_unfitted_raises():
    knn = KNNModel()
    with pytest.raises(ValueError, match="not fitted"):
        knn.predict_proba(np.array([[1.0, 2.0, 3.0, 4.0]]))


# --- GaussianNB predict_proba + theta_/var_ --------------------------------


def test_gnb_predict_proba_shape_and_rowsum():
    iris = load_iris()
    gnb = GaussianNBModel()
    gnb.fit(iris.data, iris.target)

    proba = gnb.predict_proba(iris.data)
    assert proba.shape == (len(iris.data), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gnb_predict_proba_columns_follow_classes():
    iris = load_iris()
    gnb = GaussianNBModel()
    gnb.fit(iris.data, iris.target)

    proba = gnb.predict_proba(iris.data)
    predicted = gnb.classes_[np.argmax(proba, axis=1)]
    assert np.array_equal(predicted, gnb.predict(iris.data))


def test_gnb_theta_var_shapes():
    iris = load_iris()
    gnb = GaussianNBModel()
    gnb.fit(iris.data, iris.target)

    assert gnb.theta_.shape == (3, 4)  # (n_classes, n_features)
    assert gnb.var_.shape == (3, 4)
    # theta_ is the per-class mean of each feature
    assert np.isfinite(gnb.theta_).all()
    assert np.isfinite(gnb.var_).all()
    assert (gnb.var_ > 0).all()


def test_gnb_unfitted_raises_on_proba_and_stats():
    gnb = GaussianNBModel()
    with pytest.raises(ValueError, match="not fitted"):
        gnb.predict_proba(np.array([[1.0, 2.0, 3.0, 4.0]]))
    with pytest.raises(ValueError, match="not fitted"):
        _ = gnb.theta_
    with pytest.raises(ValueError, match="not fitted"):
        _ = gnb.var_


# --- ml_bridge.get_predict_scores ------------------------------------------


def _make_iris():
    iris = load_iris()
    return iris.data, iris.target


def test_scores_dict_structure():
    X, y = _make_iris()
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X, y)

    result = ml_bridge.get_predict_scores(lr, X)
    assert result is not None
    assert set(result.keys()) == {"y_score", "score_type", "score_classes"}
    assert result["score_type"] == "probability"
    assert isinstance(result["y_score"], list)
    assert len(result["y_score"]) == len(y)
    assert all(len(row) == 3 for row in result["y_score"])
    assert result["score_classes"] == list(lr.classes_)


def test_scores_decision_type_linear_svm():
    from ml_engine.models.linear_svm import LinearSVM

    X, y = _make_iris()
    svm = LinearSVM()
    svm.fit(X, y)

    result = ml_bridge.get_predict_scores(svm, X)
    assert result is not None
    assert result["score_type"] == "decision"
    # LinearSVM is multiclass OVR → scores are (n_samples, n_classes)
    assert result["score_classes"] is not None


def test_scores_decision_type_binary_one_d_converted():
    """Binary decision_function 1-D output must become two columns."""
    from sklearn.svm import LinearSVC

    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([0, 0, 1, 1, 1])
    svc = LinearSVC()
    svc.fit(X, y)

    result = ml_bridge.get_predict_scores(svc, X)
    assert result is not None
    assert result["score_type"] == "decision"
    # Must be 2D with 2 columns
    assert all(len(row) == 2 for row in result["y_score"])
    # argmax should match predicted class
    predicted = svc.classes_[np.argmax(np.array(result["y_score"]), axis=1)]
    assert np.array_equal(predicted, svc.predict(X))


def test_scores_none_when_no_proba_or_decision():
    class NoScoreModel:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    model = NoScoreModel()
    result = ml_bridge.get_predict_scores(model, np.array([[1, 2]]))
    assert result is None


def test_scores_non_finite_raises():
    class BadScoreModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[np.nan, np.inf]] * len(X))

    model = BadScoreModel()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        ml_bridge.get_predict_scores(model, np.array([[1.0, 2.0]]))


def test_scores_all_models_have_probability_or_decision():
    """Every currently registered classification model must yield a score dict."""
    X, y = _make_iris()

    from ml_engine.models.knn import KNNModel
    from ml_engine.models.linear_svm import LinearSVM
    from ml_engine.models.logistic_regression import LogisticRegressionGD
    from ml_engine.models.mlp_classifier import MLPClassifier
    from ml_engine.models.naive_bayes import GaussianNBModel

    models = [
        ("knn", KNNModel(n_neighbors=3)),
        ("gnb", GaussianNBModel()),
        ("logistic", LogisticRegressionGD(learning_rate=0.05, max_iter=500)),
        ("svm", LinearSVM(learning_rate=0.05, max_iter=500)),
        ("mlp", MLPClassifier(max_iter=200, random_state=7)),
    ]
    for name, m in models:
        m.fit(X, y)
        result = ml_bridge.get_predict_scores(m, X)
        assert result is not None, f"{name} returned None"
        assert result["score_type"] in ("probability", "decision"), name
        assert result["score_classes"] is not None, name


# --- ml_bridge.get_training_info extensions --------------------------------


def test_training_info_accepts_feature_names_and_fit_duration():
    X, y = _make_iris()
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X, y)

    info = ml_bridge.get_training_info(
        lr, feature_names=["f0", "f1", "f2", "f3"], fit_duration_ms=1234.56
    )
    assert info["feature_names"] == ["f0", "f1", "f2", "f3"]
    assert info["fit_duration_ms"] == 1234.56


def test_training_info_feature_names_optional():
    X, y = _make_iris()
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X, y)

    info = ml_bridge.get_training_info(lr)
    assert "feature_names" not in info
    assert "fit_duration_ms" not in info


# --- experiment_service pipeline (integration) ----------------------------


def _fake_dataset():
    return SimpleNamespace(
        X=np.tile(np.arange(8, dtype=float).reshape(4, 2), (10, 1)),
        y=np.array([0, 1] * 20),
        task_type="classification",
        feature_names=["f0", "f1"],
        target_names=["class_0", "class_1"],
    )


def _fake_split(X, y, test_size, random_state=42, task_type="classification", standardize=True):
    n = len(y)
    n_test = max(1, int(round(n * test_size)))
    return X[n_test:], X[:n_test], y[n_test:], y[:n_test]


def _fake_evaluate(task_type, y_true, y_pred, y_score=None, class_labels=None):
    result = {
        "metrics": {"accuracy": 1.0},
        "confusion_matrix": [[1, 0], [0, 1]] if task_type == "classification" else None,
        "class_labels": class_labels if task_type == "classification" else None,
    }
    if task_type == "classification":
        result["per_class_metrics"] = [
            {"class": "class_0", "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
            {"class": "class_1", "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
        ]
    return result


def _fake_model_detail(task_type: str):
    def fake(name: str) -> dict:
        return {"name": name, "description": "", "task_type": task_type, "default_parameters": {}}
    return fake


class _FakeClassifier:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class _FakeNoScore:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _FakeRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_classification_predictions_stores_all_new_fields(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(ml_bridge, "create_estimator",
                        lambda name, params, random_state=None: _FakeClassifier())
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post("/api/v1/experiments",
                          json={"dataset_name": "iris", "model_name": "logistic_regression"}).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    evaluation = detail["evaluation"]
    assert evaluation["y_true"] is not None
    assert evaluation["y_pred"] is not None
    assert evaluation["y_score"] is not None
    assert evaluation["score_type"] == "probability"
    assert evaluation["score_classes"] is not None
    assert evaluation["per_class_metrics"] is not None
    assert len(evaluation["per_class_metrics"]) == 2


def test_classification_without_score_model_predictions_have_null_score(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(ml_bridge, "create_estimator",
                        lambda name, params, random_state=None: _FakeNoScore())
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post("/api/v1/experiments",
                          json={"dataset_name": "iris", "model_name": "logistic_regression"}).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    ev = detail["evaluation"]
    assert ev["y_true"] is not None
    assert ev["y_pred"] is not None
    assert ev["y_score"] is None
    assert ev["score_type"] is None
    assert ev["score_classes"] is None


def test_regression_predictions_shape_unchanged(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: SimpleNamespace(
        X=np.arange(40, dtype=float).reshape(20, 2), y=np.arange(40, dtype=float),
        task_type="regression", feature_names=["f0", "f1"], target_names=[],
    ))
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("regression"))
    monkeypatch.setattr(ml_bridge, "create_estimator",
                        lambda name, params, random_state=None: _FakeRegressor())
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate",
                        lambda *a, **kw: {"metrics": {"r2": 0.0},
                                          "confusion_matrix": None, "class_labels": None})

    created = client.post(
        "/api/v1/experiments",
        json={"dataset_name": "diabetes", "model_name": "linear_regression"},
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    ev = detail["evaluation"]
    assert ev["y_true"] is not None
    assert ev["y_pred"] is not None
    assert ev["y_score"] is None
    assert ev["per_class_metrics"] is None


def test_training_info_contains_feature_names_and_fit_duration(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(ml_bridge, "create_estimator",
                        lambda name, params, random_state=None: _FakeClassifier())
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post("/api/v1/experiments",
                          json={"dataset_name": "iris", "model_name": "logistic_regression"}).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    ti = detail["training_info"]
    assert ti["feature_names"] == ["f0", "f1"]
    assert isinstance(ti["fit_duration_ms"], float)
    assert ti["fit_duration_ms"] >= 0


def test_evaluation_new_fields_json_serializable(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(ml_bridge, "create_estimator",
                        lambda name, params, random_state=None: _FakeClassifier())
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post("/api/v1/experiments",
                          json={"dataset_name": "iris", "model_name": "logistic_regression"}).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    # Round-trip through JSON should succeed
    json_str = json.dumps(detail, default=str)
    parsed = json.loads(json_str)
    ev = parsed["evaluation"]
    assert "y_score" in ev
    assert "score_type" in ev
    assert "per_class_metrics" in ev


def test_old_experiment_schema_parses_without_new_fields(client, monkeypatch):
    """EvaluationResult must still parse from DB rows that lack new fields."""
    from backend.api.schemas import EvaluationResult

    old = EvaluationResult(
        metrics={"accuracy": 0.8},
        confusion_matrix=[[1, 2], [3, 4]],
        class_labels=["A", "B"],
    )
    assert old.y_true is None
    assert old.y_pred is None
    assert old.y_score is None
    assert old.score_type is None
    assert old.score_classes is None
    assert old.per_class_metrics is None


def test_per_class_metrics_serializes_with_class_key():
    """PerClassMetrics must round-trip with the JSON key 'class'."""
    from backend.api.schemas import PerClassMetrics

    pcm = PerClassMetrics.model_validate(
        {"class": "iris-setosa", "precision": 0.9, "recall": 0.8,
         "f1": 0.85, "support": 50}
    )
    dumped = pcm.model_dump()
    assert "class" in dumped
    assert "class_value" not in dumped
    assert dumped["class"] == "iris-setosa"

    json_str = pcm.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["class"] == "iris-setosa"


def test_evaluation_block_round_trips_class_key(client, monkeypatch):
    """Full API response must have 'class' (not 'class_value') inside per_class_metrics."""
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(ml_bridge, "create_estimator",
                        lambda name, params, random_state=None: _FakeClassifier())
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post("/api/v1/experiments",
                          json={"dataset_name": "iris", "model_name": "logistic_regression"}).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    pcm_list = detail["evaluation"]["per_class_metrics"]
    assert len(pcm_list) == 2
    for item in pcm_list:
        assert "class" in item, f"Expected 'class' key, got keys: {item.keys()}"
        assert "class_value" not in item
        assert isinstance(item["class"], str)
