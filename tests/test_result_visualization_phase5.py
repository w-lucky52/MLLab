"""Phase 5 — friendly class display names for classification results.

``score_class_names`` is a display-only companion to ``score_classes``. It is
written only when every class value can be provably mapped to a non-empty
``target_names`` entry; otherwise it stays ``None`` and the frontend keeps
rendering the raw class values of older experiments.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

from backend.api.schemas import EvaluationResult
from backend.services import ml_bridge
from backend.services.experiment_service import _score_class_names

FRONTEND_TESTS = Path(__file__).parent / "frontend" / "class_display.test.mjs"


def _names(dataset) -> list[str]:
    return [str(name) for name in dataset.target_names]


# --- per-dataset mappings ---------------------------------------------------


def test_iris_classes_map_to_target_names():
    names = _names(load_iris())

    assert ml_bridge.resolve_score_class_names([0, 1, 2], names) == names


def test_wine_mapping_count_and_order():
    names = _names(load_wine())

    mapped = ml_bridge.resolve_score_class_names([0, 1, 2], names)

    assert mapped == names
    assert len(mapped) == 3


def test_breast_cancer_binary_names():
    names = _names(load_breast_cancer())

    assert ml_bridge.resolve_score_class_names([0, 1], names) == names
    assert len(names) == 2


def test_digits_maps_all_ten_classes():
    names = _names(load_digits())
    classes = list(range(10))

    mapped = ml_bridge.resolve_score_class_names(classes, names)

    assert mapped == [str(index) for index in range(10)]
    assert mapped == names


def test_reordered_classes_follow_target_names_order():
    names = _names(load_iris())
    classes = [2, 0, 1]
    original = list(classes)

    mapped = ml_bridge.resolve_score_class_names(classes, names)

    assert mapped == [names[2], names[0], names[1]]
    # The input list itself is never modified.
    assert classes == original


def test_non_contiguous_classes_out_of_range_are_rejected():
    names = _names(load_iris())

    assert ml_bridge.resolve_score_class_names([1, 3, 5], names) is None


def test_negative_and_positive_classes_are_not_mapped():
    names = _names(load_breast_cancer())

    assert ml_bridge.resolve_score_class_names([-1, 1], names) is None


def test_string_classes_are_not_mapped_by_position():
    names = _names(load_iris())

    assert ml_bridge.resolve_score_class_names(["setosa", "versicolor"], names) is None


def test_nan_and_inf_classes_are_rejected():
    names = _names(load_iris())

    assert ml_bridge.resolve_score_class_names([0.0, float("nan")], names) is None
    assert ml_bridge.resolve_score_class_names([0.0, float("inf")], names) is None


def test_empty_target_name_rejects_whole_mapping():
    names = ["setosa", "", "virginica"]

    assert ml_bridge.resolve_score_class_names([0, 1, 2], names) is None


def test_regression_never_gets_score_class_names():
    score_info = {"score_classes": [0, 1], "score_type": "probability"}

    assert _score_class_names("regression", score_info, ["a", "b"]) is None
    assert _score_class_names("classification", None, ["a", "b"]) is None


def test_old_experiment_evaluation_parses_without_names():
    old = EvaluationResult(
        metrics={"accuracy": 0.8},
        confusion_matrix=[[1, 2], [3, 4]],
        class_labels=["A", "B"],
        score_classes=[0, 1],
    )

    assert old.score_class_names is None


# --- experiment pipeline integration ---------------------------------------


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
    return {
        "metrics": {"accuracy": 1.0},
        "confusion_matrix": [[1, 0], [0, 1]],
        "class_labels": class_labels,
        "per_class_metrics": [
            {"class": "class_0", "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
            {"class": "class_1", "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
        ],
    }


def _fake_model_detail(task_type: str):
    def fake(name: str) -> dict:
        return {"name": name, "description": "", "task_type": task_type, "default_parameters": {}}

    return fake


class _FakeClassifier:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class _FakeRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_pipeline_writes_display_names_without_touching_score_classes(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(
        ml_bridge,
        "create_estimator",
        lambda name, params, random_state=None: _FakeClassifier(),
    )
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post(
        "/api/v1/experiments", json={"dataset_name": "iris", "model_name": "logistic_regression"}
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    evaluation = detail["evaluation"]
    # ``score_classes`` keeps its pre-existing serialized form (raw values
    # were stringified by ``_jsonable`` long before this phase); the new
    # display names must align with it without changing it.
    assert evaluation["score_classes"] == ["0", "1"]
    assert evaluation["score_class_names"] == ["class_0", "class_1"]
    assert json.loads(json.dumps(evaluation))["score_class_names"] == ["class_0", "class_1"]


def test_regression_pipeline_has_no_display_names(client, monkeypatch):
    monkeypatch.setattr(
        ml_bridge,
        "load_dataset",
        lambda name: SimpleNamespace(
            X=np.arange(40, dtype=float).reshape(20, 2),
            y=np.arange(40, dtype=float),
            task_type="regression",
            feature_names=["f0", "f1"],
            target_names=["target"],
        ),
    )
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("regression"))
    monkeypatch.setattr(
        ml_bridge,
        "create_estimator",
        lambda name, params, random_state=None: _FakeRegressor(),
    )
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(
        ml_bridge,
        "evaluate",
        lambda *args, **kwargs: {
            "metrics": {"r2": 0.0},
            "confusion_matrix": None,
            "class_labels": None,
        },
    )

    created = client.post(
        "/api/v1/experiments", json={"dataset_name": "diabetes", "model_name": "linear_regression"}
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    assert detail["evaluation"]["score_class_names"] is None


# --- frontend pure-function contract --------------------------------------


def test_frontend_class_display_contracts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    result = subprocess.run(
        [node, "--test", str(FRONTEND_TESTS)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
