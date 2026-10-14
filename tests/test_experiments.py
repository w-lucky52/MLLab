from types import SimpleNamespace

import numpy as np

from backend.services import ml_bridge

PAYLOAD = {"dataset_name": "iris", "model_name": "logistic_regression"}


class FakeEstimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _fake_dataset() -> SimpleNamespace:
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
        "confusion_matrix": [[1, 0], [0, 1]] if task_type == "classification" else None,
        "class_labels": class_labels if task_type == "classification" else None,
    }


def _fake_model_detail(task_type: str):
    def fake(name: str) -> dict:
        return {"name": name, "description": "", "task_type": task_type, "default_parameters": {}}

    return fake


def test_create_experiment_returns_pending(client):
    response = client.post("/api/v1/experiments", json=PAYLOAD)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["name"] == "logistic_regression on iris"
    assert body["test_size"] == 0.3
    assert body["task_type"] is None


def test_create_experiment_rejects_invalid_test_size(client):
    response = client.post("/api/v1/experiments", json={**PAYLOAD, "test_size": 5})

    assert response.status_code == 422


def test_create_experiment_accepts_three_splits(client):
    for split in (0.2, 0.3, 0.4):
        response = client.post("/api/v1/experiments", json={**PAYLOAD, "test_size": split})
        assert response.status_code == 201
        assert response.json()["test_size"] == split


def test_list_experiments_ordered_newest_first(client):
    client.post("/api/v1/experiments", json={**PAYLOAD, "name": "first"})
    client.post("/api/v1/experiments", json={**PAYLOAD, "name": "second"})

    response = client.get("/api/v1/experiments")

    assert response.status_code == 200
    assert [experiment["name"] for experiment in response.json()] == ["second", "first"]


def test_get_experiment_not_found(client):
    assert client.get("/api/v1/experiments/999").status_code == 404


def test_experiment_fails_on_unknown_model(client):
    # iris is a real registered dataset, but the model name is not in the
    # catalog; the pipeline must fail with a clear model error.
    created = client.post(
        "/api/v1/experiments",
        json={**PAYLOAD, "model_name": "no_such_model"},
    ).json()

    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    assert detail["status"] == "failed"
    assert "Unknown model" in detail["error"]


def test_experiment_completes_with_fake_ml_engine(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(
        ml_bridge, "create_estimator",
        lambda name, params, random_state=None: FakeEstimator(),
    )
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post("/api/v1/experiments", json=PAYLOAD).json()

    detail = client.get(f"/api/v1/experiments/{created['id']}").json()
    assert detail["status"] == "completed"
    assert detail["task_type"] == "classification"
    assert detail["metrics"] == {"accuracy": 1.0}
    assert detail["confusion_matrix"] == [[1, 0], [0, 1]]
    assert detail["class_labels"] == ["class_0", "class_1"]
    assert detail["training_info"] == {}
    assert detail["charts"] == []


def test_completed_classification_exposure_evaluation_block(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(
        ml_bridge, "create_estimator",
        lambda name, params, random_state=None: FakeEstimator(),
    )
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post("/api/v1/experiments", json=PAYLOAD).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    evaluation = detail["evaluation"]
    assert evaluation["metrics"] == {"accuracy": 1.0}
    assert evaluation["confusion_matrix"] == [[1, 0], [0, 1]]
    assert evaluation["class_labels"] == ["class_0", "class_1"]
    assert evaluation["y_true"] is None
    assert evaluation["y_pred"] is None
    # Raw predictions are internal only; never serialized at top level.
    assert "predictions" not in detail


def _fake_regression_dataset() -> SimpleNamespace:
    return SimpleNamespace(
        X=np.tile(np.arange(8, dtype=float).reshape(4, 2), (10, 1)),
        y=np.arange(40, dtype=float),
        task_type="regression",
        feature_names=["f0", "f1"],
        target_names=[],
    )


def test_completed_regression_exposure_evaluation_block(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_regression_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("regression"))
    monkeypatch.setattr(
        ml_bridge, "create_estimator",
        lambda name, params, random_state=None: FakeEstimator(),
    )
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    payload = {"dataset_name": "diabetes", "model_name": "linear_regression"}
    created = client.post("/api/v1/experiments", json=payload).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()

    assert detail["status"] == "completed"
    assert detail["task_type"] == "regression"
    evaluation = detail["evaluation"]
    assert evaluation["metrics"] == {"accuracy": 1.0}
    assert evaluation["confusion_matrix"] is None
    assert evaluation["class_labels"] is None
    assert len(evaluation["y_true"]) == len(evaluation["y_pred"])
    assert len(evaluation["y_true"]) > 0


def test_pending_experiment_evaluation_is_null(client):
    created = client.post("/api/v1/experiments", json=PAYLOAD).json()
    assert created["evaluation"] is None


def test_experiment_fails_on_task_type_mismatch(client, monkeypatch):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("regression"))

    created = client.post("/api/v1/experiments", json=PAYLOAD).json()

    detail = client.get(f"/api/v1/experiments/{created['id']}").json()
    assert detail["status"] == "failed"
    assert "supports" in detail["error"]


def test_delete_experiment_removes_artifacts(client, monkeypatch, tmp_path):
    monkeypatch.setattr(ml_bridge, "load_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr(ml_bridge, "model_detail", _fake_model_detail("classification"))
    monkeypatch.setattr(
        ml_bridge, "create_estimator",
        lambda name, params, random_state=None: FakeEstimator(),
    )
    monkeypatch.setattr(ml_bridge, "split_dataset", _fake_split)
    monkeypatch.setattr(ml_bridge, "evaluate", _fake_evaluate)

    created = client.post("/api/v1/experiments", json=PAYLOAD).json()

    detail = client.get(f"/api/v1/experiments/{created['id']}").json()
    assert detail["status"] == "completed"
    assert detail["task_type"] == "classification"
    assert detail["metrics"] == {"accuracy": 1.0}
    artifact_dir = tmp_path / "experiments" / str(created["id"])
    assert detail["artifact_dir"] is not None
    assert (artifact_dir / "summary.json").exists()

    assert client.delete(f"/api/v1/experiments/{created['id']}").status_code == 204
    assert client.get(f"/api/v1/experiments/{created['id']}").status_code == 404
    assert not artifact_dir.exists()


def test_delete_experiment_not_found(client):
    assert client.delete("/api/v1/experiments/999").status_code == 404
