from backend.services import ml_bridge
from backend.services.ml_bridge import MlEngineUnavailable

# Models whose branches have been merged into develop (A: PR #2, B: PR #3 /
# #8 / #10 / #11, C: feature/tree_ensemble).
EXPECTED_MODELS = {
    "knn_classifier": "classification",
    "gaussian_nb": "classification",
    "linear_regression": "regression",
    "ridge_regression": "regression",
    "logistic_regression": "classification",
    "linear_svm": "classification",
    "mlp_classifier": "classification",
    "decision_tree_regressor": "regression",
    "random_forest_regressor": "regression",
    "gbdt_regressor": "regression",
}


def test_list_models_returns_registered_catalog(client):
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    catalog = {m["name"]: m["task_type"] for m in response.json()}
    assert catalog == EXPECTED_MODELS


def test_list_models_filters_by_task_type(client):
    classification = client.get("/api/v1/models", params={"task_type": "classification"})
    regression = client.get("/api/v1/models", params={"task_type": "regression"})

    assert {m["name"] for m in classification.json()} == {
        "knn_classifier",
        "gaussian_nb",
        "logistic_regression",
        "linear_svm",
        "mlp_classifier",
    }
    assert {m["name"] for m in regression.json()} == {
        "linear_regression",
        "ridge_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
        "gbdt_regressor",
    }


def test_registered_models_have_consistent_parameter_specs(client):
    for name in EXPECTED_MODELS:
        detail = client.get(f"/api/v1/models/{name}")
        assert detail.status_code == 200
        body = detail.json()
        params = body["parameters"]
        assert params  # every registered model declares at least one param
        assert body["default_parameters"] == {
            p["name"]: p["default"] for p in params
        }
        for spec in params:
            assert spec["type"] in {"int", "float", "bool", "select"}


def test_get_unknown_model_returns_404(client):
    response = client.get("/api/v1/models/no_such_model")

    assert response.status_code == 404


def test_list_models_unavailable_returns_503(client, monkeypatch):
    def raise_unavailable(*_args, **_kwargs):
        raise MlEngineUnavailable("ml_engine not installed")

    monkeypatch.setattr(ml_bridge, "list_models", raise_unavailable)

    response = client.get("/api/v1/models")

    assert response.status_code == 503


def test_get_model_unavailable_returns_503(client, monkeypatch):
    def raise_unavailable(*_args, **_kwargs):
        raise MlEngineUnavailable("ml_engine not installed")

    monkeypatch.setattr(ml_bridge, "model_detail", raise_unavailable)

    response = client.get("/api/v1/models/logistic_regression")

    assert response.status_code == 503


def test_list_models_with_filter(client, monkeypatch):
    models = [
        {
            "name": "logistic_regression",
            "description": "d",
            "task_type": "classification",
            "default_parameters": {},
            "parameters": [],
        },
        {
            "name": "ridge_regression",
            "description": "d",
            "task_type": "regression",
            "default_parameters": {"alpha": 1.0},
            "parameters": [],
        },
    ]
    captured: dict = {}

    def fake_list_models(task_type):
        captured["task_type"] = task_type
        return [m for m in models if task_type is None or m["task_type"] == task_type]

    monkeypatch.setattr(ml_bridge, "list_models", fake_list_models)

    assert client.get("/api/v1/models").json() == models
    assert client.get("/api/v1/models", params={"task_type": "regression"}).json() == [models[1]]
    assert captured["task_type"] == "regression"


def test_get_model_not_found(client, monkeypatch):
    def raise_key_error(name):
        raise KeyError(name)

    monkeypatch.setattr(ml_bridge, "model_detail", raise_key_error)

    response = client.get("/api/v1/models/unknown_model")

    assert response.status_code == 404
    assert "unknown_model" in response.json()["detail"]
