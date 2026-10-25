"""Phase 7B2A — real model explanation data (``training_info.explanation_data``).

Every payload below is built from a fitted estimator and the real
train/test arrays; the tests verify the mathematical reconstructions
(contributions, likelihoods, tree paths), the KNN neighbour contract, the
MLP permutation-importance contract and the end-to-end storage contract
(summary.json == API, no ``chart_count`` in the completion log).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.api.schemas import TrainingInfo
from backend.core.config import get_settings
from backend.services import ml_bridge
from backend.services.model_explanation import (
    MAX_SAMPLES,
    ExplanationError,
    build_explanation_data,
)
from ml_engine.models.knn import KNNModel

ROOT = Path(__file__).resolve().parents[1]
_APPROX = {"rel": 1e-6, "abs": 1e-6}


@dataclass
class _Prepared:
    data: Any
    estimator: Any
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray


_CACHE: dict[tuple, _Prepared] = {}


def _prepare(
    model_name: str,
    dataset_name: str,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> _Prepared:
    key = (
        model_name,
        dataset_name,
        json.dumps(hyperparameters or {}, sort_keys=True),
        random_state,
    )
    if key in _CACHE:
        return _CACHE[key]
    data = ml_bridge.load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = ml_bridge.split_dataset(
        data.X,
        data.y,
        test_size=0.3,
        random_state=random_state,
        task_type=data.task_type,
        standardize=True,
    )
    estimator = ml_bridge.create_estimator(
        model_name, hyperparameters, random_state=random_state
    )
    estimator.fit(X_train, y_train)
    prepared = _Prepared(
        data=data,
        estimator=estimator,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=estimator.predict(X_test),
    )
    _CACHE[key] = prepared
    return prepared


def _explain(prepared: _Prepared, model_name: str, random_state: int = 42) -> dict | None:
    return build_explanation_data(
        model_name=model_name,
        estimator=prepared.estimator,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        y_pred=prepared.y_pred,
        feature_names=list(prepared.data.feature_names),
        random_state=random_state,
    )


def _iter_floats(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_floats(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_floats(item)
    elif isinstance(value, float):
        yield value


# --- common contract ----------------------------------------------------------


def test_linear_payload_common_contract():
    prepared = _prepare("linear_regression", "diabetes", {"max_iter": 50})
    payload = _explain(prepared, "linear_regression")

    assert payload["schema_version"] == 1
    assert payload["kind"] == "linear_contribution"
    assert payload["feature_space"] == "standardized"
    assert payload["feature_names"] == list(prepared.data.feature_names)
    assert payload["sample_count"] == len(payload["samples"]) == MAX_SAMPLES
    assert [s["sample_index"] for s in payload["samples"]] == list(
        range(1, MAX_SAMPLES + 1)
    )


def test_only_first_twenty_test_samples_are_explained():
    prepared = _prepare("logistic_regression", "iris", {"max_iter": 30})
    assert len(prepared.X_test) == 45
    payload = _explain(prepared, "logistic_regression")
    assert payload["sample_count"] == 20
    assert len(payload["samples"]) == 20
    assert [s["sample_index"] for s in payload["samples"]] == list(range(1, 21))


# --- 1/2. linear & ridge contributions ---------------------------------------


@pytest.mark.parametrize(
    "model_name", ["linear_regression", "ridge_regression"]
)
def test_regression_contributions_reconstruct_prediction(model_name):
    prepared = _prepare(model_name, "diabetes", {"max_iter": 50})
    payload = _explain(prepared, model_name)

    coef = np.asarray(prepared.estimator.coef_, dtype=float).ravel()
    intercept = float(prepared.estimator.intercept_)
    assert payload["kind"] == "linear_contribution"

    for position, sample in enumerate(payload["samples"]):
        features = prepared.X_test[position]
        assert np.allclose(sample["feature_values"], features)
        # Signed contributions, never absolute values.
        assert np.allclose(sample["contributions"], features * coef)
        assert sum(sample["contributions"]) + sample["intercept"] == pytest.approx(
            sample["reconstructed_prediction"], **_APPROX
        )
        assert sample["reconstructed_prediction"] == pytest.approx(
            float(prepared.y_pred[position]), **_APPROX
        )
        assert sample["intercept"] == pytest.approx(intercept, **_APPROX)
        assert sample["y_pred"] == pytest.approx(float(prepared.y_pred[position]), **_APPROX)
        assert sample["y_true"] == pytest.approx(float(prepared.y_test[position]), **_APPROX)


# --- 3/4. multiclass linear scores ------------------------------------------


def test_logistic_predicted_class_score_reconstructs():
    prepared = _prepare("logistic_regression", "iris", {"max_iter": 100})
    payload = _explain(prepared, "logistic_regression")

    coef = np.asarray(prepared.estimator.coef_, dtype=float)
    intercept = np.asarray(prepared.estimator.intercept_, dtype=float).ravel()
    classes = np.asarray(prepared.estimator.classes_)

    assert payload["kind"] == "multiclass_linear_contribution"
    assert payload["score_name"] == "logit"
    assert payload["classes"] == [int(c) for c in classes]

    for position, sample in enumerate(payload["samples"]):
        class_index = sample["class_index"]
        assert classes[class_index] == sample["predicted_class"]
        assert sample["predicted_class"] == int(prepared.y_pred[position])
        features = prepared.X_test[position]
        assert np.allclose(sample["contributions"], features * coef[class_index])
        assert sum(sample["contributions"]) + sample["intercept"] == pytest.approx(
            sample["reconstructed_score"], **_APPROX
        )
        logits = prepared.X_test[position] @ coef.T + intercept
        assert sample["reconstructed_score"] == pytest.approx(
            float(logits[class_index]), **_APPROX
        )
        # A softmax logit's argmax is exactly the predicted class.
        assert int(np.argmax(logits)) == class_index


def test_svm_predicted_class_decision_score_reconstructs():
    prepared = _prepare("linear_svm", "iris", {"max_iter": 100})
    payload = _explain(prepared, "linear_svm")

    coef = np.asarray(prepared.estimator.coef_, dtype=float)
    scores = np.asarray(prepared.estimator.decision_function(prepared.X_test))

    assert payload["kind"] == "multiclass_linear_contribution"
    assert payload["score_name"] == "decision_score"

    for position, sample in enumerate(payload["samples"]):
        class_index = sample["class_index"]
        features = prepared.X_test[position]
        assert np.allclose(sample["contributions"], features * coef[class_index])
        assert sum(sample["contributions"]) + sample["intercept"] == pytest.approx(
            sample["reconstructed_score"], **_APPROX
        )
        assert sample["reconstructed_score"] == pytest.approx(
            float(scores[position, class_index]), **_APPROX
        )


# --- 5-8. KNN neighbours -----------------------------------------------------


def test_knn_uniform_neighbours_come_from_real_training_rows():
    prepared = _prepare("knn_classifier", "iris", None)
    payload = _explain(prepared, "knn_classifier")

    assert payload["kind"] == "knn_neighbors"
    assert payload["n_neighbors"] == 5
    assert payload["weights"] == "uniform"
    assert payload["sample_count"] == MAX_SAMPLES

    for position, sample in enumerate(payload["samples"]):
        neighbours = sample["neighbors"]
        assert len(neighbours) == 5
        assert [n["rank"] for n in neighbours] == [1, 2, 3, 4, 5]
        distances = [n["distance"] for n in neighbours]
        assert distances == sorted(distances)
        for neighbour in neighbours:
            train_index = neighbour["train_index"]
            assert 0 <= train_index < len(prepared.X_train)
            assert neighbour["label"] == int(prepared.y_train[train_index])
            assert neighbour["vote_weight"] == 1.0

        totals = sample["vote_totals"]
        assert len(totals) == len(np.unique(prepared.y_train))
        assert sum(item["weight"] for item in totals) == pytest.approx(5.0)
        top = max(item["weight"] for item in totals)
        top_classes = [item["class"] for item in totals if item["weight"] == top]
        assert int(prepared.y_pred[position]) in top_classes
        assert sample["y_pred"] == int(prepared.y_pred[position])


def test_knn_distance_weights_follow_inverse_distance():
    prepared = _prepare("knn_classifier", "iris", {"weights": "distance"})
    payload = _explain(prepared, "knn_classifier")

    assert payload["weights"] == "distance"
    for position, sample in enumerate(payload["samples"]):
        for neighbour in sample["neighbors"]:
            distance = neighbour["distance"]
            assert distance > 0.0  # standardized iris has no zero distances
            assert neighbour["vote_weight"] * distance == pytest.approx(1.0, **_APPROX)
        totals = sample["vote_totals"]
        top = max(item["weight"] for item in totals)
        top_classes = [item["class"] for item in totals if item["weight"] == top]
        assert int(prepared.y_pred[position]) in top_classes


def test_knn_zero_distance_rule_matches_sklearn():
    X_train = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.0, 0.0]])  # exact duplicate of training row 0
    y_test = np.array([0])

    estimator = KNNModel(n_neighbors=3, weights="distance").fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    assert y_pred[0] == 0

    payload = build_explanation_data(
        model_name="knn_classifier",
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        feature_names=["f1", "f2"],
        random_state=42,
    )

    sample = payload["samples"][0]
    assert sample["neighbors"][0]["distance"] == 0.0
    assert sample["neighbors"][0]["vote_weight"] == 1.0
    assert all(n["vote_weight"] == 0.0 for n in sample["neighbors"][1:])
    totals = {item["class"]: item["weight"] for item in sample["vote_totals"]}
    assert totals == {0: 1.0, 1: 0.0}


# --- 9/10. GaussianNB likelihoods -------------------------------------------


def test_gaussian_nb_probabilities_matrix_is_real():
    prepared = _prepare("gaussian_nb", "iris", None)
    payload = _explain(prepared, "gaussian_nb")
    probabilities = np.asarray(prepared.estimator.predict_proba(prepared.X_test[:20]))
    classes = np.asarray(prepared.estimator.classes_)

    assert payload["kind"] == "gaussian_nb_likelihood"
    assert payload["classes"] == [int(c) for c in classes]

    for position, sample in enumerate(payload["samples"]):
        assert sum(
            item["probability"] for item in sample["class_probabilities"]
        ) == pytest.approx(1.0, abs=1e-9)
        class_index = int(
            np.flatnonzero(classes == sample["predicted_class"])[0]
        )
        assert classes[class_index] == sample["predicted_class"]
        assert sample["predicted_class"] == int(prepared.y_pred[position])
        for column, item in enumerate(sample["class_probabilities"]):
            assert item["class"] == int(classes[column])
            assert item["probability"] == pytest.approx(
                float(probabilities[position, column]), **_APPROX
            )


def test_gaussian_nb_joint_log_likelihood_reconstruction():
    prepared = _prepare("gaussian_nb", "iris", None)
    payload = _explain(prepared, "gaussian_nb")

    theta = np.asarray(prepared.estimator.theta_, dtype=float)
    var = np.asarray(prepared.estimator.var_, dtype=float)
    prior = np.asarray(prepared.estimator._model.class_prior_, dtype=float)
    probabilities = np.asarray(prepared.estimator.predict_proba(prepared.X_test[:20]))

    for position, sample in enumerate(payload["samples"]):
        features = prepared.X_test[position]
        joint_all = (
            np.log(prior)
            - 0.5 * np.sum(np.log(2.0 * np.pi * var), axis=1)
            - 0.5 * np.sum(((features - theta) ** 2) / var, axis=1)
        )
        normalised = np.exp(joint_all - joint_all.max())
        normalised /= normalised.sum()
        # Independent rebuild of predict_proba from theta_/var_/class_prior_.
        assert np.allclose(normalised, probabilities[position], rtol=1e-6, atol=1e-9)

        class_index = int(
            np.flatnonzero(np.asarray(prepared.estimator.classes_) == sample["predicted_class"])[0]
        )
        assert sample["log_prior"] == pytest.approx(float(np.log(prior[class_index])))
        assert sample["joint_log_likelihood"] == pytest.approx(
            float(joint_all[class_index]), **_APPROX
        )
        assert sample["joint_log_likelihood"] == pytest.approx(
            sample["log_prior"] + sum(sample["feature_log_likelihoods"]), **_APPROX
        )
        expected_feature_terms = (
            -0.5 * np.log(2.0 * np.pi * var[class_index])
            - ((features - theta[class_index]) ** 2) / (2.0 * var[class_index])
        )
        assert np.allclose(sample["feature_log_likelihoods"], expected_feature_terms)


# --- 11. CART decision paths --------------------------------------------------


def test_decision_tree_path_reaches_the_real_leaf():
    prepared = _prepare("decision_tree_regressor", "diabetes", {"max_depth": 3})
    payload = _explain(prepared, "decision_tree_regressor")

    assert payload["kind"] == "decision_path"
    names = list(prepared.data.feature_names)

    for position, sample in enumerate(payload["samples"]):
        node = prepared.estimator.tree
        assert node is not None
        features = prepared.X_test[position]
        for expected_step, step in enumerate(sample["path"], start=1):
            assert step["step"] == expected_step
            assert node["leaf"] is False
            index = int(node["feature"])
            assert step["feature_index"] == index
            assert step["feature_name"] == names[index]
            assert step["threshold"] == float(node["threshold"])
            value = float(features[index])
            assert step["value"] == value
            expected_direction = "left" if value <= float(node["threshold"]) else "right"
            assert step["direction"] == expected_direction
            node = node[expected_direction]

        assert node["leaf"] is True
        assert sample["leaf_prediction"] == pytest.approx(float(node["value"]), **_APPROX)
        assert sample["leaf_prediction"] == pytest.approx(
            float(prepared.y_pred[position]), **_APPROX
        )
        assert sample["y_pred"] == pytest.approx(float(prepared.y_pred[position]), **_APPROX)


# --- 12-15. MLP permutation importance ---------------------------------------


def _mlp_prepared() -> _Prepared:
    return _prepare("mlp_classifier", "iris", {"max_iter": 20, "hidden_size": 8})


def test_mlp_permutation_importance_shape_and_baseline():
    prepared = _mlp_prepared()
    payload = _explain(prepared, "mlp_classifier")

    assert payload["kind"] == "permutation_importance"
    assert payload["feature_space"] == "standardized"
    assert payload["importance_name"] == "test_permutation_accuracy_drop"
    assert payload["feature_names"] == list(prepared.data.feature_names)
    assert len(payload["importances"]) == len(prepared.data.feature_names)
    assert "samples" not in payload

    real_accuracy = float(
        np.mean(np.asarray(prepared.estimator.predict(prepared.X_test)) == prepared.y_test)
    )
    assert payload["baseline_accuracy"] == pytest.approx(real_accuracy, abs=1e-12)
    for importance in payload["importances"]:
        assert math.isfinite(importance)


def test_mlp_permutation_importance_is_deterministic():
    prepared = _mlp_prepared()
    first = _explain(prepared, "mlp_classifier")
    second = _explain(prepared, "mlp_classifier")
    assert first == second


def test_mlp_permutation_importance_does_not_modify_x_test():
    prepared = _mlp_prepared()
    snapshot = prepared.X_test.copy()
    _explain(prepared, "mlp_classifier")
    assert np.array_equal(prepared.X_test, snapshot)


def test_mlp_permutation_importance_does_not_touch_global_random_state():
    prepared = _mlp_prepared()
    np.random.seed(1234)
    expected = np.random.random(5).tolist()

    np.random.seed(1234)
    _explain(prepared, "mlp_classifier")
    actual = np.random.random(5).tolist()
    assert actual == expected


# --- RF/GBDT test-set permutation importance (MSE increase) ------------------


def _ensemble_prepared(model_name: str) -> _Prepared:
    return _prepare(model_name, "diabetes", {"n_estimators": 5})


@pytest.mark.parametrize(
    "model_name", ["random_forest_regressor", "gbdt_regressor"]
)
def test_ensemble_permutation_importance_shape_and_baseline(model_name):
    prepared = _ensemble_prepared(model_name)
    payload = _explain(prepared, model_name)

    assert payload["kind"] == "permutation_importance"
    assert payload["feature_space"] == "standardized"
    assert payload["importance_name"] == "test_permutation_mse_increase"
    assert payload["feature_names"] == list(prepared.data.feature_names)
    assert len(payload["importances"]) == len(prepared.data.feature_names)
    assert "samples" not in payload

    real_mse = float(
        np.mean((prepared.y_test - prepared.estimator.predict(prepared.X_test)) ** 2)
    )
    assert payload["baseline_mse"] == pytest.approx(real_mse, abs=1e-12)
    for importance in payload["importances"]:
        assert math.isfinite(importance)


@pytest.mark.parametrize(
    "model_name", ["random_forest_regressor", "gbdt_regressor"]
)
def test_ensemble_permutation_importance_is_deterministic(model_name):
    prepared = _ensemble_prepared(model_name)
    assert _explain(prepared, model_name) == _explain(prepared, model_name)


@pytest.mark.parametrize(
    "model_name", ["random_forest_regressor", "gbdt_regressor"]
)
def test_ensemble_permutation_importance_does_not_modify_x_test(model_name):
    prepared = _ensemble_prepared(model_name)
    snapshot = prepared.X_test.copy()
    _explain(prepared, model_name)
    assert np.array_equal(prepared.X_test, snapshot)


@pytest.mark.parametrize(
    "model_name", ["random_forest_regressor", "gbdt_regressor"]
)
def test_ensemble_permutation_importance_keeps_global_random_state(model_name):
    prepared = _ensemble_prepared(model_name)
    np.random.seed(1234)
    expected = np.random.random(5).tolist()

    np.random.seed(1234)
    _explain(prepared, model_name)
    actual = np.random.random(5).tolist()
    assert actual == expected


def test_ensemble_permutation_importance_allows_negative_values():
    """A permutation that improves the MSE must stay negative (never clipped)."""

    class _FirstColumnRegressor:
        def predict(self, X):
            return np.asarray(X, dtype=np.float64)[:, 0]

    # Prediction = first column; y is that column reversed, so the reversed
    # permutation reconstructs y exactly and yields a negative importance.
    X_test = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_test = np.array([3.0, 2.0, 1.0, 0.0])
    seed = next(
        s
        for s in range(100)
        if list(np.random.default_rng(s).permutation(4)) == [3, 2, 1, 0]
    )

    payload = build_explanation_data(
        model_name="random_forest_regressor",
        estimator=_FirstColumnRegressor(),
        X_train=X_test,
        y_train=y_test,
        X_test=X_test,
        y_test=y_test,
        y_pred=np.zeros(4),
        feature_names=["only"],
        random_state=seed,
    )

    baseline = float(np.mean((y_test - X_test[:, 0]) ** 2))
    assert payload["baseline_mse"] == pytest.approx(baseline, abs=1e-12)
    assert payload["importances"] == [pytest.approx(-baseline, abs=1e-12)]
    assert payload["importances"][0] < 0.0


def test_decision_tree_still_returns_decision_path_not_permutation():
    prepared = _prepare("decision_tree_regressor", "diabetes", {"max_depth": 3})
    payload = _explain(prepared, "decision_tree_regressor")
    assert payload["kind"] == "decision_path"


# --- 17/18. serialization & validation ---------------------------------------


_JSON_CASES = [
    ("linear_regression", "diabetes", {"max_iter": 30}),
    ("ridge_regression", "diabetes", {"max_iter": 30}),
    ("logistic_regression", "iris", {"max_iter": 30}),
    ("linear_svm", "iris", {"max_iter": 30}),
    ("knn_classifier", "iris", None),
    ("gaussian_nb", "iris", None),
    ("decision_tree_regressor", "diabetes", {"max_depth": 3}),
    ("mlp_classifier", "iris", {"max_iter": 10, "hidden_size": 8}),
    ("random_forest_regressor", "diabetes", {"n_estimators": 5}),
    ("gbdt_regressor", "diabetes", {"n_estimators": 5}),
]


@pytest.mark.parametrize("model_name,dataset_name,hyperparameters", _JSON_CASES)
def test_payload_is_json_serializable_and_inputs_unchanged(
    model_name, dataset_name, hyperparameters
):
    prepared = _prepare(model_name, dataset_name, hyperparameters)
    snapshots = {
        "X_train": prepared.X_train.copy(),
        "X_test": prepared.X_test.copy(),
        "y_train": prepared.y_train.copy(),
        "y_test": prepared.y_test.copy(),
        "y_pred": prepared.y_pred.copy(),
    }

    payload = _explain(prepared, model_name)
    assert payload is not None

    # allow_nan=False turns any NaN / Infinity into a hard failure.
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == payload
    for value in _iter_floats(payload):
        assert math.isfinite(value)

    assert np.array_equal(prepared.X_train, snapshots["X_train"])
    assert np.array_equal(prepared.X_test, snapshots["X_test"])
    assert np.array_equal(prepared.y_train, snapshots["y_train"])
    assert np.array_equal(prepared.y_test, snapshots["y_test"])
    assert np.array_equal(prepared.y_pred, snapshots["y_pred"])


def _fake_linear(coef, intercept):
    class _Dummy:
        pass

    dummy = _Dummy()
    dummy.coef_ = np.asarray(coef, dtype=float)
    dummy.intercept_ = float(intercept)
    return dummy


def test_non_finite_features_are_rejected():
    for value in (np.nan, np.inf, -np.inf):
        X_test = np.zeros((2, 2))
        X_test[0, 0] = value
        with pytest.raises(ExplanationError, match="linear_regression"):
            build_explanation_data(
                model_name="linear_regression",
                estimator=_fake_linear([1.0, 1.0], 0.0),
                X_train=np.zeros((4, 2)),
                y_train=np.zeros(4),
                X_test=X_test,
                y_test=np.zeros(2),
                y_pred=np.zeros(2),
                feature_names=["a", "b"],
                random_state=42,
            )


def test_non_finite_y_true_is_rejected():
    with pytest.raises(ExplanationError, match="non-finite y_true"):
        build_explanation_data(
            model_name="linear_regression",
            estimator=_fake_linear([1.0, 1.0], 0.0),
            X_train=np.zeros((4, 2)),
            y_train=np.zeros(4),
            X_test=np.array([[1.0, 2.0]]),
            y_test=np.array([np.nan]),
            y_pred=np.array([3.0]),
            feature_names=["a", "b"],
            random_state=42,
        )


def test_inconsistent_reconstruction_is_rejected():
    with pytest.raises(ExplanationError, match="reconstruction mismatch"):
        build_explanation_data(
            model_name="linear_regression",
            estimator=_fake_linear([1.0, 1.0], 0.0),
            X_train=np.zeros((4, 2)),
            y_train=np.zeros(4),
            X_test=np.array([[1.0, 2.0]]),
            y_test=np.array([3.0]),
            y_pred=np.array([999.0]),
            feature_names=["a", "b"],
            random_state=42,
        )


def test_feature_name_mismatch_is_rejected():
    with pytest.raises(ExplanationError, match="feature_names"):
        build_explanation_data(
            model_name="linear_regression",
            estimator=_fake_linear([1.0, 1.0], 0.0),
            X_train=np.zeros((4, 2)),
            y_train=np.zeros(4),
            X_test=np.array([[1.0, 2.0]]),
            y_test=np.array([3.0]),
            y_pred=np.array([3.0]),
            feature_names=["only-one"],
            random_state=42,
        )


def test_mismatched_train_test_feature_count_is_rejected():
    with pytest.raises(ExplanationError, match="features"):
        build_explanation_data(
            model_name="linear_regression",
            estimator=_fake_linear([1.0, 1.0], 0.0),
            X_train=np.zeros((4, 3)),
            y_train=np.zeros(4),
            X_test=np.array([[1.0, 2.0]]),
            y_test=np.array([3.0]),
            y_pred=np.array([3.0]),
            feature_names=["a", "b"],
            random_state=42,
        )


# --- 19. schema compatibility -------------------------------------------------


def test_training_info_without_explanation_data_still_parses():
    info = TrainingInfo.model_validate({"loss_history": [1.0], "n_iter": 1})
    assert info.explanation_data is None


def test_training_info_with_explanation_data_parses():
    payload = {
        "schema_version": 1,
        "kind": "linear_contribution",
        "feature_space": "standardized",
        "feature_names": ["a"],
        "sample_count": 0,
        "samples": [],
    }
    info = TrainingInfo.model_validate({"explanation_data": payload})
    assert info.explanation_data == payload


# --- 21. unsupported models ---------------------------------------------------


def test_estimator_without_required_fitted_state_returns_none():
    class _FakeClassifier:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    result = build_explanation_data(
        model_name="logistic_regression",
        estimator=_FakeClassifier(),
        X_train=np.zeros((4, 2)),
        y_train=np.zeros(4),
        X_test=np.zeros((2, 2)),
        y_test=np.zeros(2),
        y_pred=np.zeros(2),
        feature_names=["a", "b"],
        random_state=42,
    )
    assert result is None


@pytest.mark.parametrize("model_name", ["unknown_model"])
def test_models_without_faithful_explanation_return_none(model_name):
    result = build_explanation_data(
        model_name=model_name,
        estimator=None,
        X_train=np.zeros((4, 2)),
        y_train=np.zeros(4),
        X_test=np.zeros((2, 2)),
        y_test=np.zeros(2),
        y_pred=np.zeros(2),
        feature_names=["a", "b"],
        random_state=42,
    )
    assert result is None


# --- KNNModel.kneighbors read-only proxy -------------------------------------


def test_knn_kneighbors_requires_a_fitted_model():
    with pytest.raises(ValueError, match="not fitted"):
        KNNModel().kneighbors(np.zeros((1, 2)))


def test_knn_kneighbors_returns_real_training_indices_without_changing_behavior():
    X_train = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2.25], [10.6]])

    model = KNNModel(n_neighbors=2, weights="uniform").fit(X_train, y_train)

    distances, indices = model.kneighbors(X_test)
    assert distances.shape == (2, 2)
    assert indices.shape == (2, 2)

    brute_force = np.linalg.norm(
        X_test[:, None, :] - X_train[None, :, :], axis=2
    )
    expected_indices = np.argsort(brute_force, axis=1)[:, :2]
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(
        distances, np.take_along_axis(brute_force, expected_indices, axis=1)
    )
    assert np.all([0 <= int(i) < len(X_train) for i in indices.ravel()])

    only_indices = model.kneighbors(X_test, n_neighbors=3, return_distance=False)
    assert only_indices.shape == (2, 3)

    # fit / predict behaviour is unchanged by the read-only proxy.
    predictions = model.predict(X_test)
    assert predictions.tolist() == [0, 1]
    probabilities = model.predict_proba(X_test)
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.n_features_in_ == 1


# --- 20/22/23/24. end-to-end storage & completion log ------------------------


def _run_experiment(client, dataset_name, model_name, hyperparameters=None):
    payload = {"dataset_name": dataset_name, "model_name": model_name}
    if hyperparameters:
        payload["hyperparameters"] = hyperparameters
    created = client.post("/api/v1/experiments", json=payload).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()
    assert detail["status"] == "completed", detail.get("error")
    return detail


def test_summary_json_explanation_matches_api(client):
    detail = _run_experiment(
        client, "iris", "logistic_regression", {"max_iter": 20}
    )
    info = detail["training_info"]
    explanation = info["explanation_data"]
    assert explanation["kind"] == "multiclass_linear_contribution"
    assert explanation["feature_space"] == "standardized"

    # Existing training_info fields must be preserved, not overwritten.
    assert info["coefficients"]
    assert detail["hyperparameters"] == {"max_iter": 20}

    summary_path = (
        Path(get_settings().experiments_dir) / detail["artifact_dir"] / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["training_info"]["explanation_data"] == explanation


def test_summary_json_explanation_matches_api_for_ensemble(client):
    detail = _run_experiment(
        client, "diabetes", "random_forest_regressor", {"n_estimators": 5}
    )
    explanation = detail["training_info"]["explanation_data"]
    summary_path = (
        Path(get_settings().experiments_dir) / detail["artifact_dir"] / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["training_info"]["explanation_data"] == explanation


@pytest.mark.parametrize("model_name", ["random_forest_regressor", "gbdt_regressor"])
def test_tree_ensembles_store_real_permutation_importance(client, model_name):
    detail = _run_experiment(client, "diabetes", model_name, {"n_estimators": 5})
    info = detail["training_info"]
    explanation = info["explanation_data"]
    assert explanation["kind"] == "permutation_importance"
    assert explanation["importance_name"] == "test_permutation_mse_increase"
    assert math.isfinite(explanation["baseline_mse"])
    assert len(explanation["importances"]) == len(info["feature_names"])
    # The built-in training importance is kept alongside; the two are
    # different measurements and must both survive.
    assert info["feature_importances"]
    assert len(info["feature_importances"]) == len(info["feature_names"])


def test_experiment_completed_never_logs_a_chart_count(client):
    detail = _run_experiment(client, "iris", "knn_classifier")
    log = detail["training_info"]["process_log"]
    completed = [entry for entry in log if entry["event"] == "experiment_completed"]
    assert len(completed) == 1
    details = completed[0]["details"]
    assert set(details) == {"total_duration_ms"}
    assert isinstance(details["total_duration_ms"], float)
    assert math.isfinite(details["total_duration_ms"])
    assert details["total_duration_ms"] >= 0.0

    serialized = json.dumps(log, ensure_ascii=False)
    for forbidden in ("chart_count", "共生成", "生成 0 张图表", "张图表"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "relative_path",
    [
        "backend/services/experiment_service.py",
        "backend/services/training_log.py",
        "frontend/src/utils/resultLabels.js",
    ],
)
def test_production_code_never_renders_a_chart_count(relative_path):
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    for forbidden in ("共生成", "生成 0 张图表", "张图表"):
        assert forbidden not in text, f"{forbidden!r} found in {relative_path}"
