"""Real, JSON-serializable model explanation data for experiment results.

The result page must show *actual* model behaviour, never invented numbers.
Every value produced here is read from the fitted estimator, the real
train/test arrays, or an actual forward/query call. Models that cannot
expose a faithful explanation simply return ``None``.

Important context: ``X_train`` / ``X_test`` arrive already standardized by
``ml_engine.datasets.base.split_dataset``, so every payload records
``"feature_space": "standardized"`` and never pretends to be raw units.

Supported kinds
---------------
``linear_contribution``
    Linear / ridge regression: per-sample ``x_j * coef_j`` terms that
    reconstruct ``y_pred`` when summed with the intercept.
``multiclass_linear_contribution``
    Logistic regression (softmax logits) and linear SVM (decision scores):
    the predicted class' linear score components.
``knn_neighbors``
    Real neighbours from ``KNeighborsClassifier.kneighbors`` with training-set
    indices, distances, true labels and sklearn's vote weights.
``gaussian_nb_likelihood``
    GaussianNB: real ``predict_proba`` plus feature log-likelihoods rebuilt
    from the fitted ``theta_`` / ``var_`` / ``class_prior_``.
``decision_path``
    CART regression tree: a real root-to-leaf traversal of the fitted tree.
``permutation_importance``
    MLPClassifier (accuracy drop) and the tree ensembles (MSE increase):
    a deterministic test-set permutation importance, never internal neuron
    weights and never a re-labelled copy of the built-in ``feature_importances_``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MAX_SAMPLES = 20
SCHEMA_VERSION = 1
FEATURE_SPACE = "standardized"

_LINEAR_REGRESSION_MODELS = frozenset({"linear_regression", "ridge_regression"})
# model_name -> ``score_name`` recorded in the payload.
_MULTICLASS_LINEAR_MODELS = {
    "logistic_regression": "logit",
    "linear_svm": "decision_score",
}
# Tree ensembles expose a test-set permutation importance (MSE increase).
_PERMUTATION_REGRESSORS = frozenset({"random_forest_regressor", "gbdt_regressor"})

_ATOL = 1e-6
_RTOL = 1e-6

# sklearn's GaussianNB uses ``class_log_prior_``; fall back to log(class_prior_).
_TWO_PI = 2.0 * math.pi


class ExplanationError(ValueError):
    """Raised when an explanation cannot be built from consistent real data."""


# --- small conversion helpers ------------------------------------------------


def _json_scalar(value: Any, *, model_name: str, context: str) -> Any:
    """Convert one label/response value into a JSON-native scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExplanationError(f"{model_name}: non-finite {context} value {value!r}.")
        return value
    if value is None:
        return None
    raise ExplanationError(
        f"{model_name}: unsupported {context} type {type(value).__name__}."
    )


def _finite_floats(values: Any, *, model_name: str, context: str) -> list[float]:
    """Return ``values`` as a finite float list (raises on NaN/Inf)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size and not np.all(np.isfinite(array)):
        raise ExplanationError(f"{model_name}: non-finite values in {context}.")
    return [float(v) for v in array.ravel()]


def _as_2d(X: Any, *, model_name: str, name: str) -> np.ndarray:
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ExplanationError(f"{model_name}: {name} must be 2D, got {array.ndim}D.")
    if not np.all(np.isfinite(array)):
        raise ExplanationError(f"{model_name}: non-finite values in {name}.")
    return array


def _as_1d(values: Any, *, model_name: str, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ExplanationError(f"{model_name}: {name} must be 1D, got {array.ndim}D.")
    return array


def _check_close(
    actual: float, expected: float, *, model_name: str, context: str
) -> None:
    """Raise when a reconstructed value does not match the model output."""
    if not math.isclose(actual, expected, rel_tol=_RTOL, abs_tol=_ATOL):
        raise ExplanationError(
            f"{model_name}: {context} reconstruction mismatch "
            f"(reconstructed={actual!r}, model={expected!r})."
        )


def _resolve_feature_names(
    feature_names: Any, n_features: int, *, model_name: str
) -> list[str]:
    if feature_names is None:
        return [f"feature_{index + 1}" for index in range(n_features)]
    names = [str(name) for name in feature_names]
    if len(names) != n_features:
        raise ExplanationError(
            f"{model_name}: feature_names has {len(names)} entries but X has "
            f"{n_features} features."
        )
    return names


def _base_payload(
    *,
    kind: str,
    feature_names: list[str],
    sample_count: int,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "feature_space": FEATURE_SPACE,
        "feature_names": feature_names,
        "sample_count": sample_count,
        "samples": samples,
    }


def _n_samples_to_explain(n_test: int) -> int:
    return min(MAX_SAMPLES, n_test)


# --- 1. linear / ridge regression -------------------------------------------


def _linear_contribution(
    *,
    model_name: str,
    estimator: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    coef = np.asarray(estimator.coef_, dtype=np.float64).ravel()
    intercept = float(estimator.intercept_)
    n_features = X_test.shape[1]
    if coef.shape[0] != n_features:
        raise ExplanationError(
            f"{model_name}: coef_ has {coef.shape[0]} entries but X has "
            f"{n_features} features."
        )

    samples: list[dict[str, Any]] = []
    for index in range(_n_samples_to_explain(len(X_test))):
        features = X_test[index]
        contributions = features * coef
        reconstructed = float(np.sum(contributions) + intercept)
        _check_close(
            reconstructed,
            float(np.asarray(y_pred[index], dtype=np.float64)),
            model_name=model_name,
            context=f"linear contribution (sample {index + 1})",
        )
        samples.append(
            {
                "sample_index": index + 1,
                "y_true": _json_scalar(y_test[index], model_name=model_name, context="y_true"),
                "y_pred": _json_scalar(y_pred[index], model_name=model_name, context="y_pred"),
                "feature_values": _finite_floats(
                    features, model_name=model_name, context="feature_values"
                ),
                "contributions": _finite_floats(
                    contributions, model_name=model_name, context="contributions"
                ),
                "intercept": intercept,
                "reconstructed_prediction": reconstructed,
            }
        )

    return _base_payload(
        kind="linear_contribution",
        feature_names=feature_names,
        sample_count=len(samples),
        samples=samples,
    )


# --- 2. logistic regression / linear SVM ------------------------------------


def _multiclass_linear_contribution(
    *,
    model_name: str,
    score_name: str,
    estimator: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    coef = np.asarray(estimator.coef_, dtype=np.float64)
    intercept = np.asarray(estimator.intercept_, dtype=np.float64).ravel()
    classes = np.asarray(estimator.classes_)
    n_classes, n_features = coef.shape
    if coef.shape[1] != X_test.shape[1] or intercept.shape[0] != n_classes:
        raise ExplanationError(
            f"{model_name}: coefficient shape {coef.shape} / intercept "
            f"{intercept.shape} do not match X ({X_test.shape}) and classes "
            f"({classes.shape[0]})."
        )

    # ``decision_function`` gives an independent reference score where the
    # model exposes one (LinearSVM). Logistic regression only exposes
    # probabilities, so its score is rebuilt from the fitted coefficients.
    reference_scores = None
    if hasattr(estimator, "decision_function"):
        reference_scores = np.asarray(estimator.decision_function(X_test), dtype=np.float64)

    classes_list = [_json_scalar(c, model_name=model_name, context="class") for c in classes]
    samples: list[dict[str, Any]] = []
    for index in range(_n_samples_to_explain(len(X_test))):
        predicted = y_pred[index]
        matches = np.flatnonzero(classes == predicted)
        if matches.size == 0:
            raise ExplanationError(
                f"{model_name}: predicted label {predicted!r} is not in classes_."
            )
        class_index = int(matches[0])

        features = X_test[index]
        contributions = features * coef[class_index]
        reconstructed = float(np.sum(contributions) + intercept[class_index])
        if reference_scores is not None:
            _check_close(
                reconstructed,
                float(reference_scores[index, class_index]),
                model_name=model_name,
                context=f"multiclass linear score (sample {index + 1})",
            )

        samples.append(
            {
                "sample_index": index + 1,
                "y_true": _json_scalar(y_test[index], model_name=model_name, context="y_true"),
                "y_pred": _json_scalar(predicted, model_name=model_name, context="y_pred"),
                "predicted_class": classes_list[class_index],
                "class_index": class_index,
                "feature_values": _finite_floats(
                    features, model_name=model_name, context="feature_values"
                ),
                "contributions": _finite_floats(
                    contributions, model_name=model_name, context="contributions"
                ),
                "intercept": float(intercept[class_index]),
                "reconstructed_score": reconstructed,
            }
        )

    payload = _base_payload(
        kind="multiclass_linear_contribution",
        feature_names=feature_names,
        sample_count=len(samples),
        samples=samples,
    )
    payload["score_name"] = score_name
    payload["classes"] = classes_list
    return payload


# --- 3. KNN neighbours -------------------------------------------------------


def _knn_weights(distances: np.ndarray, mode: str) -> np.ndarray:
    """Reproduce sklearn's neighbour weights, including the zero-distance rule.

    ``uniform``: every neighbour weighs 1.
    ``distance``: weight = 1 / distance; when a sample has any zero-distance
    neighbour, that row becomes 1 for the zero-distance neighbours and 0 for
    the rest (exactly sklearn's ``_get_weights`` behaviour).
    """
    if mode == "uniform":
        return np.ones_like(distances, dtype=np.float64)
    if mode != "distance":
        raise ExplanationError(f"KNN: unsupported weights mode {mode!r}.")
    with np.errstate(divide="ignore"):
        weights = 1.0 / distances
    for row_index in range(weights.shape[0]):
        infinite = np.isinf(weights[row_index])
        if infinite.any():
            weights[row_index] = infinite.astype(np.float64)
    return weights


def _knn_neighbors(
    *,
    model_name: str,
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    classes = np.asarray(estimator.classes_)
    n_train = len(X_train)
    requested = int(getattr(estimator, "n_neighbors", 5))
    k = max(1, min(requested, n_train))
    mode = str(getattr(estimator, "weights", "uniform"))

    count = _n_samples_to_explain(len(X_test))
    distances, indices = estimator.kneighbors(
        X_test[:count], n_neighbors=k, return_distance=True
    )
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices)
    if distances.shape != (count, k) or indices.shape != (count, k):
        raise ExplanationError(
            f"{model_name}: kneighbors returned shapes {distances.shape} / "
            f"{indices.shape}, expected ({count}, {k})."
        )
    if not np.all(np.isfinite(distances)):
        raise ExplanationError(f"{model_name}: non-finite neighbour distances.")

    weights = _knn_weights(distances, mode)

    samples: list[dict[str, Any]] = []
    for row in range(count):
        order = np.argsort(distances[row], kind="stable")
        neighbours: list[dict[str, Any]] = []
        totals = np.zeros(len(classes), dtype=np.float64)
        for rank, position in enumerate(order, start=1):
            train_index = int(indices[row, position])
            if train_index < 0 or train_index >= n_train:
                raise ExplanationError(
                    f"{model_name}: neighbour train_index {train_index} is out of "
                    f"range for {n_train} training samples."
                )
            label = y_train[train_index]
            matches = np.flatnonzero(classes == label)
            if matches.size == 0:
                raise ExplanationError(
                    f"{model_name}: neighbour label {label!r} is not in classes_."
                )
            weight = float(weights[row, position])
            totals[int(matches[0])] += weight
            neighbours.append(
                {
                    "rank": rank,
                    "train_index": train_index,
                    "distance": float(distances[row, position]),
                    "label": _json_scalar(label, model_name=model_name, context="neighbour label"),
                    "vote_weight": weight,
                }
            )

        vote_totals = [
            {
                "class": _json_scalar(classes[i], model_name=model_name, context="class"),
                "weight": float(totals[i]),
            }
            for i in range(len(classes))
        ]
        top = np.flatnonzero(np.isclose(totals, totals.max(), rtol=0.0, atol=1e-12))
        top_classes = [classes[i] for i in top]
        if not any(top_class == y_pred[row] for top_class in top_classes):
            raise ExplanationError(
                f"{model_name}: vote_totals does not agree with the prediction "
                f"{y_pred[row]!r} at sample {row + 1}."
            )

        samples.append(
            {
                "sample_index": row + 1,
                "y_true": _json_scalar(y_test[row], model_name=model_name, context="y_true"),
                "y_pred": _json_scalar(y_pred[row], model_name=model_name, context="y_pred"),
                "neighbors": neighbours,
                "vote_totals": vote_totals,
            }
        )

    payload = _base_payload(
        kind="knn_neighbors",
        feature_names=feature_names,
        sample_count=len(samples),
        samples=samples,
    )
    payload["n_neighbors"] = k
    payload["weights"] = mode
    return payload


# --- 4. GaussianNB likelihoods ----------------------------------------------


def _gaussian_nb_likelihood(
    *,
    model_name: str,
    estimator: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    classes = np.asarray(estimator.classes_)
    theta = np.asarray(estimator.theta_, dtype=np.float64)  # (n_classes, n_features)
    var = np.asarray(estimator.var_, dtype=np.float64)
    n_classes, n_features = theta.shape
    if var.shape != theta.shape or n_features != X_test.shape[1]:
        raise ExplanationError(
            f"{model_name}: theta_/var_ shapes {theta.shape} / {var.shape} do not "
            f"match X with {X_test.shape[1]} features."
        )

    inner = getattr(estimator, "_model", None)
    class_prior = getattr(inner, "class_prior_", None)
    if class_prior is None:
        raise ExplanationError(f"{model_name}: fitted class_prior_ is not available.")
    class_prior = np.asarray(class_prior, dtype=np.float64).ravel()
    if class_prior.shape[0] != n_classes:
        raise ExplanationError(
            f"{model_name}: class_prior_ has {class_prior.shape[0]} entries but "
            f"classes_ has {n_classes}."
        )

    count = _n_samples_to_explain(len(X_test))
    probabilities = np.asarray(estimator.predict_proba(X_test[:count]), dtype=np.float64)
    if probabilities.shape != (count, n_classes):
        raise ExplanationError(
            f"{model_name}: predict_proba returned {probabilities.shape}, expected "
            f"({count}, {n_classes})."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ExplanationError(f"{model_name}: non-finite predict_proba values.")

    joint_reference = None
    joint_fn = getattr(inner, "_joint_log_likelihood", None)
    if callable(joint_fn):
        joint_reference = np.asarray(joint_fn(X_test[:count]), dtype=np.float64)

    classes_list = [_json_scalar(c, model_name=model_name, context="class") for c in classes]
    samples: list[dict[str, Any]] = []
    for row in range(count):
        matches = np.flatnonzero(classes == y_pred[row])
        if matches.size == 0:
            raise ExplanationError(
                f"{model_name}: predicted label {y_pred[row]!r} is not in classes_."
            )
        class_index = int(matches[0])

        features = X_test[row]
        log_prior = float(np.log(class_prior[class_index]))
        feature_log_likelihoods = (
            -0.5 * np.log(_TWO_PI * var[class_index])
            - ((features - theta[class_index]) ** 2) / (2.0 * var[class_index])
        )
        joint = float(log_prior + np.sum(feature_log_likelihoods))

        probability_sum = float(np.sum(probabilities[row]))
        _check_close(
            probability_sum,
            1.0,
            model_name=model_name,
            context=f"class probabilities row sum (sample {row + 1})",
        )
        if joint_reference is not None:
            _check_close(
                joint,
                float(joint_reference[row, class_index]),
                model_name=model_name,
                context=f"joint log likelihood (sample {row + 1})",
            )

        samples.append(
            {
                "sample_index": row + 1,
                "y_true": _json_scalar(y_test[row], model_name=model_name, context="y_true"),
                "y_pred": _json_scalar(y_pred[row], model_name=model_name, context="y_pred"),
                "predicted_class": classes_list[class_index],
                "class_probabilities": [
                    {
                        "class": classes_list[i],
                        "probability": float(probabilities[row, i]),
                    }
                    for i in range(n_classes)
                ],
                "feature_values": _finite_floats(
                    features, model_name=model_name, context="feature_values"
                ),
                "feature_log_likelihoods": _finite_floats(
                    feature_log_likelihoods,
                    model_name=model_name,
                    context="feature_log_likelihoods",
                ),
                "log_prior": log_prior,
                "joint_log_likelihood": joint,
            }
        )

    payload = _base_payload(
        kind="gaussian_nb_likelihood",
        feature_names=feature_names,
        sample_count=len(samples),
        samples=samples,
    )
    payload["classes"] = classes_list
    return payload


# --- 5. CART decision path ---------------------------------------------------


def _decision_path(
    *,
    model_name: str,
    estimator: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    root = getattr(estimator, "tree", None)
    if not isinstance(root, dict):
        raise ExplanationError(f"{model_name}: fitted tree structure is not available.")

    def _require_node(node: Any, context: str) -> dict[str, Any]:
        if not isinstance(node, dict):
            raise ExplanationError(f"{model_name}: malformed tree node at {context}.")
        if "leaf" not in node:
            raise ExplanationError(f"{model_name}: tree node at {context} lacks 'leaf'.")
        if not node["leaf"]:
            for key in ("feature", "threshold", "left", "right"):
                if key not in node:
                    raise ExplanationError(
                        f"{model_name}: split node at {context} lacks {key!r}."
                    )
        elif "value" not in node:
            raise ExplanationError(f"{model_name}: leaf node at {context} lacks 'value'.")
        return node

    count = _n_samples_to_explain(len(X_test))
    samples: list[dict[str, Any]] = []
    for row in range(count):
        node = _require_node(root, "root")
        features = X_test[row]
        path: list[dict[str, Any]] = []
        step = 1
        while not node["leaf"]:
            feature_index = int(node["feature"])
            threshold = float(node["threshold"])
            if feature_index < 0 or feature_index >= features.shape[0]:
                raise ExplanationError(
                    f"{model_name}: split feature index {feature_index} is out of "
                    f"range for {features.shape[0]} features."
                )
            value = float(features[feature_index])
            direction = "left" if value <= threshold else "right"
            path.append(
                {
                    "step": step,
                    "feature_index": feature_index,
                    "feature_name": feature_names[feature_index],
                    "value": value,
                    "threshold": threshold,
                    "direction": direction,
                }
            )
            node = _require_node(node[direction], f"sample {row + 1} step {step}")
            step += 1

        leaf_prediction = float(node["value"])
        _check_close(
            leaf_prediction,
            float(np.asarray(y_pred[row], dtype=np.float64)),
            model_name=model_name,
            context=f"leaf prediction (sample {row + 1})",
        )

        samples.append(
            {
                "sample_index": row + 1,
                "y_true": _json_scalar(y_test[row], model_name=model_name, context="y_true"),
                "y_pred": _json_scalar(y_pred[row], model_name=model_name, context="y_pred"),
                "path": path,
                "leaf_prediction": leaf_prediction,
            }
        )

    return _base_payload(
        kind="decision_path",
        feature_names=feature_names,
        sample_count=len(samples),
        samples=samples,
    )


# --- 6. MLP permutation importance ------------------------------------------


def _permutation_importance(
    *,
    model_name: str,
    estimator: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    random_state: int | None,
) -> dict[str, Any]:
    """Deterministic test-set permutation importance (accuracy drop).

    One permutation per feature, driven by a local ``np.random.default_rng``
    seeded with the experiment's ``random_state``. ``X_test`` is never
    modified and NumPy's global random state is untouched. Importances may be
    negative (a permutation that happens to help) and are never clipped.
    """
    y_true = np.asarray(y_test)

    def _accuracy(features: np.ndarray) -> float:
        predictions = np.asarray(estimator.predict(features))
        return float(np.mean(predictions == y_true))

    baseline = _accuracy(X_test)
    n_features = X_test.shape[1]
    seed = 0 if random_state is None else int(random_state)
    rng = np.random.default_rng(seed)

    importances: list[float] = []
    for index in range(n_features):
        permuted = X_test.copy()
        permuted[:, index] = permuted[rng.permutation(permuted.shape[0]), index]
        importances.append(baseline - _accuracy(permuted))

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "permutation_importance",
        "feature_space": FEATURE_SPACE,
        "feature_names": feature_names,
        "baseline_accuracy": baseline,
        "importance_name": "test_permutation_accuracy_drop",
        "importances": importances,
    }


# --- 7. tree-ensemble permutation importance ---------------------------------


def _permutation_importance_mse(
    *,
    model_name: str,
    estimator: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    random_state: int | None,
) -> dict[str, Any]:
    """Deterministic test-set permutation importance (MSE increase).

    ``baseline_mse`` is the real test-set MSE of the fitted ensemble. Each
    feature column is permuted exactly once via a local
    ``np.random.default_rng`` seeded with the experiment's ``random_state``;
    ``importance_j = permuted_mse_j - baseline_mse``. Negative values (a
    permutation that happens to help) are kept as-is. ``X_test`` is never
    modified, the estimator is never retrained, and NumPy's global random
    state is untouched. This is a test-set measurement, not the built-in
    ``feature_importances_`` of the training procedure.
    """
    y_true = np.asarray(y_test, dtype=np.float64)
    if not np.all(np.isfinite(y_true)):
        raise ExplanationError(f"{model_name}: non-finite values in y_test.")

    def _mse(features: np.ndarray) -> float:
        predictions = np.asarray(estimator.predict(features), dtype=np.float64)
        return float(np.mean((y_true - predictions) ** 2))

    baseline = _mse(X_test)
    if not math.isfinite(baseline):
        raise ExplanationError(f"{model_name}: non-finite baseline MSE.")
    n_features = X_test.shape[1]
    seed = 0 if random_state is None else int(random_state)
    rng = np.random.default_rng(seed)

    importances: list[float] = []
    for index in range(n_features):
        permuted = X_test.copy()
        permuted[:, index] = permuted[rng.permutation(permuted.shape[0]), index]
        permuted_mse = _mse(permuted)
        if not math.isfinite(permuted_mse):
            raise ExplanationError(
                f"{model_name}: non-finite permuted MSE for feature {index}."
            )
        importances.append(permuted_mse - baseline)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "permutation_importance",
        "feature_space": FEATURE_SPACE,
        "feature_names": feature_names,
        "baseline_mse": baseline,
        "importance_name": "test_permutation_mse_increase",
        "importances": importances,
    }


# --- public entry point ------------------------------------------------------


def _explanation_supported(model_name: str, estimator: Any) -> bool:
    """Whether ``estimator`` exposes the fitted state an explanation needs.

    A model whose estimator lacks the required attributes/methods has no
    faithful explanation available, so the caller stores nothing instead of
    inventing values. Shape/consistency problems on a *complete* estimator
    still raise in the dedicated builders below.
    """
    if model_name in _LINEAR_REGRESSION_MODELS:
        return hasattr(estimator, "coef_") and hasattr(estimator, "intercept_")
    if model_name in _MULTICLASS_LINEAR_MODELS:
        return all(
            hasattr(estimator, name) for name in ("coef_", "intercept_", "classes_")
        )
    if model_name == "knn_classifier":
        return (
            callable(getattr(estimator, "kneighbors", None))
            and hasattr(estimator, "classes_")
            and hasattr(estimator, "n_neighbors")
        )
    if model_name == "gaussian_nb":
        inner = getattr(estimator, "_model", None)
        return (
            all(
                hasattr(estimator, name)
                for name in ("classes_", "theta_", "var_", "predict_proba")
            )
            and getattr(inner, "class_prior_", None) is not None
        )
    if model_name == "decision_tree_regressor":
        return isinstance(getattr(estimator, "tree", None), dict)
    if model_name == "mlp_classifier":
        return callable(getattr(estimator, "predict", None))
    if model_name in _PERMUTATION_REGRESSORS:
        return callable(getattr(estimator, "predict", None))
    return False


def build_explanation_data(
    *,
    model_name: str,
    estimator: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    y_pred: Any,
    feature_names: Any,
    random_state: int | None,
) -> dict[str, Any] | None:
    """Build the ``training_info.explanation_data`` payload for one experiment.

    Returns ``None`` for models without a faithful explanation (unknown model
    names, or estimators missing the required fitted state). Raises
    :class:`ExplanationError` when the model is supported but the real data
    cannot be reconciled — never returns invented values.
    """
    if model_name not in _SUPPORTED_MODELS:
        return None
    if not _explanation_supported(model_name, estimator):
        return None

    X_train_arr = _as_2d(X_train, model_name=model_name, name="X_train")
    X_test_arr = _as_2d(X_test, model_name=model_name, name="X_test")
    y_train_arr = _as_1d(y_train, model_name=model_name, name="y_train")
    y_test_arr = _as_1d(y_test, model_name=model_name, name="y_test")
    y_pred_arr = _as_1d(y_pred, model_name=model_name, name="y_pred")

    if X_train_arr.shape[1] != X_test_arr.shape[1]:
        raise ExplanationError(
            f"{model_name}: X_train has {X_train_arr.shape[1]} features but X_test "
            f"has {X_test_arr.shape[1]}."
        )
    if len(y_test_arr) != len(X_test_arr) or len(y_pred_arr) != len(X_test_arr):
        raise ExplanationError(
            f"{model_name}: y_test / y_pred lengths ({len(y_test_arr)} / "
            f"{len(y_pred_arr)}) do not match X_test ({len(X_test_arr)})."
        )

    names = _resolve_feature_names(feature_names, X_test_arr.shape[1], model_name=model_name)
    common = {
        "model_name": model_name,
        "estimator": estimator,
        "X_test": X_test_arr,
        "y_test": y_test_arr,
        "y_pred": y_pred_arr,
        "feature_names": names,
    }

    if model_name in _LINEAR_REGRESSION_MODELS:
        return _linear_contribution(**common)
    if model_name in _MULTICLASS_LINEAR_MODELS:
        return _multiclass_linear_contribution(
            score_name=_MULTICLASS_LINEAR_MODELS[model_name], **common
        )
    if model_name == "knn_classifier":
        return _knn_neighbors(
            X_train=X_train_arr, y_train=y_train_arr, **common
        )
    if model_name == "gaussian_nb":
        return _gaussian_nb_likelihood(**common)
    if model_name == "decision_tree_regressor":
        return _decision_path(**common)
    if model_name == "mlp_classifier":
        # Permutation importance compares against ``y_test`` only; the
        # estimator's recorded ``y_pred`` is not used here.
        return _permutation_importance(
            model_name=model_name,
            estimator=estimator,
            X_test=X_test_arr,
            y_test=y_test_arr,
            feature_names=names,
            random_state=random_state,
        )
    if model_name in _PERMUTATION_REGRESSORS:
        return _permutation_importance_mse(
            model_name=model_name,
            estimator=estimator,
            X_test=X_test_arr,
            y_test=y_test_arr,
            feature_names=names,
            random_state=random_state,
        )
    return None


_SUPPORTED_MODELS = frozenset(
    {
        *_LINEAR_REGRESSION_MODELS,
        *_MULTICLASS_LINEAR_MODELS,
        *_PERMUTATION_REGRESSORS,
        "knn_classifier",
        "gaussian_nb",
        "decision_tree_regressor",
        "mlp_classifier",
    }
)
