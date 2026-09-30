"""Tests for NumPy batch-sub-gradient-descent One-vs-Rest linear SVM."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ml_engine.models import LinearSVM


@pytest.fixture(scope="module")
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated 2D Gaussian blobs, labels 0/1."""
    rng = np.random.default_rng(7)
    centers = np.array([[-2.5, -2.5], [2.5, -2.5]])
    y = rng.integers(0, 2, size=200)
    X = rng.standard_normal((200, 2)) + centers[y]
    return X, y


@pytest.fixture(scope="module")
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated 2D Gaussian blobs, labels 0/1/2."""
    rng = np.random.default_rng(42)
    centers = np.array([[-3.0, -3.0], [3.0, -3.0], [0.0, 3.0]])
    y = rng.integers(0, 3, size=300)
    X = rng.standard_normal((300, 2)) + centers[y]
    return X, y


# ---------------------------------------------------------------------------
# Core training and prediction
# ---------------------------------------------------------------------------


def test_fit_returns_self(binary_data) -> None:
    X, y = binary_data
    model = LinearSVM(learning_rate=0.01, max_iter=500)
    assert model.fit(X, y) is model


def test_binary_training_and_accuracy(binary_data) -> None:
    X, y = binary_data
    model = LinearSVM(learning_rate=0.01, max_iter=5000, tol=1e-10).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.95


def test_multiclass_training_and_accuracy(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=5000, tol=1e-10).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.95


def test_decision_function_shape_and_finiteness(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=500, tol=0.0).fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (X.shape[0], 3)
    assert np.isfinite(scores).all()


def test_predict_agrees_with_decision_function_argmax(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=2000, tol=1e-10).fit(X, y)
    scores = model.decision_function(X)
    np.testing.assert_array_equal(
        model.classes_[np.argmax(scores, axis=1)], model.predict(X)
    )


# ---------------------------------------------------------------------------
# Label types
# ---------------------------------------------------------------------------


def test_non_contiguous_integer_labels(multi_data) -> None:
    X, codes = multi_data
    labels = np.array([-1, 5, 9])
    y = labels[codes]
    model = LinearSVM(learning_rate=0.01, max_iter=2000, tol=1e-10).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array([-1, 5, 9]))
    assert np.mean(model.predict(X) == y) > 0.95


def test_string_labels(multi_data) -> None:
    X, codes = multi_data
    names = np.array(["bird", "cat", "dog"])
    y = names[codes]
    model = LinearSVM(learning_rate=0.01, max_iter=2000, tol=1e-10).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array(["bird", "cat", "dog"]))
    predictions = model.predict(X)
    assert set(np.unique(predictions)).issubset({"bird", "cat", "dog"})
    assert np.mean(predictions == y) > 0.95


def test_object_dtype_string_labels(multi_data) -> None:
    X, codes = multi_data
    names = np.array(["bird", "cat", "dog"], dtype=object)
    y = names[codes]
    assert y.dtype == object
    model = LinearSVM(learning_rate=0.01, max_iter=2000, tol=1e-10).fit(X, y)
    np.testing.assert_array_equal(
        model.classes_, np.array(["bird", "cat", "dog"], dtype=object)
    )
    assert np.mean(model.predict(X) == y) > 0.95


# ---------------------------------------------------------------------------
# Attribute shapes
# ---------------------------------------------------------------------------


def test_attribute_shapes(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=500, tol=0.0).fit(X, y)
    assert model.classes_.shape == (3,)
    assert model.coef_.shape == (3, 2)
    assert model.intercept_.shape == (3,)
    assert model.n_features_in_ == 2


# ---------------------------------------------------------------------------
# sklearn reference comparison
# ---------------------------------------------------------------------------


def test_sklearn_reference_comparison(multi_data) -> None:
    # sklearn is allowed in tests only; used as an independent reference.
    from sklearn.svm import LinearSVC

    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=5000, tol=1e-10, C=1.0).fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sklearn_model = LinearSVC(C=1.0, max_iter=10000, dual="auto").fit(X, y)

    our_predictions = model.predict(X)
    sklearn_predictions = sklearn_model.predict(X)
    assert np.mean(our_predictions == y) > 0.95
    assert np.mean(sklearn_predictions == y) > 0.95
    assert np.mean(our_predictions == sklearn_predictions) > 0.9


# ---------------------------------------------------------------------------
# Gradient and objective verification
# ---------------------------------------------------------------------------


def test_first_subgradient_step_matches_formula() -> None:
    # At W = 0, b = 0 all margins equal 1 > 0, so active is everywhere True.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([0, 1, 0, 2])
    n_samples, n_features = X.shape
    n_classes = 3
    C = 1.0
    learning_rate = 0.01
    model = LinearSVM(
        learning_rate=learning_rate, max_iter=1, tol=0.0, C=C
    ).fit(X, y)

    Y = np.full((n_samples, n_classes), -1.0)
    Y[np.arange(n_samples), y] = 1.0
    active = np.ones((n_samples, n_classes), dtype=bool)
    expected_W = learning_rate * (C / n_samples) * (active * Y).T @ X
    expected_b = learning_rate * (C / n_samples) * np.sum(active * Y, axis=0)

    np.testing.assert_allclose(model.coef_, expected_W)
    np.testing.assert_allclose(model.intercept_, expected_b)
    assert model.n_iter_ == 1
    assert model.stop_reason_ == "max_iter"
    assert len(model.loss_history_) == 1


def test_C_affects_model_results(multi_data) -> None:
    X, y = multi_data
    small_C = LinearSVM(
        C=0.01, learning_rate=0.01, max_iter=3000, tol=0.0
    ).fit(X, y)
    large_C = LinearSVM(
        C=10.0, learning_rate=0.01, max_iter=3000, tol=0.0
    ).fit(X, y)
    # Larger C → weaker regularization → larger weight norm.
    assert np.linalg.norm(large_C.coef_) > np.linalg.norm(small_C.coef_)


def test_intercept_not_regularized_in_objective() -> None:
    # Verify the objective formula: only W is regularized, not b.
    # Construct a tiny problem, compute loss manually, compare to loss_history_.
    rng = np.random.default_rng(5)
    X = rng.standard_normal((20, 2))
    y = rng.integers(0, 2, size=20)
    C = 2.0
    model = LinearSVM(learning_rate=0.01, max_iter=1, tol=0.0, C=C).fit(X, y)

    n = X.shape[0]
    Y = np.full((n, 2), -1.0)
    Y[np.arange(n), y] = 1.0
    scores = X @ model.coef_.T + model.intercept_
    margins = 1.0 - Y * scores
    manual_loss = 0.5 * np.sum(model.coef_ ** 2) + (C / n) * np.sum(
        np.maximum(0.0, margins)
    )
    np.testing.assert_allclose(model.loss_history_[0], manual_loss, atol=1e-10)


# ---------------------------------------------------------------------------
# Loss history and stopping consistency
# ---------------------------------------------------------------------------


def test_max_iter_stop_state_is_consistent(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=5, tol=0.0).fit(X, y)
    assert model.stop_reason_ == "max_iter"
    assert model.n_iter_ == 5
    assert len(model.loss_history_) == 5


def test_converged_stop_state_is_consistent(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=10000, tol=1e-10).fit(X, y)
    assert model.stop_reason_ == "converged"
    assert model.n_iter_ < 10000
    assert len(model.loss_history_) == model.n_iter_
    assert abs(model.loss_history_[-1] - model.loss_history_[-2]) <= 1e-10


def test_loss_history_length_equals_n_iter(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=100, tol=0.0).fit(X, y)
    assert len(model.loss_history_) == model.n_iter_


# ---------------------------------------------------------------------------
# Refit semantics
# ---------------------------------------------------------------------------


def test_repeated_successful_fit_resets_state(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=2000, tol=1e-10).fit(X, y)
    old_history = model.loss_history_

    rng = np.random.default_rng(11)
    X_new = rng.standard_normal((80, 4))
    y_new = rng.integers(0, 2, size=80)
    model.fit(X_new, y_new)

    assert model.loss_history_ is not old_history
    assert len(model.loss_history_) == model.n_iter_
    assert model.n_iter_ <= model.max_iter
    assert model.n_features_in_ == 4
    assert model.coef_.shape == (2, 4)
    assert model.intercept_.shape == (2,)


def test_failed_refit_preserves_last_successful_state(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=2000, tol=1e-10).fit(X, y)
    old_classes = model.classes_.copy()
    old_coef = model.coef_.copy()
    old_intercept = model.intercept_.copy()
    old_history = model.loss_history_.copy()
    old_n_iter = model.n_iter_
    old_n_features = model.n_features_in_

    bad_y = y.astype(float)
    bad_y[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(X, bad_y)

    np.testing.assert_array_equal(model.classes_, old_classes)
    np.testing.assert_array_equal(model.coef_, old_coef)
    np.testing.assert_array_equal(model.intercept_, old_intercept)
    assert model.loss_history_ == old_history
    assert model.n_iter_ == old_n_iter
    assert model.n_features_in_ == old_n_features
    assert model.predict(X[:4]).shape == (4,)


# ---------------------------------------------------------------------------
# Pre-fit and feature-mismatch errors
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises() -> None:
    model = LinearSVM()
    X = np.zeros((2, 2))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.decision_function(X)


def test_feature_count_mismatch_raises(binary_data) -> None:
    X, y = binary_data
    model = LinearSVM(learning_rate=0.01, max_iter=100).fit(X, y)
    with pytest.raises(ValueError, match="feature"):
        model.predict(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="feature"):
        model.decision_function(np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=int)),
        (np.zeros((2, 2, 2)), np.zeros(2, dtype=int)),
        (np.zeros((0, 2)), np.zeros(0, dtype=int)),
    ],
)
def test_invalid_X_shape_raises(X, y) -> None:
    with pytest.raises(ValueError):
        LinearSVM().fit(X, y)


def test_y_must_be_one_dimensional(multi_data) -> None:
    X, _ = multi_data
    with pytest.raises(ValueError, match="y must be a 1D"):
        LinearSVM().fit(X, np.zeros((X.shape[0], 1), dtype=int))


def test_sample_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        LinearSVM().fit(np.zeros((4, 2)), np.zeros(3, dtype=int))


@pytest.mark.parametrize(
    "X, y, match",
    [
        (np.array([[1.0, np.nan], [0.0, 1.0]]), np.array([0, 1]), "NaN or infinite"),
        (np.array([[1.0, np.inf], [0.0, 1.0]]), np.array([0, 1]), "NaN or infinite"),
        (np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([np.nan, 1.0]), "NaN or infinite"),
    ],
)
def test_non_finite_inputs_raise(X, y, match) -> None:
    with pytest.raises(ValueError, match=match):
        LinearSVM().fit(X, y)


@pytest.mark.parametrize(
    "bad_y",
    [
        np.array([0.0, np.nan, 1.0, 1.0], dtype=object),
        np.array([0.0, np.inf, 1.0, 1.0], dtype=object),
    ],
)
def test_object_dtype_non_finite_labels_raise(bad_y) -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        LinearSVM(learning_rate=0.01, max_iter=10).fit(X, bad_y)


def test_single_class_raises() -> None:
    with pytest.raises(ValueError, match="at least two distinct classes"):
        LinearSVM().fit(np.array([[1.0], [2.0]]), np.array([5, 5]))


def test_mixed_incomparable_object_labels_raise_clear_error() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    y = np.array(["a", 1.0, "b", 1.0], dtype=object)
    with pytest.raises(ValueError, match="incomparable label types"):
        LinearSVM(learning_rate=0.01, max_iter=10).fit(X, y)


# ---------------------------------------------------------------------------
# Hyperparameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": -0.2}, "learning_rate"),
        ({"learning_rate": np.nan}, "learning_rate"),
        ({"learning_rate": np.inf}, "learning_rate"),
        ({"learning_rate": True}, "learning_rate"),
        ({"learning_rate": "fast"}, "learning_rate"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": -5}, "max_iter"),
        ({"max_iter": 2.5}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"max_iter": "many"}, "max_iter"),
        ({"tol": -1e-9}, "tol"),
        ({"tol": np.nan}, "tol"),
        ({"tol": np.inf}, "tol"),
        ({"tol": True}, "tol"),
        ({"tol": "tiny"}, "tol"),
        ({"C": 0.0}, "C"),
        ({"C": -1.0}, "C"),
        ({"C": np.nan}, "C"),
        ({"C": np.inf}, "C"),
        ({"C": True}, "C"),
        ({"C": "strong"}, "C"),
    ],
)
def test_invalid_hyperparameters_raise(kwargs, match) -> None:
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises((ValueError, TypeError), match=match):
        LinearSVM(**kwargs).fit(X, y)


# ---------------------------------------------------------------------------
# Divergence and numerical safety
# ---------------------------------------------------------------------------


def test_divergence_raises_floating_point_error(multi_data) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=1e305, max_iter=500, tol=0.0)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        model.fit(X, y)
    assert not hasattr(model, "coef_")


def test_prediction_overflow_with_finite_X_raises(multi_data) -> None:
    X, y = multi_data
    # Large C produces larger coefficients so that 1e308 @ coef_.T overflows.
    model = LinearSVM(learning_rate=0.01, max_iter=2000, tol=1e-10, C=1000.0).fit(X, y)

    # 1e308 is finite itself, but X @ coef_.T overflows to +/-inf.
    X_huge = np.full((5, X.shape[1]), 1e308)
    assert np.isfinite(X_huge).all()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        model.decision_function(X_huge)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        model.predict(X_huge)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_predict_non_finite_input_raises(multi_data, bad_value) -> None:
    X, y = multi_data
    model = LinearSVM(learning_rate=0.01, max_iter=2000, tol=1e-10).fit(X, y)
    X_bad = X[:3].copy()
    X_bad[0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.predict(X_bad)
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.decision_function(X_bad)
