"""Tests for NumPy batch-gradient-descent softmax logistic regression."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ml_engine.models import LogisticRegressionGD


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


@pytest.fixture(scope="module")
def hard_multi_data() -> tuple[np.ndarray, np.ndarray]:
    """Three overlapping 2D blobs; the problem is not perfectly separable."""
    rng = np.random.default_rng(99)
    centers = np.array([[-1.5, 0.0], [1.5, 0.0], [0.0, 1.5]])
    y = rng.integers(0, 3, size=600)
    X = rng.standard_normal((600, 2)) + centers[y]
    return X, y


def test_binary_training_and_prediction(binary_data) -> None:
    X, y = binary_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)

    assert model.n_features_in_ == 2
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))
    assert np.mean(model.predict(X) == y) > 0.95


def test_multiclass_training(multi_data) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)

    np.testing.assert_array_equal(model.classes_, np.array([0, 1, 2]))
    assert np.mean(model.predict(X) == y) > 0.95


@pytest.mark.parametrize(
    "fixture_name,n_classes",
    [("binary_data", 2), ("multi_data", 3)],
)
def test_predict_proba_shape_range_and_row_sum(request, fixture_name, n_classes) -> None:
    X, y = request.getfixturevalue(fixture_name)
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (X.shape[0], n_classes)
    assert np.isfinite(probabilities).all()
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(X.shape[0]), atol=1e-8)


def test_predict_returns_original_string_and_int_labels(multi_data) -> None:
    X, codes = multi_data
    string_names = np.array(["bird", "cat", "dog"])
    y_string = string_names[codes]

    model_string = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10)
    model_string.fit(X, y_string)
    predictions_string = model_string.predict(X)
    assert np.issubdtype(predictions_string.dtype, np.str_)
    assert set(np.unique(predictions_string)).issubset(set(string_names))
    assert np.mean(predictions_string == y_string) > 0.95

    int_labels = np.array([-1, 5, 9])
    y_int = int_labels[codes]
    model_int = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y_int)
    np.testing.assert_array_equal(model_int.classes_, np.array([-1, 5, 9]))
    predictions_int = model_int.predict(X)
    assert set(np.unique(predictions_int)).issubset({-1, 5, 9})
    assert np.mean(predictions_int == y_int) > 0.95


def test_coef_and_intercept_shapes(multi_data) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=500, tol=0.0).fit(X, y)

    assert model.coef_.shape == (3, 2)
    assert model.intercept_.shape == (3,)
    assert model.classes_.shape == (3,)
    assert model.n_features_in_ == 2


def test_sklearn_reference_comparison(hard_multi_data) -> None:
    # sklearn is allowed in tests only; used as an independent reference.
    from sklearn.linear_model import LogisticRegression

    X, y = hard_multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=20000, tol=1e-11).fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sklearn_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000).fit(X, y)

    our_predictions = model.predict(X)
    sklearn_predictions = sklearn_model.predict(X)
    assert np.mean(our_predictions == y) >= 0.72
    assert np.mean(sklearn_predictions == y) >= 0.72
    assert np.mean(our_predictions == sklearn_predictions) >= 0.95


def test_larger_alpha_reduces_weight_norm(multi_data) -> None:
    X, y = multi_data
    unregularized = LogisticRegressionGD(
        alpha=0.0, learning_rate=0.1, max_iter=3000, tol=0.0
    ).fit(X, y)
    regularized = LogisticRegressionGD(
        alpha=10.0, learning_rate=0.1, max_iter=3000, tol=0.0
    ).fit(X, y)

    assert np.linalg.norm(regularized.coef_) < np.linalg.norm(unregularized.coef_)


def test_first_gradient_step_matches_formula() -> None:
    # At W = 0, b = 0 the softmax output is uniform: P = 1/K.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0, 0, 1, 2])
    n_samples, n_features = X.shape
    model = LogisticRegressionGD(
        learning_rate=0.2, max_iter=1, tol=0.0, alpha=0.0
    ).fit(X, y)

    n_classes = 3
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y] = 1.0
    delta = one_hot - 1.0 / n_classes
    expected_W = 0.2 * (delta / n_samples).T @ X
    expected_b = 0.2 * (delta / n_samples).sum(axis=0)

    np.testing.assert_allclose(model.coef_, expected_W)
    np.testing.assert_allclose(model.intercept_, expected_b)
    assert model.n_iter_ == 1
    assert model.stop_reason_ == "max_iter"
    assert len(model.loss_history_) == 1


def test_numerical_stability_with_large_inputs(multi_data) -> None:
    X, y = multi_data
    X_large = X * 100.0
    model = LogisticRegressionGD(
        learning_rate=1e-4, max_iter=5000, tol=1e-12
    ).fit(X_large, y)

    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.intercept_).all()
    assert np.isfinite(model.loss_history_).all()
    probabilities = model.predict_proba(X_large)
    assert np.isfinite(probabilities).all()
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(X.shape[0]), atol=1e-8)
    assert np.mean(model.predict(X_large) == y) > 0.9


def test_repeated_successful_fit_resets_state(multi_data) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)
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
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)
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


def test_predict_before_fit_raises() -> None:
    model = LogisticRegressionGD()
    X = np.zeros((2, 2))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(X)


def test_feature_count_mismatch_raises(binary_data) -> None:
    X, y = binary_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=100).fit(X, y)
    with pytest.raises(ValueError, match="feature"):
        model.predict(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="feature"):
        model.predict_proba(np.zeros((5, 3)))


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=int)),  # 1D X
        (np.zeros((2, 2, 2)), np.zeros(2, dtype=int)),  # 3D X
        (np.zeros((0, 2)), np.zeros(0, dtype=int)),  # empty X
    ],
)
def test_invalid_X_shape_raises(X, y) -> None:
    with pytest.raises(ValueError):
        LogisticRegressionGD().fit(X, y)


def test_y_must_be_one_dimensional(multi_data) -> None:
    X, _ = multi_data
    with pytest.raises(ValueError, match="y must be a 1D"):
        LogisticRegressionGD().fit(X, np.zeros((X.shape[0], 1), dtype=int))


def test_sample_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        LogisticRegressionGD().fit(np.zeros((4, 2)), np.zeros(3, dtype=int))


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
        LogisticRegressionGD().fit(X, y)


def test_single_class_raises() -> None:
    with pytest.raises(ValueError, match="at least two distinct classes"):
        LogisticRegressionGD().fit(np.array([[1.0], [2.0]]), np.array([5, 5]))


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
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": np.nan}, "alpha"),
        ({"alpha": np.inf}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"alpha": "strong"}, "alpha"),
    ],
)
def test_invalid_hyperparameters_raise(kwargs, match) -> None:
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises((ValueError, TypeError), match=match):
        LogisticRegressionGD(**kwargs).fit(X, y)


def test_fit_returns_self(binary_data) -> None:
    X, y = binary_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=100)
    assert model.fit(X, y) is model


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
        LogisticRegressionGD(learning_rate=0.1, max_iter=10).fit(X, bad_y)


def test_object_dtype_string_labels_train_and_predict(multi_data) -> None:
    X, codes = multi_data
    names = np.array(["bird", "cat", "dog"], dtype=object)
    y = names[codes]
    assert y.dtype == object

    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array(["bird", "cat", "dog"], dtype=object))
    predictions = model.predict(X)
    assert predictions.dtype == object
    assert set(np.unique(predictions)).issubset({"bird", "cat", "dog"})
    assert np.mean(predictions == y) > 0.95


def test_mixed_incomparable_object_labels_raise_clear_error() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    y = np.array(["a", 1.0, "b", 1.0], dtype=object)
    with pytest.raises(ValueError, match="incomparable label types"):
        LogisticRegressionGD(learning_rate=0.1, max_iter=10).fit(X, y)


def test_prediction_overflow_with_finite_X_raises(multi_data) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)

    # 1e308 is finite itself, but X @ coef_.T overflows to +/-inf.
    X_huge = np.full((5, X.shape[1]), 1e308)
    assert np.isfinite(X_huge).all()
    with pytest.raises(FloatingPointError, match="Non-finite logits"):
        model.predict_proba(X_huge)
    with pytest.raises(FloatingPointError, match="Non-finite logits"):
        model.predict(X_huge)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_predict_non_finite_input_raises(multi_data, bad_value) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)
    X_bad = X[:3].copy()
    X_bad[0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.predict(X_bad)
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.predict_proba(X_bad)


def test_max_iter_stop_state_is_consistent(multi_data) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=5, tol=0.0).fit(X, y)

    assert model.stop_reason_ == "max_iter"
    assert model.n_iter_ == 5
    assert len(model.loss_history_) == 5


def test_converged_stop_state_is_consistent(hard_multi_data) -> None:
    X, y = hard_multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=20000, tol=1e-11).fit(X, y)

    assert model.stop_reason_ == "converged"
    assert model.n_iter_ < 20000
    assert len(model.loss_history_) == model.n_iter_
    assert abs(model.loss_history_[-1] - model.loss_history_[-2]) <= 1e-11


def test_early_losses_strictly_decrease(multi_data) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=3, tol=0.0).fit(X, y)

    assert model.loss_history_[1] < model.loss_history_[0]
    assert model.loss_history_[2] < model.loss_history_[1]


def test_predict_agrees_with_predict_proba_argmax(multi_data) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)

    probabilities = model.predict_proba(X)
    argmax_labels = model.classes_[np.argmax(probabilities, axis=1)]
    np.testing.assert_array_equal(argmax_labels, model.predict(X))


def test_divergence_raises_floating_point_error(multi_data) -> None:
    X, y = multi_data
    model = LogisticRegressionGD(learning_rate=1e305, max_iter=500, tol=0.0)

    with pytest.raises(FloatingPointError, match="Non-finite"):
        model.fit(X, y)
    assert not hasattr(model, "coef_")
    assert not hasattr(model, "classes_")
