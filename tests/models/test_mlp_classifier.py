"""Tests for the NumPy single-hidden-layer MLP classifier."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ml_engine.models import MLPClassifier

# ---------------------------------------------------------------------------
# Module-scope fixtures for expensive training results
# ---------------------------------------------------------------------------


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
def xor_data() -> tuple[np.ndarray, np.ndarray]:
    """XOR pattern with Gaussian noise (non-linearly-separable)."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 1, 1, 0])
    idx = rng.integers(0, 4, size=240)
    X = rng.standard_normal((240, 2)) * 0.18 + centers[idx]
    y = labels[idx]
    return X, y


@pytest.fixture(scope="module")
def fitted_binary(binary_data) -> MLPClassifier:
    return MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=3000, tol=1e-9, random_state=42
    ).fit(*binary_data)


@pytest.fixture(scope="module")
def fitted_multi(multi_data) -> MLPClassifier:
    return MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=3000, tol=1e-9, random_state=42
    ).fit(*multi_data)


@pytest.fixture(scope="module")
def fitted_xor(xor_data) -> MLPClassifier:
    return MLPClassifier(
        hidden_size=8, learning_rate=0.05, max_iter=5000, tol=1e-9, random_state=42
    ).fit(*xor_data)


# ---------------------------------------------------------------------------
# 1. fit returns self
# ---------------------------------------------------------------------------


def test_fit_returns_self(binary_data) -> None:
    X, y = binary_data
    model = MLPClassifier(hidden_size=8, max_iter=50, random_state=42)
    assert model.fit(X, y) is model


# ---------------------------------------------------------------------------
# 2-3. Binary / multiclass training and accuracy
# ---------------------------------------------------------------------------


def test_binary_training_and_accuracy(fitted_binary, binary_data) -> None:
    X, y = binary_data
    assert np.mean(fitted_binary.predict(X) == y) > 0.95


def test_multiclass_training_and_accuracy(fitted_multi, multi_data) -> None:
    X, y = multi_data
    assert np.mean(fitted_multi.predict(X) == y) > 0.95


# ---------------------------------------------------------------------------
# 4. XOR non-linear classification
# ---------------------------------------------------------------------------


def test_xor_nonlinear_classification(fitted_xor, xor_data) -> None:
    X, y = xor_data
    # A linear classifier cannot solve XOR above ~0.5 accuracy.
    assert np.mean(fitted_xor.predict(X) == y) > 0.9


# ---------------------------------------------------------------------------
# 5. predict_proba shape / finiteness / range / row-sum
# ---------------------------------------------------------------------------


def test_predict_proba_properties(fitted_multi, multi_data) -> None:
    X, y = multi_data
    proba = fitted_multi.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.isfinite(proba).all()
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(X.shape[0]), atol=1e-12)


# ---------------------------------------------------------------------------
# 6. predict agrees with predict_proba argmax
# ---------------------------------------------------------------------------


def test_predict_matches_proba_argmax(fitted_multi, multi_data) -> None:
    X, _ = multi_data
    proba = fitted_multi.predict_proba(X)
    expected = fitted_multi.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(fitted_multi.predict(X), expected)


# ---------------------------------------------------------------------------
# 7-9. Label types
# ---------------------------------------------------------------------------


def test_non_contiguous_integer_labels(multi_data) -> None:
    X, codes = multi_data
    labels = np.array([-1, 5, 9])
    y = labels[codes]
    model = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=2000, tol=1e-9, random_state=42
    ).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array([-1, 5, 9]))
    assert np.mean(model.predict(X) == y) > 0.95


def test_string_labels(multi_data) -> None:
    X, codes = multi_data
    names = np.array(["bird", "cat", "dog"])
    y = names[codes]
    model = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=2000, tol=1e-9, random_state=42
    ).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array(["bird", "cat", "dog"]))
    assert np.mean(model.predict(X) == y) > 0.95


def test_object_dtype_string_labels(multi_data) -> None:
    X, codes = multi_data
    names = np.array(["bird", "cat", "dog"], dtype=object)
    y = names[codes]
    assert y.dtype == object
    model = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=2000, tol=1e-9, random_state=42
    ).fit(X, y)
    np.testing.assert_array_equal(
        model.classes_, np.array(["bird", "cat", "dog"], dtype=object)
    )
    assert np.mean(model.predict(X) == y) > 0.95


# ---------------------------------------------------------------------------
# 10. Attribute shapes
# ---------------------------------------------------------------------------


def test_attribute_shapes(fitted_multi, multi_data) -> None:
    _, y = multi_data
    model = fitted_multi
    assert model.classes_.shape == (3,)
    assert model.coefs_[0].shape == (2, 16)
    assert model.coefs_[1].shape == (16, 3)
    assert model.intercepts_[0].shape == (16,)
    assert model.intercepts_[1].shape == (3,)
    assert model.n_features_in_ == 2
    assert model.n_classes_ == 3
    assert model.hidden_size_ == 16


# ---------------------------------------------------------------------------
# 11. Same random_state reproduces parameters and predictions
# ---------------------------------------------------------------------------


def test_same_random_state_reproduces_results(multi_data) -> None:
    X, y = multi_data
    a = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=500, tol=0.0, random_state=42
    ).fit(X, y)
    b = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=500, tol=0.0, random_state=42
    ).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_allclose(a.coefs_[0], b.coefs_[0])
    np.testing.assert_allclose(a.coefs_[1], b.coefs_[1])
    np.testing.assert_allclose(a.intercepts_[0], b.intercepts_[0])
    np.testing.assert_allclose(a.intercepts_[1], b.intercepts_[1])


# ---------------------------------------------------------------------------
# 12. No pollution of NumPy global random state
# ---------------------------------------------------------------------------


def test_does_not_pollute_global_random_state(multi_data) -> None:
    X, y = multi_data
    np.random.seed(123)
    expected = np.random.random(100)
    np.random.seed(123)
    MLPClassifier(hidden_size=8, max_iter=50, random_state=42).fit(X, y)
    after = np.random.random(100)
    np.testing.assert_array_equal(expected, after)


# ---------------------------------------------------------------------------
# 13. sklearn reference comparison
# ---------------------------------------------------------------------------


def test_sklearn_reference_comparison(multi_data) -> None:
    from sklearn.neural_network import MLPClassifier as SklearnMLP

    X, y = multi_data
    model = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=3000, tol=1e-9,
        alpha=0.0001, random_state=42,
    ).fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sklearn_model = SklearnMLP(
            hidden_layer_sizes=(16,), activation="relu", solver="sgd",
            learning_rate_init=0.05, max_iter=5000, alpha=0.0001,
            random_state=42, batch_size=X.shape[0], momentum=0.0,
            n_iter_no_change=10000, tol=1e-9,
        ).fit(X, y)

    our_pred = model.predict(X)
    sklearn_pred = sklearn_model.predict(X)
    assert np.mean(our_pred == y) > 0.9
    assert np.mean(sklearn_pred == y) > 0.9
    assert np.mean(our_pred == sklearn_pred) > 0.85


# ---------------------------------------------------------------------------
# 14. First gradient step matches manual formula
# ---------------------------------------------------------------------------


def test_first_gradient_step_matches_formula() -> None:
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [0.5, 0.5], [1.5, 0.0]]
    )
    y = np.array([0, 1, 0, 2, 1, 2])
    n_samples, n_features = X.shape
    n_classes = 3
    hidden_size = 4
    learning_rate = 0.01
    alpha = 0.0001
    random_state = 42

    model = MLPClassifier(
        hidden_size=hidden_size, learning_rate=learning_rate,
        max_iter=1, tol=0.0, alpha=alpha, random_state=random_state,
    ).fit(X, y)

    # Reconstruct the exact initialization.
    rng = np.random.default_rng(random_state)
    W1 = rng.standard_normal((n_features, hidden_size)) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(hidden_size)
    W2 = rng.standard_normal((hidden_size, n_classes)) * np.sqrt(1.0 / hidden_size)
    b2 = np.zeros(n_classes)

    classes = np.array([0, 1, 2])
    codes = np.searchsorted(classes, y)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), codes] = 1.0

    # Forward at initial params.
    Z1 = X @ W1 + b1
    H = np.maximum(0.0, Z1)
    logits = H @ W2 + b2
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    P = exp / np.sum(exp, axis=1, keepdims=True)

    # Backprop.
    error = (P - one_hot) / n_samples
    grad_W2 = H.T @ error + alpha * W2
    grad_b2 = np.sum(error, axis=0)
    grad_H = error @ W2.T
    grad_Z1 = grad_H * (Z1 > 0.0)
    grad_W1 = X.T @ grad_Z1 + alpha * W1
    grad_b1 = np.sum(grad_Z1, axis=0)

    expected_W1 = W1 - learning_rate * grad_W1
    expected_b1 = b1 - learning_rate * grad_b1
    expected_W2 = W2 - learning_rate * grad_W2
    expected_b2 = b2 - learning_rate * grad_b2

    np.testing.assert_allclose(model.coefs_[0], expected_W1, atol=1e-12)
    np.testing.assert_allclose(model.coefs_[1], expected_W2, atol=1e-12)
    np.testing.assert_allclose(model.intercepts_[0], expected_b1, atol=1e-12)
    np.testing.assert_allclose(model.intercepts_[1], expected_b2, atol=1e-12)
    assert model.n_iter_ == 1
    assert model.stop_reason_ == "max_iter"
    assert len(model.loss_history_) == 1


# ---------------------------------------------------------------------------
# 15. alpha affects weight norm
# ---------------------------------------------------------------------------


def test_alpha_affects_weight_norm(multi_data) -> None:
    X, y = multi_data
    no_reg = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=2000, tol=0.0,
        alpha=0.0, random_state=42,
    ).fit(X, y)
    strong_reg = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=2000, tol=0.0,
        alpha=1.0, random_state=42,
    ).fit(X, y)
    norm_no = np.sum(no_reg.coefs_[0] ** 2) + np.sum(no_reg.coefs_[1] ** 2)
    norm_strong = np.sum(strong_reg.coefs_[0] ** 2) + np.sum(strong_reg.coefs_[1] ** 2)
    assert norm_strong < norm_no


# ---------------------------------------------------------------------------
# 16. Objective verifies biases are not regularized
# ---------------------------------------------------------------------------


def test_biases_not_regularized_in_objective() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((20, 2))
    y = rng.integers(0, 2, size=20)
    alpha = 0.1
    model = MLPClassifier(
        hidden_size=8, learning_rate=0.01, max_iter=1, tol=0.0,
        alpha=alpha, random_state=42,
    ).fit(X, y)

    W1, W2 = model.coefs_
    b1, b2 = model.intercepts_
    codes = np.searchsorted(model.classes_, y)

    # Manual stable log-softmax to avoid exp overflow.
    Z1 = X @ W1 + b1
    H = np.maximum(0.0, Z1)
    logits = H @ W2 + b2
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_prob = shifted - log_denom
    cross_entropy = -np.mean(log_prob[np.arange(20), codes])

    # Only W1 and W2 enter the regularization term; b1 and b2 do not.
    reg = alpha * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) / 2.0
    manual_loss = cross_entropy + reg

    np.testing.assert_allclose(model.loss_history_[0], manual_loss, atol=1e-10)


# ---------------------------------------------------------------------------
# 17-19. Stopping states and loss history length
# ---------------------------------------------------------------------------


def test_max_iter_stop_state(multi_data) -> None:
    X, y = multi_data
    model = MLPClassifier(
        hidden_size=8, learning_rate=0.01, max_iter=5, tol=0.0, random_state=42
    ).fit(X, y)
    assert model.stop_reason_ == "max_iter"
    assert model.n_iter_ == 5
    assert len(model.loss_history_) == 5


def test_converged_stop_state(binary_data) -> None:
    X, y = binary_data
    model = MLPClassifier(
        hidden_size=8, learning_rate=0.1, max_iter=20000, tol=1e-7, random_state=42
    ).fit(X, y)
    assert model.stop_reason_ == "converged"
    assert model.n_iter_ < 20000
    assert len(model.loss_history_) == model.n_iter_
    assert abs(model.loss_history_[-1] - model.loss_history_[-2]) <= 1e-7


def test_loss_history_length_equals_n_iter(multi_data) -> None:
    X, y = multi_data
    model = MLPClassifier(
        hidden_size=8, learning_rate=0.01, max_iter=100, tol=0.0, random_state=42
    ).fit(X, y)
    assert len(model.loss_history_) == model.n_iter_


# ---------------------------------------------------------------------------
# 20-21. Refit semantics (must use independent models)
# ---------------------------------------------------------------------------


def test_successful_refit_replaces_lists(multi_data) -> None:
    X, y = multi_data
    model = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=500, tol=0.0, random_state=42
    ).fit(X, y)
    old_coefs = model.coefs_
    old_intercepts = model.intercepts_
    old_history = model.loss_history_

    rng = np.random.default_rng(11)
    X_new = rng.standard_normal((80, 4))
    y_new = rng.integers(0, 2, size=80)
    model.fit(X_new, y_new)

    assert model.coefs_ is not old_coefs
    assert model.intercepts_ is not old_intercepts
    assert model.loss_history_ is not old_history
    assert model.coefs_[0].shape == (4, 16)
    assert model.coefs_[1].shape == (16, 2)
    assert model.intercepts_[0].shape == (16,)
    assert model.intercepts_[1].shape == (2,)
    assert len(model.loss_history_) == model.n_iter_


def test_failed_refit_preserves_state(multi_data) -> None:
    X, y = multi_data
    model = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=500, tol=1e-9, random_state=42
    ).fit(X, y)
    old_classes = model.classes_.copy()
    old_coefs = [c.copy() for c in model.coefs_]
    old_intercepts = [c.copy() for c in model.intercepts_]
    old_history = list(model.loss_history_)
    old_n_iter = model.n_iter_
    old_n_features = model.n_features_in_

    bad_y = y.astype(float)
    bad_y[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(X, bad_y)

    np.testing.assert_array_equal(model.classes_, old_classes)
    for got, exp in zip(model.coefs_, old_coefs, strict=True):
        np.testing.assert_array_equal(got, exp)
    for got, exp in zip(model.intercepts_, old_intercepts, strict=True):
        np.testing.assert_array_equal(got, exp)
    assert model.loss_history_ == old_history
    assert model.n_iter_ == old_n_iter
    assert model.n_features_in_ == old_n_features
    assert model.predict(X[:4]).shape == (4,)


# ---------------------------------------------------------------------------
# 22-23. Pre-fit and feature-mismatch errors
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises() -> None:
    model = MLPClassifier()
    X = np.zeros((2, 2))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(X)


def test_feature_count_mismatch_raises(fitted_binary) -> None:
    with pytest.raises(ValueError, match="feature"):
        fitted_binary.predict(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="feature"):
        fitted_binary.predict_proba(np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# 24-28. Input shape / dimension / sample-count errors
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
        MLPClassifier().fit(X, y)


def test_zero_features_raises() -> None:
    with pytest.raises(ValueError, match="feature"):
        MLPClassifier().fit(np.zeros((4, 0)), np.zeros(4, dtype=int))


def test_y_must_be_one_dimensional(multi_data) -> None:
    X, _ = multi_data
    with pytest.raises(ValueError, match="y must be a 1D"):
        MLPClassifier().fit(X, np.zeros((X.shape[0], 1), dtype=int))


def test_sample_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        MLPClassifier().fit(np.zeros((4, 2)), np.zeros(3, dtype=int))


# ---------------------------------------------------------------------------
# 29-30. Non-finite X / y / object-dtype labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "X, y, match",
    [
        (np.array([[1.0, np.nan], [0.0, 1.0]]), np.array([0, 1]), "NaN or infinite"),
        (np.array([[1.0, np.inf], [0.0, 1.0]]), np.array([0, 1]), "NaN or infinite"),
        (np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([np.nan, 1.0]), "NaN or infinite"),
        (np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([np.inf, 1.0]), "NaN or infinite"),
    ],
)
def test_non_finite_inputs_raise(X, y, match) -> None:
    with pytest.raises(ValueError, match=match):
        MLPClassifier().fit(X, y)


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
        MLPClassifier(learning_rate=0.01, max_iter=10).fit(X, bad_y)


# ---------------------------------------------------------------------------
# 31-32. Single class / mixed incomparable labels
# ---------------------------------------------------------------------------


def test_single_class_raises() -> None:
    with pytest.raises(ValueError, match="at least two distinct classes"):
        MLPClassifier().fit(np.array([[1.0], [2.0]]), np.array([5, 5]))


def test_mixed_incomparable_object_labels_raise_clear_error() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    y = np.array(["a", 1.0, "b", 1.0], dtype=object)
    with pytest.raises(ValueError, match="incomparable label types"):
        MLPClassifier(learning_rate=0.01, max_iter=10).fit(X, y)


# ---------------------------------------------------------------------------
# 33. Hyperparameter boundary validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"hidden_size": -5}, "hidden_size"),
        ({"hidden_size": 2.5}, "hidden_size"),
        ({"hidden_size": True}, "hidden_size"),
        ({"hidden_size": "32"}, "hidden_size"),
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
        ({"alpha": -1.0}, "alpha"),
        ({"alpha": np.nan}, "alpha"),
        ({"alpha": np.inf}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"alpha": "strong"}, "alpha"),
        ({"random_state": 1.5}, "random_state"),
        ({"random_state": True}, "random_state"),
        ({"random_state": "seed"}, "random_state"),
        ({"random_state": -1}, "random_state"),
    ],
)
def test_invalid_hyperparameters_raise(kwargs, match) -> None:
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises((ValueError, TypeError), match=match):
        MLPClassifier(**kwargs).fit(X, y)


def test_negative_random_state_raises_clear_error() -> None:
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="random_state"):
        MLPClassifier(random_state=-1).fit(X, y)


# ---------------------------------------------------------------------------
# 34. Divergence protection
# ---------------------------------------------------------------------------


def test_divergence_raises_floating_point_error(multi_data) -> None:
    X, y = multi_data
    model = MLPClassifier(
        hidden_size=8, learning_rate=1e10, max_iter=200, tol=0.0, random_state=42
    )
    with pytest.raises(FloatingPointError, match="Non-finite"):
        model.fit(X, y)
    assert not hasattr(model, "coefs_")


def test_initial_forward_overflow_raises_without_runtime_warning() -> None:
    # 1e308 is finite, but the initial X @ W1 overflows during the very first
    # forward pass (before any gradient step), so the error must surface at
    # initialization without leaking a RuntimeWarning.
    X = np.full((4, 2), 1e308)
    y = np.array([0, 1, 0, 1])
    assert np.isfinite(X).all()
    model = MLPClassifier(
        hidden_size=32, learning_rate=0.01, max_iter=10, random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(FloatingPointError, match="initialization"):
            model.fit(X, y)
    assert not hasattr(model, "coefs_")


def test_initial_forward_overflow_preserves_existing_state(multi_data) -> None:
    # An already-fitted model that hits an initialization overflow on a refit
    # must keep its previous successful state intact.
    X, y = multi_data
    fitted = MLPClassifier(
        hidden_size=16, learning_rate=0.05, max_iter=200, tol=0.0, random_state=42
    ).fit(X, y)
    old_coefs = [c.copy() for c in fitted.coefs_]
    old_intercepts = [c.copy() for c in fitted.intercepts_]
    old_n_iter = fitted.n_iter_

    X_huge = np.full((4, 2), 1e308)
    y_huge = np.array([0, 1, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(FloatingPointError, match="initialization"):
            fitted.fit(X_huge, y_huge)

    for got, exp in zip(fitted.coefs_, old_coefs, strict=True):
        np.testing.assert_array_equal(got, exp)
    for got, exp in zip(fitted.intercepts_, old_intercepts, strict=True):
        np.testing.assert_array_equal(got, exp)
    assert fitted.n_iter_ == old_n_iter


# ---------------------------------------------------------------------------
# 35. Extreme finite prediction input overflows matmul
# ---------------------------------------------------------------------------


def test_prediction_overflow_with_finite_X_raises(fitted_multi) -> None:
    # 1e308 is finite, but X @ W1 overflows to +/-inf for trained coefficients.
    X_huge = np.full((5, 2), 1e308)
    assert np.isfinite(X_huge).all()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        fitted_multi.predict_proba(X_huge)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        fitted_multi.predict(X_huge)


# ---------------------------------------------------------------------------
# 36. Prediction input with NaN/inf raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_predict_non_finite_input_raises(fitted_multi, bad_value) -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    X_bad = X.copy()
    X_bad[0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN or infinite"):
        fitted_multi.predict(X_bad)
    with pytest.raises(ValueError, match="NaN or infinite"):
        fitted_multi.predict_proba(X_bad)
