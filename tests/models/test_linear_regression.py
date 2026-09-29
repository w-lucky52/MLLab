"""Tests for NumPy batch-gradient-descent linear and ridge regression."""

from __future__ import annotations

import numpy as np
import pytest

from ml_engine.models import LinearRegressionGD, RidgeRegressionGD


@pytest.fixture(scope="module")
def linear_problem() -> tuple[np.ndarray, np.ndarray]:
    """y = 3*x1 - 2*x2 + 1 with two fixed-seed independent features."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0
    return X, y


def test_linear_recovers_known_coefficients(linear_problem) -> None:
    X, y = linear_problem
    model = LinearRegressionGD(learning_rate=0.1, max_iter=5000, tol=1e-10)

    assert model.fit(X, y) is model
    np.testing.assert_allclose(model.coef_, [3.0, -2.0], atol=1e-3)
    assert model.intercept_ == pytest.approx(1.0, abs=1e-3)

    predictions = model.predict(X)
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))
    assert rmse < 1e-2
    assert model.stop_reason_ == "converged"


def test_predictions_match_numpy_lstsq(linear_problem) -> None:
    X, y = linear_problem
    model = LinearRegressionGD(learning_rate=0.1, max_iter=5000, tol=1e-10).fit(X, y)

    design = np.column_stack([X, np.ones(X.shape[0])])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    lstsq_predictions = design @ solution

    np.testing.assert_allclose(model.predict(X), lstsq_predictions, atol=1e-3)


def test_ridge_coefficient_norm_is_smaller(linear_problem) -> None:
    X, y = linear_problem
    linear = LinearRegressionGD(learning_rate=0.1, max_iter=5000, tol=1e-10).fit(X, y)
    ridge = RidgeRegressionGD(
        alpha=1.0, learning_rate=0.1, max_iter=20000, tol=1e-10
    ).fit(X, y)

    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(linear.coef_)
    # b is unpenalized: at convergence grad_b = 0, so b* = mean(y) - mean(X) @ w*.
    expected_b = float(y.mean() - X.mean(axis=0) @ ridge.coef_)
    assert ridge.intercept_ == pytest.approx(expected_b, abs=1e-4)


def test_ridge_alpha_zero_matches_linear_regression(linear_problem) -> None:
    X, y = linear_problem
    linear = LinearRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)
    ridge = RidgeRegressionGD(
        alpha=0.0, learning_rate=0.1, max_iter=2000, tol=1e-10
    ).fit(X, y)

    np.testing.assert_allclose(ridge.coef_, linear.coef_, atol=1e-8)
    assert ridge.intercept_ == pytest.approx(linear.intercept_, abs=1e-8)


def test_ridge_matches_closed_form_solution() -> None:
    # Non-centered data (non-zero mean and scale), fixed seed, positive alpha.
    rng = np.random.default_rng(123)
    n_samples, n_features = 120, 3
    X = rng.normal(loc=2.0, scale=3.0, size=(n_samples, n_features))
    y = X @ np.array([1.5, -2.0, 0.7]) + 0.5 + rng.normal(scale=0.01, size=n_samples)
    alpha = 0.8

    model = RidgeRegressionGD(
        alpha=alpha, learning_rate=0.02, max_iter=100000, tol=1e-12
    ).fit(X, y)

    # Closed form of the SAME objective: center first so that b is unpenalized.
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    closed_w = np.linalg.solve(
        X_centered.T @ X_centered + n_samples * alpha * np.eye(n_features),
        X_centered.T @ y_centered,
    )
    closed_b = float(y.mean() - X.mean(axis=0) @ closed_w)

    np.testing.assert_allclose(model.coef_, closed_w, atol=2e-5)
    assert model.intercept_ == pytest.approx(closed_b, abs=5e-5)
    assert model.stop_reason_ == "converged"


def test_predict_output_shape(linear_problem) -> None:
    X, y = linear_problem
    model = LinearRegressionGD().fit(X, y)

    assert model.predict(X).shape == (X.shape[0],)
    assert model.predict(X[:5]).shape == (5,)
    assert model.predict(X[:1]).shape == (1,)


def test_first_gradient_step_matches_objective_formula() -> None:
    # At the initial zero parameters: residual = -y, w update is
    # w1 = lr * X.T @ y / n and b1 = lr * mean(y).
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = LinearRegressionGD(learning_rate=0.1, max_iter=1, tol=0.0).fit(X, y)

    expected_w = 0.1 * (X.T @ y) / 3.0
    expected_b = 0.1 * float(y.mean())
    np.testing.assert_allclose(model.coef_, expected_w)
    assert model.intercept_ == pytest.approx(expected_b)
    assert model.n_iter_ == 1
    assert model.stop_reason_ == "max_iter"


def test_loss_history_is_post_update_and_counted_correctly(linear_problem) -> None:
    X, y = linear_problem
    model = LinearRegressionGD(learning_rate=0.1, max_iter=3, tol=0.0).fit(X, y)

    assert model.n_iter_ == 3
    assert len(model.loss_history_) == 3
    # Post-update losses should strictly decrease on this well-conditioned problem.
    assert model.loss_history_[1] < model.loss_history_[0]
    assert model.loss_history_[2] < model.loss_history_[1]
    assert model.n_features_in_ == 2


def test_repeated_fit_resets_training_state(linear_problem) -> None:
    X, y = linear_problem
    max_iter = 2000
    model = LinearRegressionGD(learning_rate=0.1, max_iter=max_iter, tol=1e-10).fit(X, y)
    old_history = model.loss_history_

    rng = np.random.default_rng(7)
    X_new = rng.standard_normal((40, 3))
    y_new = X_new @ np.array([1.0, -1.0, 2.0])
    model.fit(X_new, y_new)

    # The second run must produce a fresh history object, not mutate the old one.
    assert model.loss_history_ is not old_history
    assert len(model.loss_history_) == model.n_iter_
    assert model.n_iter_ <= max_iter
    assert model.n_features_in_ == 3
    assert model.coef_.shape == (3,)


def test_failed_refit_preserves_last_successful_state(linear_problem) -> None:
    X, y = linear_problem
    model = LinearRegressionGD(learning_rate=0.1, max_iter=2000, tol=1e-10).fit(X, y)
    old_coef = model.coef_.copy()
    old_intercept = model.intercept_
    old_history = model.loss_history_.copy()
    old_n_iter = model.n_iter_
    old_n_features = model.n_features_in_

    bad_y = y.copy()
    bad_y[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(X, bad_y)

    # The last successful state must remain fully intact and usable.
    np.testing.assert_array_equal(model.coef_, old_coef)
    assert model.intercept_ == old_intercept
    assert model.loss_history_ == old_history
    assert model.n_iter_ == old_n_iter
    assert model.n_features_in_ == old_n_features
    assert model.predict(X[:3]).shape == (3,)


def test_divergence_raises_clear_error(linear_problem) -> None:
    X, y = linear_problem
    model = LinearRegressionGD(learning_rate=1e6, max_iter=100, tol=0.0)

    with pytest.raises(FloatingPointError, match="Non-finite"):
        model.fit(X, y)
    # A diverged fit must not leave silently usable NaN parameters behind.
    assert not hasattr(model, "coef_")


def test_predict_before_fit_raises() -> None:
    model = LinearRegressionGD()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros((2, 2)))


@pytest.mark.parametrize(
    "X",
    [
        np.array([1.0, 2.0, 3.0]),  # 1D
        np.zeros((2, 2, 2)),  # 3D
        np.zeros((0, 2)),  # no samples
    ],
)
def test_invalid_X_shape_raises(X) -> None:
    y = np.ones(max(X.shape[0], 1)) if X.ndim == 2 else np.ones(3)
    with pytest.raises(ValueError):
        LinearRegressionGD().fit(X, y)


def test_y_must_be_one_dimensional(linear_problem) -> None:
    X, _ = linear_problem
    with pytest.raises(ValueError, match="y must be a 1D"):
        LinearRegressionGD().fit(X, np.zeros((X.shape[0], 1)))


def test_sample_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        LinearRegressionGD().fit(np.zeros((4, 2)), np.zeros(3))


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([[1.0, np.nan]]), np.array([1.0])),
        (np.array([[1.0, np.inf]]), np.array([1.0])),
        (np.array([[1.0, 2.0]]), np.array([np.nan])),
        (np.array([[1.0, 2.0]]), np.array([-np.inf])),
    ],
)
def test_non_finite_inputs_raise(X, y) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        LinearRegressionGD().fit(X, y)


def test_predict_feature_count_mismatch_raises(linear_problem) -> None:
    X, y = linear_problem
    model = LinearRegressionGD().fit(X, y)
    with pytest.raises(ValueError, match="feature"):
        model.predict(np.zeros((5, 3)))


def test_predict_non_finite_input_raises(linear_problem) -> None:
    X, y = linear_problem
    model = LinearRegressionGD().fit(X, y)
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.predict(np.array([[1.0, np.nan]]))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": -0.1}, "learning_rate"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": -3}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"tol": -1e-9}, "tol"),
    ],
)
def test_invalid_hyperparameters_raise(kwargs, match) -> None:
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    with pytest.raises((ValueError, TypeError), match=match):
        LinearRegressionGD(**kwargs).fit(X, y)


def test_invalid_ridge_alpha_raises() -> None:
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="alpha"):
        RidgeRegressionGD(alpha=-0.5).fit(X, y)
