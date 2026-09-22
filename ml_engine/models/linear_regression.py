"""Linear and ridge regression implemented with NumPy batch gradient descent.

Two public estimators are provided:

- :class:`LinearRegressionGD`: ordinary linear regression, ``alpha`` fixed at 0.
- :class:`RidgeRegressionGD`: ridge regression with a non-negative ``alpha``.

Both minimize the same objective

    L(w, b) = ||Xw + b - y||^2 / (2n) + alpha * ||w||^2 / 2

where the intercept ``b`` is excluded from the L2 penalty. The matching
gradients are

    grad_w = X.T @ (Xw + b - y) / n + alpha * w
    grad_b = mean(Xw + b - y)

The models deliberately do not load datasets, split data, standardize inputs,
or compute evaluation metrics.
"""

from __future__ import annotations

import numpy as np

_STOP_CONVERGED = "converged"
_STOP_MAX_ITER = "max_iter"


def _check_real(value: float, name: str, *, positive: bool) -> float:
    """Validate a finite scalar hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if positive:
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}.")
    elif value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


def _check_max_iter(value: int) -> int:
    """Validate that max_iter is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"max_iter must be an integer, got {type(value).__name__}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"max_iter must be a positive integer, got {value}.")
    return value


def _check_xy(X: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert training inputs to float64 arrays."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if X_arr.ndim != 2:
        raise ValueError(
            f"X must be a 2D array of shape (n_samples, n_features); got {X_arr.ndim}D array."
        )
    if y_arr.ndim != 1:
        raise ValueError(f"y must be a 1D array of shape (n_samples,); got {y_arr.ndim}D array.")

    n_samples = X_arr.shape[0]
    if n_samples == 0:
        raise ValueError("X must contain at least one sample; got 0.")
    if y_arr.shape[0] != n_samples:
        raise ValueError(
            f"X and y must contain the same number of samples; "
            f"got {n_samples} and {y_arr.shape[0]}."
        )

    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y contains NaN or infinite values.")

    return X_arr, y_arr


class _BaseLinearRegressionGD:
    """Shared full-batch gradient descent implementation.

    Subclasses define the regularization strength through :meth:`_get_alpha`.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _get_alpha(self) -> float:
        raise NotImplementedError

    def fit(self, X: object, y: object) -> _BaseLinearRegressionGD:
        """Fit coefficients with full-batch gradient descent.

        ``loss_history_[k]`` stores the objective evaluated **after** the
        (k + 1)-th parameter update, so ``len(loss_history_) == n_iter_``.
        Training stops with ``stop_reason_ == "converged"`` once the absolute
        difference between the objective at the current (post-update)
        parameter state and the objective at the previous parameter state
        satisfies ``|L_k - L_(k-1)| <= tol``; for the first update the
        "previous state" is the initial (zero) parameter state. Otherwise
        training stops at ``"max_iter"``. No validation/test data is used for
        the stopping decision.

        Training runs entirely on local variables; the learned attributes are
        replaced as a group only after training finishes successfully. If a
        repeated fit on an already fitted model raises, the attributes from
        the last successful fit are left unchanged.
        """
        learning_rate = _check_real(self.learning_rate, "learning_rate", positive=True)
        max_iter = _check_max_iter(self.max_iter)
        tol = _check_real(self.tol, "tol", positive=False)
        alpha = self._get_alpha()

        X_arr, y_arr = _check_xy(X, y)
        n_samples, n_features = X_arr.shape

        # All candidate state lives in local variables until training
        # succeeds, so a failed refit never mutates the existing attributes.
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        residual = -y_arr  # X @ w + b - y at the initial zero parameters
        previous_loss = float(residual @ residual) / (2.0 * n_samples)

        loss_history: list[float] = []
        n_iter = 0
        stop_reason = _STOP_MAX_ITER

        for iteration in range(max_iter):
            # Overflow on a diverging step is caught explicitly below and turned
            # into a clear error, so suppress the low-level numeric warnings.
            with np.errstate(over="ignore", invalid="ignore"):
                grad_w = X_arr.T @ residual / n_samples + alpha * w
                grad_b = float(residual.mean())

                w = w - learning_rate * grad_w
                b = b - learning_rate * grad_b

                residual = X_arr @ w + b - y_arr
                loss = (
                    float(residual @ residual) / (2.0 * n_samples)
                    + alpha * float(w @ w) / 2.0
                )

            if not (np.isfinite(loss) and np.isfinite(w).all() and np.isfinite(b)):
                raise FloatingPointError(
                    f"Non-finite loss or parameters at iteration {iteration + 1}; "
                    "the optimization diverged, try reducing learning_rate."
                )

            loss_history.append(loss)
            n_iter = iteration + 1

            if abs(previous_loss - loss) <= tol:
                stop_reason = _STOP_CONVERGED
                break
            previous_loss = loss

        # Commit the new training state as a group only on full success.
        self.coef_ = w
        self.intercept_ = float(b)
        self.loss_history_ = loss_history
        self.n_iter_ = n_iter
        self.stop_reason_ = stop_reason
        self.n_features_in_ = n_features
        return self

    def predict(self, X: object) -> np.ndarray:
        """Return predictions ``X @ coef_ + intercept_``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError(
                "This model is not fitted yet; call fit() before calling predict()."
            )

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be a 2D array of shape (n_samples, n_features); got {X_arr.ndim}D array."
            )
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} feature(s), but the model was fitted with "
                f"{self.n_features_in_}."
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values.")

        return X_arr @ self.coef_ + self.intercept_


class LinearRegressionGD(_BaseLinearRegressionGD):
    """Ordinary linear regression (``alpha = 0``) via batch gradient descent.

    Parameters
    ----------
    learning_rate:
        Positive step size for every gradient descent update.
    max_iter:
        Maximum number of full-batch parameter updates (positive integer).
    tol:
        Non-negative convergence tolerance on the absolute change of the
        objective between successive updates.

    Attributes
    ----------
    coef_, intercept_, loss_history_, n_iter_, stop_reason_, n_features_in_
        Populated by :meth:`fit`; see its docstring for their semantics.
    """

    def _get_alpha(self) -> float:
        return 0.0


class RidgeRegressionGD(_BaseLinearRegressionGD):
    """Ridge regression (``alpha >= 0``) via batch gradient descent.

    The intercept is not penalized. ``alpha = 0`` is equivalent to ordinary
    linear regression. Parameters and attributes are the same as
    :class:`LinearRegressionGD`, with ``alpha`` added.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-8,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(learning_rate=learning_rate, max_iter=max_iter, tol=tol)
        self.alpha = alpha

    def _get_alpha(self) -> float:
        return _check_real(self.alpha, "alpha", positive=False)
