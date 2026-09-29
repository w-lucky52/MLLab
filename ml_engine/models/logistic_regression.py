"""Multinomial logistic regression implemented with NumPy.

The single public estimator, :class:`LogisticRegressionGD`, is a softmax
regression trained with full-batch gradient descent. It handles both binary
and multiclass labels (including non-numeric labels such as strings) and
minimizes

    L(W, b) = cross_entropy + alpha * ||W||^2 / 2

where the per-class intercepts ``b`` are excluded from the L2 penalty. With a
batch size of ``n`` samples and probabilities ``P`` (softmax of the logits),
the gradients are

    grad_W = X.T @ (P - Y) / n + alpha * W, reshaped as (n_classes, n_features)
    grad_b = sum over samples of (P - Y) / n

Softmax and cross entropy are evaluated in a numerically stable form by
subtracting the per-row maximum logit before taking exponentials.

The model deliberately does not load datasets, split data, standardize
inputs, or compute evaluation metrics.
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


def _check_X(X: object, *, require_nonempty: bool) -> np.ndarray:
    """Validate and convert X to a 2D float64 array."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(
            f"X must be a 2D array of shape (n_samples, n_features); got {X_arr.ndim}D array."
        )
    if require_nonempty and X_arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample; got 0.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")
    return X_arr


def _encode_labels(y: object, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert 1D labels to ``(classes, integer_codes)``.

    The original label dtype/values are preserved in ``classes`` so that
    predictions can return the same labels the user trained on.
    """
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be a 1D array of shape (n_samples,); got {y_arr.ndim}D array.")
    if y_arr.shape[0] != n_samples:
        raise ValueError(
            f"X and y must contain the same number of samples; "
            f"got {n_samples} and {y_arr.shape[0]}."
        )
    if not _labels_are_finite(y_arr):
        raise ValueError("y contains NaN or infinite values.")

    try:
        classes, codes = np.unique(y_arr, return_inverse=True)
    except TypeError as exc:
        # np.unique cannot order mixed/incomparable label types (e.g. a mix
        # of strings and numbers inside an object array).
        raise ValueError(
            "y contains incomparable label types; use labels of a single "
            f"comparable type (all numbers or all strings). Original error: {exc}"
        ) from exc
    if classes.shape[0] < 2:
        raise ValueError("LogisticRegressionGD requires at least two distinct classes; got 1.")
    return classes, codes


def _labels_are_finite(y_arr: np.ndarray) -> bool:
    """Reject NaN/infinite labels for numeric and object dtypes.

    Pure-string labels (fixed-width unicode dtype or object arrays of
    strings) are always accepted; only numeric scalars are finite-checked.
    """
    kind = y_arr.dtype.kind
    if kind in ("f", "c"):
        return bool(np.all(np.isfinite(y_arr)))
    if kind == "O":
        for value in y_arr.tolist():
            if isinstance(value, (int, float, complex, np.number)) and not np.isfinite(
                complex(value)
                if isinstance(value, (complex, np.complexfloating))
                else float(value)
            ):
                return False
    return True


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (per-row max subtraction)."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax (per-row max subtraction)."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_denominator = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_denominator


class LogisticRegressionGD:
    """Multinomial (softmax) logistic regression via batch gradient descent.

    Works for binary and multiclass targets; labels may be integers or
    strings and are returned unchanged by :meth:`predict`.

    Parameters
    ----------
    learning_rate:
        Positive step size for every gradient descent update.
    max_iter:
        Maximum number of full-batch parameter updates (positive integer).
    tol:
        Non-negative convergence tolerance on the absolute change of the
        objective between successive parameter states.
    alpha:
        Non-negative L2 regularization strength applied to ``coef_`` only;
        the intercepts are never penalized.

    Attributes
    ----------
    classes_:
        Sorted unique training labels, shape ``(n_classes,)``.
    coef_:
        Weight matrix, shape ``(n_classes, n_features)``.
    intercept_:
        Per-class biases, shape ``(n_classes,)``.
    loss_history_, n_iter_, stop_reason_, n_features_in_
        Populated by :meth:`fit`; see its docstring for their semantics.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-8,
        alpha: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

    def fit(self, X: object, y: object) -> LogisticRegressionGD:
        """Fit the softmax model with full-batch gradient descent.

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
        alpha = _check_real(self.alpha, "alpha", positive=False)

        X_arr = _check_X(X, require_nonempty=True)
        n_samples, n_features = X_arr.shape
        classes, codes = _encode_labels(y, n_samples)
        n_classes = classes.shape[0]

        one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), codes] = 1.0

        # All candidate state lives in local variables until training
        # succeeds, so a failed refit never mutates the existing attributes.
        W = np.zeros((n_classes, n_features), dtype=np.float64)
        b = np.zeros(n_classes, dtype=np.float64)
        previous_loss = float(np.log(n_classes))  # cross entropy at W = 0, b = 0

        loss_history: list[float] = []
        n_iter = 0
        stop_reason = _STOP_MAX_ITER

        for iteration in range(max_iter):
            # Overflow/NaN on a diverging step is caught explicitly below and
            # turned into a clear error, so suppress low-level warnings.
            with np.errstate(over="ignore", invalid="ignore"):
                probabilities = _softmax(X_arr @ W.T + b)
                gradient_logits = (probabilities - one_hot) / n_samples
                gradient_W = gradient_logits.T @ X_arr + alpha * W
                gradient_b = np.sum(gradient_logits, axis=0)

                W = W - learning_rate * gradient_W
                b = b - learning_rate * gradient_b

                log_probabilities = _stable_log_softmax(X_arr @ W.T + b)
                cross_entropy = float(
                    -np.mean(log_probabilities[np.arange(n_samples), codes])
                )
                loss = cross_entropy + alpha * float(np.sum(W * W)) / 2.0

            if not (np.isfinite(loss) and np.isfinite(W).all() and np.isfinite(b).all()):
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
        self.classes_ = classes
        self.coef_ = W
        self.intercept_ = b
        self.loss_history_ = loss_history
        self.n_iter_ = n_iter
        self.stop_reason_ = stop_reason
        self.n_features_in_ = n_features
        return self

    def _decision_proba(self, X: object) -> np.ndarray:
        """Validate inputs and return softmax probabilities."""
        if not hasattr(self, "coef_"):
            raise RuntimeError(
                "This model is not fitted yet; call fit() before using it for prediction."
            )
        X_arr = _check_X(X, require_nonempty=False)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} feature(s), but the model was fitted with "
                f"{self.n_features_in_}."
            )

        # A finite X can still overflow during X @ coef_.T; suppress the raw
        # overflow warnings and reject any non-finite logits/probabilities with
        # a clear error instead of silently returning NaN/inf.
        with np.errstate(over="ignore", invalid="ignore"):
            logits = X_arr @ self.coef_.T + self.intercept_
            if not np.all(np.isfinite(logits)):
                raise FloatingPointError(
                    "Non-finite logits during prediction; input feature magnitudes are "
                    "too large for the fitted coefficients to produce finite probabilities."
                )
            probabilities = _softmax(logits)
            if not np.all(np.isfinite(probabilities)):
                raise FloatingPointError(
                    "Non-finite probabilities during prediction; the softmax computation "
                    "overflowed even though the inputs were finite."
                )
        return probabilities

    def predict_proba(self, X: object) -> np.ndarray:
        """Return class probabilities, shape ``(n_samples, n_classes)``.

        Columns are ordered like ``classes_``. Each row sums to 1 and every
        value lies in ``[0, 1]``.
        """
        return self._decision_proba(X)

    def predict(self, X: object) -> np.ndarray:
        """Return the predicted original training labels for each row of X."""
        probabilities = self._decision_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
