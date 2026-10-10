"""Single-hidden-layer MLP classifier implemented with NumPy.

The single public estimator, :class:`MLPClassifier`, is a feed-forward
neural network with one hidden layer trained by full-batch gradient descent.
The forward pass is

    Z1 = X @ W1 + b1
    H  = ReLU(Z1)
    logits = H @ W2 + b2
    probabilities = softmax(logits)

with shapes ``W1: (n_features, hidden_size)``, ``W2: (hidden_size, n_classes)``
and biases ``b1: (hidden_size,)``, ``b2: (n_classes,)``. The hidden layer uses
He initialization ``rng.standard_normal * sqrt(2 / n_features)`` and the output
layer uses ``rng.standard_normal * sqrt(1 / hidden_size)``; biases are zero.
The random state is local (``np.random.default_rng``) and never mutates NumPy's
global state.

The objective is

    L = cross_entropy + alpha / 2 * (||W1||_F^2 + ||W2||_F^2)

with the biases excluded from the L2 penalty. With one-hot ``Y`` of shape
``(n_samples, n_classes)``, probabilities ``P`` and ``n`` training samples,
the gradients are

    error    = (P - Y) / n
    grad_W2  = H.T @ error + alpha * W2
    grad_b2  = sum(error, axis=0)
    grad_H   = error @ W2.T
    grad_Z1  = grad_H * (Z1 > 0)
    grad_W1  = X.T @ grad_Z1 + alpha * W1
    grad_b1  = sum(grad_Z1, axis=0)

Softmax and cross-entropy use a numerically stable form (per-row max
subtraction and a stable log-softmax for the loss) so no exponentials are
taken on un-shifted logits.

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


def _check_positive_int(value: int, name: str) -> int:
    """Validate that a hyperparameter is a positive integer (bool rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _check_random_state(value: int) -> int:
    """Validate random_state: a non-negative integer (bool rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"random_state must be an integer, got {type(value).__name__}.")
    value = int(value)
    if value < 0:
        raise ValueError(f"random_state must be >= 0, got {value}.")
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
    if require_nonempty and X_arr.shape[1] == 0:
        raise ValueError("X must contain at least one feature; got 0.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")
    return X_arr


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
        raise ValueError("MLPClassifier requires at least two distinct classes; got 1.")
    return classes, codes


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


class MLPClassifier:
    """Single-hidden-layer MLP classifier via batch gradient descent.

    Works for binary and multiclass targets; labels may be integers or
    strings and are returned unchanged by :meth:`predict`. The hidden layer
    uses ReLU and He initialization; the output layer is softmax.

    Parameters
    ----------
    hidden_size:
        Number of hidden units (positive integer, ``bool`` not allowed).
    learning_rate:
        Positive step size for every gradient descent update.
    max_iter:
        Maximum number of full-batch parameter updates (positive integer).
    tol:
        Non-negative convergence tolerance on the absolute change of the
        objective between successive parameter states.
    alpha:
        Non-negative L2 regularization strength applied to ``W1`` and ``W2``
        only; the biases are never penalized.
    random_state:
        Seed for the local ``np.random.default_rng`` used to initialize
        weights. NumPy's global random state is never touched.

    Attributes
    ----------
    classes_:
        Sorted unique training labels, shape ``(n_classes,)``.
    coefs_:
        List ``[W1, W2]`` with the trained weight matrices.
    intercepts_:
        List ``[b1, b2]`` with the trained bias vectors.
    loss_history_, n_iter_, stop_reason_, n_features_in_, n_classes_, hidden_size_
        Populated by :meth:`fit`; see its docstring for their semantics.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        alpha: float = 0.0001,
        random_state: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X: object, y: object) -> MLPClassifier:
        """Fit the MLP with full-batch gradient descent.

        ``loss_history_[k]`` stores the objective evaluated **after** the
        (k + 1)-th parameter update, so ``len(loss_history_) == n_iter_``.
        Training stops with ``stop_reason_ == "converged"`` once the absolute
        difference between the objective at the current (post-update)
        parameter state and the objective at the previous parameter state
        satisfies ``|L_k - L_(k-1)| <= tol``. For the first update the
        "previous state" is the initial (He-initialized) parameter state,
        evaluated with a real forward pass rather than a closed-form value.
        Otherwise training stops at ``"max_iter"``. No validation/test data
        is used for the stopping decision.

        Training runs entirely on local variables; the learned attributes
        are replaced as a group only after training finishes successfully.
        If a repeated fit on an already fitted model raises, the attributes
        from the last successful fit are left unchanged.
        """
        hidden_size = _check_positive_int(self.hidden_size, "hidden_size")
        learning_rate = _check_real(self.learning_rate, "learning_rate", positive=True)
        max_iter = _check_positive_int(self.max_iter, "max_iter")
        tol = _check_real(self.tol, "tol", positive=False)
        alpha = _check_real(self.alpha, "alpha", positive=False)
        random_state = _check_random_state(self.random_state)

        X_arr = _check_X(X, require_nonempty=True)
        n_samples, n_features = X_arr.shape
        classes, codes = _encode_labels(y, n_samples)
        n_classes = classes.shape[0]

        one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), codes] = 1.0

        # Local RNG so NumPy's global state is never touched.
        rng = np.random.default_rng(random_state)

        # He initialization for the hidden layer; smaller scale for output.
        W1 = rng.standard_normal((n_features, hidden_size)) * np.sqrt(2.0 / n_features)
        b1 = np.zeros(hidden_size, dtype=np.float64)
        W2 = rng.standard_normal((hidden_size, n_classes)) * np.sqrt(1.0 / hidden_size)
        b2 = np.zeros(n_classes, dtype=np.float64)

        def _objective(
            w1: np.ndarray, w2: np.ndarray, b1v: np.ndarray, b2v: np.ndarray
        ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Forward pass returning loss, Z1, H, logits, probabilities."""
            z1 = X_arr @ w1 + b1v
            h = np.maximum(0.0, z1)
            logits = h @ w2 + b2v
            log_prob = _stable_log_softmax(logits)
            cross_entropy = float(-np.mean(log_prob[np.arange(n_samples), codes]))
            reg = alpha * (float(np.sum(w1 * w1)) + float(np.sum(w2 * w2))) / 2.0
            loss = cross_entropy + reg
            probabilities = _softmax(logits)
            return loss, z1, h, logits, probabilities

        # previous_loss = objective at the initial (He-initialized) parameters,
        # computed via an actual forward pass (NOT a closed-form log(n_classes)).
        # The initial forward pass can overflow on extreme (but finite) inputs,
        # so suppress raw warnings and reject non-finite values before training.
        with np.errstate(over="ignore", invalid="ignore"):
            previous_loss, _, _, init_logits, init_probabilities = _objective(
                W1, W2, b1, b2
            )

        if not (
            np.isfinite(previous_loss)
            and np.isfinite(W1).all()
            and np.isfinite(b1).all()
            and np.isfinite(W2).all()
            and np.isfinite(b2).all()
            and np.isfinite(init_logits).all()
            and np.isfinite(init_probabilities).all()
        ):
            raise FloatingPointError(
                "Non-finite loss or parameters at initialization; "
                "the initial forward pass overflowed, try reducing input "
                "feature magnitudes or hidden_size."
            )

        loss_history: list[float] = []
        n_iter = 0
        stop_reason = _STOP_MAX_ITER

        for iteration in range(max_iter):
            # Overflow/NaN on a diverging step is caught explicitly below and
            # turned into a clear error, so suppress low-level warnings.
            with np.errstate(over="ignore", invalid="ignore"):
                _, z1, h, _, probabilities = _objective(W1, W2, b1, b2)

                error = (probabilities - one_hot) / n_samples
                grad_w2 = h.T @ error + alpha * W2
                grad_b2 = np.sum(error, axis=0)
                grad_h = error @ W2.T
                grad_z1 = grad_h * (z1 > 0.0)
                grad_w1 = X_arr.T @ grad_z1 + alpha * W1
                grad_b1 = np.sum(grad_z1, axis=0)

                W1 = W1 - learning_rate * grad_w1
                b1 = b1 - learning_rate * grad_b1
                W2 = W2 - learning_rate * grad_w2
                b2 = b2 - learning_rate * grad_b2

                loss, _, _, logits, probabilities = _objective(W1, W2, b1, b2)

            if not (
                np.isfinite(loss)
                and np.isfinite(W1).all()
                and np.isfinite(b1).all()
                and np.isfinite(W2).all()
                and np.isfinite(b2).all()
                and np.isfinite(logits).all()
                and np.isfinite(probabilities).all()
            ):
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
        # Use fresh list objects so a successful refit fully replaces the
        # previous coefs_/intercepts_/loss_history_ references.
        self.classes_ = classes
        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, b2]
        self.loss_history_ = loss_history
        self.n_iter_ = n_iter
        self.stop_reason_ = stop_reason
        self.n_features_in_ = n_features
        self.n_classes_ = n_classes
        self.hidden_size_ = hidden_size
        return self

    def _forward(self, X: object) -> np.ndarray:
        """Validate inputs and return softmax probabilities."""
        if not hasattr(self, "coefs_"):
            raise RuntimeError(
                "This model is not fitted yet; call fit() before using it for prediction."
            )
        X_arr = _check_X(X, require_nonempty=False)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} feature(s), but the model was fitted with "
                f"{self.n_features_in_}."
            )

        W1, W2 = self.coefs_
        b1, b2 = self.intercepts_

        # A finite X can still overflow during the matmuls; suppress the raw
        # overflow warnings and reject any non-finite logits/probabilities
        # with a clear error instead of silently returning NaN/inf.
        with np.errstate(over="ignore", invalid="ignore"):
            z1 = X_arr @ W1 + b1
            h = np.maximum(0.0, z1)
            logits = h @ W2 + b2
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
        return self._forward(X)

    def predict(self, X: object) -> np.ndarray:
        """Return the predicted original training labels for each row of X."""
        probabilities = self._forward(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
