"""Structured training-process log for experiment pipelines.

Every entry is a plain JSON-native dict::

    {"sequence": 1, "event": "dataset_loaded", "elapsed_ms": 2.41, "details": {...}}

Rules enforced here:

- ``sequence`` starts at 1 and increases by 1 for every appended entry;
- ``elapsed_ms`` is a non-negative finite float measured with
  :func:`time.perf_counter` from the pipeline start, or ``None`` for
  ``iteration_checkpoint`` entries (per-iteration timings are not recorded
  by the models, so nothing is fabricated);
- ``details`` is always a JSON object; pre-rendered human sentences are
  never stored — the frontend renders text from ``event`` + ``details``;
- iteration checkpoints are sampled from the *real* recorded loss curve
  only (see :func:`sample_iteration_checkpoints`).
"""

from __future__ import annotations

import math
import time
from typing import Any

MAX_CHECKPOINTS = 25

# Scalar metric keys copied into the ``evaluation_completed`` log entry.
# Everything else (confusion matrix, per-class rows, arrays, non-finite
# values) stays out of the log.
CLASSIFICATION_LOG_METRICS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
REGRESSION_LOG_METRICS = ("mae", "mse", "rmse", "r2")


def _finite_number(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _finite_curve_1d(curve: Any) -> list[float] | None:
    """Return ``curve`` as a finite 1-D float list, or ``None``.

    Rejects empty curves, multi-dimensional data and anything containing
    NaN / infinity or non-numeric values. The input is never modified.
    """
    if curve is None or isinstance(curve, (str, bytes, dict)):
        return None
    try:
        items = list(curve)
    except TypeError:
        return None
    if not items:
        return None

    values: list[float] = []
    for item in items:
        number = _finite_number(item)
        if number is None:
            return None
        values.append(number)
    return values


def sample_iteration_checkpoints(
    curve: Any,
    loss_name: Any = None,
    loss_source: Any = None,
    max_points: int = MAX_CHECKPOINTS,
) -> list[dict[str, Any]]:
    """Build ``iteration_checkpoint`` details from a real loss curve.

    - curves with at most ``max_points`` points are kept entirely;
    - longer curves are down-sampled to exactly ``max_points`` points using
      approximately evenly spaced indices that always include the first
      and the last iteration, de-duplicated and sorted ascending;
    - every ``value`` comes from the original curve; nothing is
      interpolated, smoothed or invented;
    - unusable curves (NaN/Inf, empty, multi-dimensional, non-numeric)
      yield an empty list, so models without a real curve (KNN,
      GaussianNB, decision tree, random forest) simply log nothing.

    ``iteration`` is 1-based. ``loss_name`` / ``loss_source`` are copied
    from the training info when available.
    """
    values = _finite_curve_1d(curve)
    if values is None:
        return []

    count = len(values)
    if count <= max_points:
        indices = list(range(count))
    else:
        indices = sorted(
            {round(i * (count - 1) / (max_points - 1)) for i in range(max_points)}
        )

    name = loss_name if isinstance(loss_name, str) and loss_name else None
    source = loss_source if isinstance(loss_source, str) and loss_source else None

    checkpoints: list[dict[str, Any]] = []
    for index in indices:
        details: dict[str, Any] = {
            "iteration": index + 1,
            "value": values[index],
        }
        if name is not None:
            details["loss_name"] = name
        if source is not None:
            details["source"] = source
        checkpoints.append(details)
    return checkpoints


def finite_scalar_metrics(
    metrics: Any,
    allowed_keys: tuple[str, ...],
) -> dict[str, float]:
    """Copy only finite scalar metrics under ``allowed_keys``.

    Non-finite values, containers and unknown keys are dropped so the log
    never duplicates confusion matrices, per-class rows or arrays.
    """
    if not isinstance(metrics, dict):
        return {}
    result: dict[str, float] = {}
    for key in allowed_keys:
        number = _finite_number(metrics.get(key))
        if number is not None:
            result[key] = number
    return result


class TrainingProcessLog:
    """Append-only builder for one experiment's ``process_log``.

    ``start`` defaults to ``time.perf_counter()`` at construction; tests
    may inject a fixed value. ``elapsed_ms`` is always measured from this
    start, never from wall-clock timestamps.
    """

    def __init__(self, start: float | None = None) -> None:
        self._start = time.perf_counter() if start is None else float(start)
        self._entries: list[dict[str, Any]] = []

    def elapsed_ms(self) -> float:
        """Milliseconds since the pipeline start (perf_counter based)."""
        return (time.perf_counter() - self._start) * 1000.0

    def add(
        self,
        event: str,
        details: dict[str, Any] | None = None,
        elapsed_ms: float | None = None,
    ) -> dict[str, Any]:
        """Append an entry and return it.

        ``elapsed_ms`` defaults to the current pipeline-relative elapsed
        time; an explicitly passed value must be a non-negative finite
        number. ``details`` must be a dict when given.
        """
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a non-empty string.")
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise TypeError("details must be a dict.")

        if elapsed_ms is None:
            elapsed = self.elapsed_ms()
        else:
            elapsed = _finite_number(elapsed_ms)
            if elapsed is None or elapsed < 0:
                raise ValueError("elapsed_ms must be a non-negative finite number.")

        entry = {
            "sequence": len(self._entries) + 1,
            "event": event,
            "elapsed_ms": elapsed,
            "details": dict(details),
        }
        self._entries.append(entry)
        return entry

    def add_iteration_checkpoint(
        self,
        iteration: int,
        value: float,
        loss_name: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Append an ``iteration_checkpoint`` entry with ``elapsed_ms=None``.

        Per-iteration timings are not recorded by the models, so the field
        stays ``None`` instead of being fabricated.
        """
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
            raise ValueError("iteration must be a positive integer.")
        number = _finite_number(value)
        if number is None:
            raise ValueError("value must be a finite number.")

        details: dict[str, Any] = {"iteration": iteration, "value": number}
        if loss_name:
            details["loss_name"] = loss_name
        if source:
            details["source"] = source

        entry = {
            "sequence": len(self._entries) + 1,
            "event": "iteration_checkpoint",
            "elapsed_ms": None,
            "details": details,
        }
        self._entries.append(entry)
        return entry

    def extend_iteration_checkpoints(self, checkpoints: list[dict[str, Any]]) -> None:
        """Append pre-sampled checkpoint dicts (see ``sample_iteration_checkpoints``)."""
        for checkpoint in checkpoints:
            self.add_iteration_checkpoint(
                checkpoint["iteration"],
                checkpoint["value"],
                loss_name=checkpoint.get("loss_name"),
                source=checkpoint.get("source"),
            )

    def entries(self) -> list[dict[str, Any]]:
        """Return a copy of the log; later mutation never leaks back."""
        return [
            {
                "sequence": entry["sequence"],
                "event": entry["event"],
                "elapsed_ms": entry["elapsed_ms"],
                "details": dict(entry["details"]),
            }
            for entry in self._entries
        ]
