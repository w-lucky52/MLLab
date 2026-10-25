"""Phase 7B1 — structured training-process log (``training_info.process_log``).

Covers the log builder, iteration-checkpoint sampling, end-to-end pipeline
recording with real datasets/models, API schema compatibility and the
summary.json contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from backend.api.schemas import TrainingInfo, TrainingLogEntry
from backend.core.config import get_settings
from backend.services.training_log import (
    CLASSIFICATION_LOG_METRICS,
    MAX_CHECKPOINTS,
    REGRESSION_LOG_METRICS,
    TrainingProcessLog,
    finite_scalar_metrics,
    sample_iteration_checkpoints,
)

# --- TrainingProcessLog builder ----------------------------------------------


def test_sequence_starts_at_one_and_increments():
    log = TrainingProcessLog()
    log.add("dataset_loaded", {"dataset_name": "iris"})
    log.add("training_started", {})
    log.add_iteration_checkpoint(1, 0.5)
    log.add("training_completed")

    sequences = [entry["sequence"] for entry in log.entries()]
    assert sequences == [1, 2, 3, 4]


def test_elapsed_ms_non_negative_finite():
    log = TrainingProcessLog()
    entry = log.add("dataset_loaded", {})
    assert isinstance(entry["elapsed_ms"], float)
    assert math.isfinite(entry["elapsed_ms"])
    assert entry["elapsed_ms"] >= 0.0


def test_iteration_checkpoint_elapsed_is_none():
    log = TrainingProcessLog()
    entry = log.add_iteration_checkpoint(3, 0.25, loss_name="objective", source="loss_history_")
    assert entry["event"] == "iteration_checkpoint"
    assert entry["elapsed_ms"] is None
    assert entry["details"] == {
        "iteration": 3,
        "value": 0.25,
        "loss_name": "objective",
        "source": "loss_history_",
    }


def test_add_rejects_invalid_arguments():
    log = TrainingProcessLog()
    with pytest.raises(ValueError):
        log.add("", {})
    with pytest.raises(TypeError):
        log.add("event", details=["not", "a", "dict"])
    with pytest.raises(ValueError):
        log.add("event", {}, elapsed_ms=-1.0)
    with pytest.raises(ValueError):
        log.add("event", {}, elapsed_ms=float("nan"))
    with pytest.raises(ValueError):
        log.add_iteration_checkpoint(0, 0.5)
    with pytest.raises(ValueError):
        log.add_iteration_checkpoint(1, float("inf"))


def test_entries_returns_detached_copy():
    log = TrainingProcessLog()
    log.add("dataset_loaded", {"dataset_name": "iris"})
    entries = log.entries()
    entries[0]["details"]["dataset_name"] = "mutated"
    entries[0]["sequence"] = 99
    fresh = log.entries()
    assert fresh[0]["details"]["dataset_name"] == "iris"
    assert fresh[0]["sequence"] == 1


# --- sample_iteration_checkpoints --------------------------------------------


def test_curve_within_limit_kept_entirely():
    curve = [0.9 - i * 0.1 for i in range(10)]
    checkpoints = sample_iteration_checkpoints(curve, "objective", "loss_history_")
    assert len(checkpoints) == 10
    assert [c["iteration"] for c in checkpoints] == list(range(1, 11))
    assert [c["value"] for c in checkpoints] == curve
    assert all(c["loss_name"] == "objective" for c in checkpoints)
    assert all(c["source"] == "loss_history_" for c in checkpoints)


def test_exactly_max_points_kept_entirely():
    curve = [float(i) for i in range(MAX_CHECKPOINTS)]
    checkpoints = sample_iteration_checkpoints(curve)
    assert len(checkpoints) == MAX_CHECKPOINTS
    assert [c["value"] for c in checkpoints] == curve


def test_long_curve_sampled_to_max_points():
    curve = [float(i) for i in range(1000)]
    checkpoints = sample_iteration_checkpoints(curve)
    assert len(checkpoints) == MAX_CHECKPOINTS


def test_sampling_includes_first_and_last_iteration():
    curve = [float(i) for i in range(500)]
    checkpoints = sample_iteration_checkpoints(curve)
    iterations = [c["iteration"] for c in checkpoints]
    assert iterations[0] == 1
    assert iterations[-1] == 500


def test_sampling_increasing_without_duplicates():
    curve = [float(i) for i in range(997)]
    checkpoints = sample_iteration_checkpoints(curve)
    iterations = [c["iteration"] for c in checkpoints]
    assert iterations == sorted(set(iterations))
    assert len(iterations) == len(set(iterations))


def test_sampled_values_come_from_original_curve():
    curve = [float(i) * 0.5 for i in range(300)]
    checkpoints = sample_iteration_checkpoints(curve)
    for checkpoint in checkpoints:
        assert checkpoint["value"] == curve[checkpoint["iteration"] - 1]


def test_input_curve_is_not_modified():
    curve = [float(i) for i in range(100)]
    snapshot = list(curve)
    sample_iteration_checkpoints(curve)
    assert curve == snapshot


@pytest.mark.parametrize(
    "curve",
    [
        None,
        [],
        [float("nan"), 1.0],
        [1.0, float("inf")],
        [1.0, float("-inf")],
        [[1.0, 2.0], [3.0, 4.0]],
        ["a", "b", "c"],
        "not-a-curve",
        42,
        {"loss": 1.0},
    ],
)
def test_invalid_curves_produce_no_checkpoints(curve):
    assert sample_iteration_checkpoints(curve) == []


def test_checkpoints_omit_missing_loss_metadata():
    checkpoints = sample_iteration_checkpoints([1.0, 0.5])
    assert "loss_name" not in checkpoints[0]
    assert "source" not in checkpoints[0]


# --- finite_scalar_metrics ----------------------------------------------------


def test_finite_scalar_metrics_filters_values():
    metrics = {
        "accuracy": 0.95,
        "precision_weighted": 0.94,
        "recall_weighted": float("nan"),
        "f1_weighted": "0.9",
        "confusion_matrix": [[1, 0], [0, 1]],
        "other": 1.0,
    }
    result = finite_scalar_metrics(metrics, CLASSIFICATION_LOG_METRICS)
    assert result == {"accuracy": 0.95, "precision_weighted": 0.94}


def test_finite_scalar_metrics_rejects_non_dict():
    assert finite_scalar_metrics(None, REGRESSION_LOG_METRICS) == {}
    assert finite_scalar_metrics([1, 2], REGRESSION_LOG_METRICS) == {}


# --- schema compatibility -----------------------------------------------------


def test_training_info_without_process_log_parses():
    info = TrainingInfo.model_validate({"loss_history": [1.0, 0.5], "n_iter": 2})
    assert info.process_log is None


def test_training_info_with_process_log_parses():
    info = TrainingInfo.model_validate(
        {
            "process_log": [
                {"sequence": 1, "event": "dataset_loaded"},
                {
                    "sequence": 2,
                    "event": "iteration_checkpoint",
                    "elapsed_ms": None,
                    "details": {"iteration": 1, "value": 0.5},
                },
            ]
        }
    )
    assert info.process_log is not None
    assert len(info.process_log) == 2
    assert info.process_log[0].elapsed_ms is None
    assert info.process_log[0].details == {}
    assert info.process_log[1].details["value"] == 0.5


def test_training_log_entry_defaults():
    entry = TrainingLogEntry(sequence=1, event="experiment_completed")
    assert entry.elapsed_ms is None
    assert entry.details == {}


# --- end-to-end pipeline ------------------------------------------------------


def _run_experiment(client, dataset_name, model_name, hyperparameters=None):
    payload = {"dataset_name": dataset_name, "model_name": model_name}
    if hyperparameters:
        payload["hyperparameters"] = hyperparameters
    created = client.post("/api/v1/experiments", json=payload).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()
    assert detail["status"] == "completed", detail.get("error")
    return detail


def _entries_by_event(detail, event):
    return [e for e in detail["training_info"]["process_log"] if e["event"] == event]


def test_dataset_loaded_uses_real_dataset_shape(client):
    detail = _run_experiment(client, "iris", "knn_classifier")
    entries = _entries_by_event(detail, "dataset_loaded")
    assert len(entries) == 1
    assert entries[0]["details"] == {
        "dataset_name": "iris",
        "sample_count": 150,
        "feature_count": 4,
        "task_type": "classification",
    }


def test_data_prepared_uses_real_sample_counts(client):
    detail = _run_experiment(client, "iris", "knn_classifier")
    entries = _entries_by_event(detail, "data_prepared")
    assert len(entries) == 1
    details = entries[0]["details"]
    assert details["train_sample_count"] == 105
    assert details["test_sample_count"] == 45
    assert details["test_size"] == 0.3
    assert details["standardize"] is True
    assert details["stratified"] is True


def test_estimator_created_and_training_started_events(client):
    detail = _run_experiment(client, "iris", "knn_classifier")
    created = _entries_by_event(detail, "estimator_created")
    assert created[0]["details"]["model_name"] == "knn_classifier"
    assert created[0]["details"]["estimator_class"] == "KNNModel"
    started = _entries_by_event(detail, "training_started")
    assert started[0]["details"]["train_sample_count"] == 105


@pytest.mark.parametrize(
    "dataset_name,model_name",
    [
        ("iris", "knn_classifier"),
        ("iris", "gaussian_nb"),
        ("diabetes", "decision_tree_regressor"),
        ("diabetes", "random_forest_regressor"),
    ],
)
def test_non_iterative_models_have_no_iteration_checkpoints(
    client, dataset_name, model_name
):
    detail = _run_experiment(client, dataset_name, model_name)
    assert _entries_by_event(detail, "iteration_checkpoint") == []
    completed = _entries_by_event(detail, "training_completed")
    assert completed[0]["details"]["loss_point_count"] == 0


@pytest.mark.parametrize(
    "model_name,hyperparameters",
    [
        ("logistic_regression", {"max_iter": 60}),
        ("linear_svm", {"max_iter": 60}),
        ("mlp_classifier", {"max_iter": 60}),
    ],
)
def test_iterative_models_use_real_loss_curve(client, model_name, hyperparameters):
    detail = _run_experiment(client, "iris", model_name, hyperparameters)
    info = detail["training_info"]
    checkpoints = _entries_by_event(detail, "iteration_checkpoint")
    assert checkpoints, f"{model_name} produced no checkpoints"

    loss_history = info["loss_history"]
    assert info["loss_name"] == "objective"
    assert info["loss_source"] == "loss_history_"

    # Sampled values must come from the real curve at the recorded iteration.
    assert len(checkpoints) == min(len(loss_history), MAX_CHECKPOINTS)
    for checkpoint in checkpoints:
        details = checkpoint["details"]
        assert details["value"] == loss_history[details["iteration"] - 1]
        assert details["loss_name"] == "objective"
        assert details["source"] == "loss_history_"
        assert checkpoint["elapsed_ms"] is None

    completed = _entries_by_event(detail, "training_completed")[0]["details"]
    assert completed["loss_point_count"] == len(loss_history)


def test_gbdt_uses_train_score_curve(client):
    detail = _run_experiment(client, "diabetes", "gbdt_regressor")
    info = detail["training_info"]
    assert info["loss_name"] == "training_loss"
    assert info["loss_source"] == "train_score_"

    checkpoints = _entries_by_event(detail, "iteration_checkpoint")
    assert len(checkpoints) == min(len(info["loss_history"]), MAX_CHECKPOINTS)
    for checkpoint in checkpoints:
        details = checkpoint["details"]
        assert details["value"] == info["loss_history"][details["iteration"] - 1]
        assert details["loss_name"] == "training_loss"
        assert details["source"] == "train_score_"


def test_short_curve_kept_entirely(client):
    detail = _run_experiment(client, "iris", "logistic_regression", {"max_iter": 10})
    info = detail["training_info"]
    checkpoints = _entries_by_event(detail, "iteration_checkpoint")
    assert len(checkpoints) == len(info["loss_history"])
    assert [c["details"]["iteration"] for c in checkpoints] == list(
        range(1, len(info["loss_history"]) + 1)
    )


def test_durations_are_finite_and_non_negative(client):
    detail = _run_experiment(client, "iris", "logistic_regression", {"max_iter": 20})
    log = detail["training_info"]["process_log"]
    for entry in log:
        if entry["event"] == "iteration_checkpoint":
            assert entry["elapsed_ms"] is None
            continue
        elapsed = entry["elapsed_ms"]
        assert isinstance(elapsed, float)
        assert math.isfinite(elapsed) and elapsed >= 0.0

    completed = _entries_by_event(detail, "training_completed")[0]["details"]
    assert math.isfinite(completed["fit_duration_ms"])
    assert completed["fit_duration_ms"] >= 0.0
    predicted = _entries_by_event(detail, "prediction_completed")[0]["details"]
    assert predicted["prediction_count"] == 45
    assert math.isfinite(predicted["predict_duration_ms"])
    assert predicted["predict_duration_ms"] >= 0.0
    evaluated = _entries_by_event(detail, "evaluation_completed")[0]["details"]
    assert math.isfinite(evaluated["evaluation_duration_ms"])
    assert evaluated["evaluation_duration_ms"] >= 0.0
    finished = _entries_by_event(detail, "experiment_completed")[0]["details"]
    assert math.isfinite(finished["total_duration_ms"])
    assert finished["total_duration_ms"] >= 0.0
    # The backend chart-file count is meaningless for the result page (its
    # charts are rendered by ECharts in the frontend), so it is no longer logged.
    assert "chart_count" not in finished


def test_evaluation_log_contains_only_finite_scalars_classification(client):
    detail = _run_experiment(client, "iris", "logistic_regression", {"max_iter": 20})
    metrics = _entries_by_event(detail, "evaluation_completed")[0]["details"]["metrics"]
    assert metrics
    assert set(metrics) <= set(CLASSIFICATION_LOG_METRICS)
    assert "confusion_matrix" not in metrics
    for value in metrics.values():
        assert isinstance(value, float)
        assert math.isfinite(value)


def test_evaluation_log_contains_only_finite_scalars_regression(client):
    detail = _run_experiment(client, "diabetes", "ridge_regression", {"max_iter": 20})
    metrics = _entries_by_event(detail, "evaluation_completed")[0]["details"]["metrics"]
    assert metrics
    assert set(metrics) <= set(REGRESSION_LOG_METRICS)
    for value in metrics.values():
        assert isinstance(value, float)
        assert math.isfinite(value)


def test_process_log_is_json_serializable(client):
    detail = _run_experiment(client, "iris", "logistic_regression", {"max_iter": 20})
    payload = json.dumps(detail["training_info"]["process_log"])
    parsed = json.loads(payload)
    assert len(parsed) == len(detail["training_info"]["process_log"])


def test_event_order_and_sequence(client):
    detail = _run_experiment(client, "iris", "logistic_regression", {"max_iter": 10})
    log = detail["training_info"]["process_log"]
    assert [entry["sequence"] for entry in log] == list(range(1, len(log) + 1))
    events = [entry["event"] for entry in log]
    assert events[:4] == [
        "dataset_loaded",
        "data_prepared",
        "estimator_created",
        "training_started",
    ]
    assert events[-4:] == [
        "training_completed",
        "prediction_completed",
        "evaluation_completed",
        "experiment_completed",
    ]
    assert set(events[4:-4]) == {"iteration_checkpoint"}


def test_summary_json_process_log_matches_api(client):
    detail = _run_experiment(client, "iris", "logistic_regression", {"max_iter": 20})
    summary_path = (
        Path(get_settings().experiments_dir) / detail["artifact_dir"] / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["training_info"]["process_log"] == detail["training_info"]["process_log"]


def test_hyperparameters_and_resolved_parameters_untouched(client):
    detail = _run_experiment(
        client, "iris", "logistic_regression", {"max_iter": 20, "learning_rate": 0.1}
    )
    assert detail["hyperparameters"] == {"max_iter": 20, "learning_rate": 0.1}
    resolved = detail["training_info"]["resolved_parameters"]
    assert isinstance(resolved, dict)
    assert resolved["max_iter"] == 20
    assert resolved["learning_rate"] == 0.1


def test_failed_experiment_keeps_existing_error_semantics(client):
    created = client.post(
        "/api/v1/experiments",
        json={"dataset_name": "does_not_exist", "model_name": "logistic_regression"},
    ).json()
    detail = client.get(f"/api/v1/experiments/{created['id']}").json()
    assert detail["status"] == "failed"
    assert "does_not_exist" in (detail["error"] or "")
    assert detail["training_info"] is None
