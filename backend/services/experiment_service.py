"""Experiment orchestration: creation, execution, and queries."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.database import session as db_session
from backend.database.models import Experiment, ExperimentStatus
from backend.services import ml_bridge, model_explanation
from backend.services.ml_contracts import TASK_REGRESSION
from backend.services.training_log import (
    CLASSIFICATION_LOG_METRICS,
    REGRESSION_LOG_METRICS,
    TrainingProcessLog,
    finite_scalar_metrics,
    sample_iteration_checkpoints,
)


def create_experiment(db: Session, payload: Any) -> Experiment:
    """Persist a new experiment in ``pending`` state (execution happens in background)."""
    name = payload.name or f"{payload.model_name} on {payload.dataset_name}"
    experiment = Experiment(
        name=name,
        dataset_name=payload.dataset_name,
        model_name=payload.model_name,
        hyperparameters=dict(payload.hyperparameters),
        test_size=payload.test_size,
        random_state=payload.random_state,
        status=ExperimentStatus.PENDING.value,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def list_experiments(db: Session) -> list[Experiment]:
    statement = select(Experiment).order_by(Experiment.id.desc())
    return list(db.scalars(statement))


def get_experiment(db: Session, experiment_id: int) -> Experiment | None:
    return db.get(Experiment, experiment_id)


def delete_experiment(db: Session, experiment_id: int) -> bool:
    """Delete an experiment and its artifacts. Returns False if it does not exist."""
    experiment = get_experiment(db, experiment_id)
    if experiment is None:
        return False
    if experiment.artifact_dir:
        artifact_path = Path(get_settings().experiments_dir) / experiment.artifact_dir
        shutil.rmtree(artifact_path, ignore_errors=True)
    db.delete(experiment)
    db.commit()
    return True


def run_experiment(experiment_id: int) -> None:
    """Entry point for the background task executing one experiment."""
    db = db_session.SessionLocal()
    try:
        experiment = db.get(Experiment, experiment_id)
        if experiment is None or experiment.status != ExperimentStatus.PENDING.value:
            return
        experiment.status = ExperimentStatus.RUNNING.value
        experiment.started_at = datetime.now(timezone.utc)
        db.commit()
        try:
            result = _execute_pipeline(experiment)
        except Exception as exc:  # noqa: BLE001 - any failure marks the experiment
            experiment.error = str(exc)[:2000]
            experiment.status = ExperimentStatus.FAILED.value
            experiment.finished_at = datetime.now(timezone.utc)
            db.commit()
            return
        experiment.metrics = result["metrics"]
        experiment.confusion_matrix = result["confusion_matrix"]
        experiment.class_labels = result["class_labels"]
        experiment.predictions = result["predictions"]
        experiment.training_info = result["training_info"]
        experiment.charts = result["charts"]
        experiment.task_type = result["task_type"]
        experiment.artifact_dir = result["artifact_dir"]
        experiment.status = ExperimentStatus.COMPLETED.value
        experiment.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def _execute_pipeline(experiment: Experiment) -> dict[str, Any]:
    """Run the train/evaluate pipeline; raises on any failure."""
    pipeline_log = TrainingProcessLog()
    try:
        data = ml_bridge.load_dataset(experiment.dataset_name)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {experiment.dataset_name!r}") from exc
    task_type = data.task_type
    pipeline_log.add(
        "dataset_loaded",
        {
            "dataset_name": experiment.dataset_name,
            "sample_count": int(len(data.y)),
            "feature_count": int(np.asarray(data.X).shape[1]),
            "task_type": task_type,
        },
    )

    try:
        model = ml_bridge.model_detail(experiment.model_name)
    except KeyError as exc:
        raise ValueError(f"Unknown model: {experiment.model_name!r}") from exc
    if model["task_type"] != task_type:
        raise ValueError(
            f"Model {experiment.model_name!r} supports {model['task_type']!r} tasks, "
            f"but dataset {experiment.dataset_name!r} is {task_type!r}."
        )

    # Standardization is not user-configurable yet; record the value actually
    # passed so training_info reflects the real preprocessing configuration.
    standardize = True
    X_train, X_test, y_train, y_test = ml_bridge.split_dataset(
        data.X,
        data.y,
        test_size=experiment.test_size,
        random_state=experiment.random_state,
        task_type=task_type,
        standardize=standardize,
    )
    pipeline_log.add(
        "data_prepared",
        {
            "train_sample_count": int(len(X_train)),
            "test_sample_count": int(len(X_test)),
            "test_size": float(experiment.test_size),
            "random_state": int(experiment.random_state),
            "standardize": bool(standardize),
            "stratified": task_type == "classification",
        },
    )

    estimator = ml_bridge.create_estimator(
        experiment.model_name, experiment.hyperparameters,
        random_state=experiment.random_state,
    )
    pipeline_log.add(
        "estimator_created",
        {
            "model_name": experiment.model_name,
            "estimator_class": type(estimator).__name__,
        },
    )
    pipeline_log.add(
        "training_started",
        {
            "model_name": experiment.model_name,
            "train_sample_count": int(len(X_train)),
        },
    )
    fit_start = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_duration_ms = (time.perf_counter() - fit_start) * 1000.0

    # ``get_training_info`` only reads the fitted estimator, so it can run
    # before prediction; this keeps the process log in chronological order.
    feature_names = list(data.feature_names) if hasattr(data, "feature_names") else None
    training_info = _jsonable(
        ml_bridge.get_training_info(
            estimator,
            feature_names=feature_names,
            fit_duration_ms=fit_duration_ms,
            standardize=standardize,
            train_sample_count=len(X_train),
            test_sample_count=len(X_test),
        )
    )

    pipeline_log.extend_iteration_checkpoints(
        sample_iteration_checkpoints(
            training_info.get("loss_history"),
            training_info.get("loss_name"),
            training_info.get("loss_source"),
        )
    )
    loss_history = training_info.get("loss_history")
    pipeline_log.add(
        "training_completed",
        {
            "fit_duration_ms": fit_duration_ms,
            "n_iter": training_info.get("n_iter"),
            "stop_reason": training_info.get("stop_reason"),
            "loss_point_count": len(loss_history) if isinstance(loss_history, list) else 0,
        },
    )

    predict_start = time.perf_counter()
    y_pred = estimator.predict(X_test)
    predict_duration_ms = (time.perf_counter() - predict_start) * 1000.0
    pipeline_log.add(
        "prediction_completed",
        {
            "prediction_count": int(len(y_pred)),
            "predict_duration_ms": predict_duration_ms,
        },
    )

    score_info = ml_bridge.get_predict_scores(estimator, X_test)
    class_labels = None if task_type == TASK_REGRESSION else list(data.target_names)
    score_class_names = _score_class_names(task_type, score_info, data.target_names)
    eval_start = time.perf_counter()
    evaluation = _jsonable(
        ml_bridge.evaluate(
            task_type,
            y_test,
            y_pred,
            y_score=score_info.get("y_score") if score_info else None,
            class_labels=class_labels,
        )
    )
    evaluation_duration_ms = (time.perf_counter() - eval_start) * 1000.0
    metrics = evaluation["metrics"]
    allowed_metrics = (
        REGRESSION_LOG_METRICS if task_type == TASK_REGRESSION else CLASSIFICATION_LOG_METRICS
    )
    pipeline_log.add(
        "evaluation_completed",
        {
            "evaluation_duration_ms": evaluation_duration_ms,
            "metrics": finite_scalar_metrics(metrics, allowed_metrics),
        },
    )
    confusion_matrix = evaluation.get("confusion_matrix")
    eval_class_labels = evaluation.get("class_labels")
    per_class_metrics = evaluation.get("per_class_metrics")

    # Predictions block: regression keeps the original compact form;
    # classification stores full y_true / y_pred plus score info and
    # per-class metrics for downstream visualization.
    predictions = None
    if task_type == TASK_REGRESSION:
        predictions = {
            "y_true": _jsonable(list(y_test)),
            "y_pred": _jsonable(list(y_pred)),
        }
    else:
        predictions = {
            "y_true": _jsonable(list(y_test)),
            "y_pred": _jsonable(list(y_pred)),
            "y_score": _jsonable(score_info["y_score"]) if score_info else None,
            "score_type": score_info["score_type"] if score_info else None,
            "score_classes": _jsonable(score_info["score_classes"]) if score_info else None,
            "score_class_names": score_class_names,
            "per_class_metrics": per_class_metrics,
        }

    # Real per-model explanation data (None for models without a faithful
    # explanation in this phase). Failures propagate instead of storing
    # invented values.
    explanation_data = model_explanation.build_explanation_data(
        model_name=experiment.model_name,
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        feature_names=feature_names,
        random_state=experiment.random_state,
    )
    if explanation_data is not None:
        training_info["explanation_data"] = _jsonable(explanation_data)

    charts: list[dict[str, Any]] = []
    artifact_subdir = str(experiment.id)
    artifact_dir = Path(get_settings().experiments_dir) / artifact_subdir
    try:
        specs = ml_bridge.build_charts(task_type, metrics, y_test, y_pred)
        charts = [
            {"file": file_name, **spec}
            for file_name, spec in zip(
                ml_bridge.save_charts(specs, artifact_dir), specs, strict=False
            )
        ]
    except Exception:  # charts are optional; keep the metrics result
        charts = []

    pipeline_log.add(
        "experiment_completed",
        {
            "total_duration_ms": pipeline_log.elapsed_ms(),
        },
    )
    training_info["process_log"] = pipeline_log.entries()

    _write_summary(
        artifact_dir,
        experiment,
        task_type,
        metrics,
        [c["file"] for c in charts],
        training_info,
    )
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "class_labels": eval_class_labels,
        "per_class_metrics": per_class_metrics,
        "predictions": predictions,
        "training_info": training_info,
        "charts": charts,
        "task_type": task_type,
        "artifact_dir": artifact_subdir,
    }


def _score_class_names(
    task_type: str,
    score_info: dict[str, Any] | None,
    target_names: Any,
) -> list[str] | None:
    """Display-only class names for classification score columns.

    Regression experiments and experiments without prediction scores never
    receive names. The mapping itself is validated in
    :func:`ml_bridge.resolve_score_class_names`.
    """
    if task_type == TASK_REGRESSION or not score_info:
        return None
    return ml_bridge.resolve_score_class_names(
        score_info.get("score_classes"), target_names
    )


def _write_summary(
    artifact_dir: Path,
    experiment: Experiment,
    task_type: str,
    metrics: dict[str, Any],
    chart_files: list[str],
    training_info: dict[str, Any],
) -> None:
    summary = {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "dataset": experiment.dataset_name,
        "model": experiment.model_name,
        "task_type": task_type,
        "hyperparameters": experiment.hyperparameters,
        "test_size": experiment.test_size,
        "random_state": experiment.random_state,
        "metrics": metrics,
        "training_info": training_info,
        "charts": chart_files,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "summary.json"
    payload = json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    path.write_text(payload, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars etc. into plain JSON types."""
    return json.loads(json.dumps(value, default=str))
