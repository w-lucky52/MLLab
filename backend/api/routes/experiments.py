"""Experiment endpoints: create, run (background), inspect, delete."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.schemas import ErrorResponse, ExperimentCreate, ExperimentOut
from backend.database import get_db
from backend.services import experiment_service

router = APIRouter(tags=["experiments"])


@router.post(
    "/experiments",
    response_model=ExperimentOut,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def create_experiment(
    payload: ExperimentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ExperimentOut:
    experiment = experiment_service.create_experiment(db, payload)
    background_tasks.add_task(experiment_service.run_experiment, experiment.id)
    return ExperimentOut.model_validate(experiment)


@router.get("/experiments", response_model=list[ExperimentOut])
def list_experiments(db: Session = Depends(get_db)) -> list[ExperimentOut]:
    experiments = experiment_service.list_experiments(db)
    return [ExperimentOut.model_validate(experiment) for experiment in experiments]


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentOut,
    responses={404: {"model": ErrorResponse}},
)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)) -> ExperimentOut:
    experiment = experiment_service.get_experiment(db, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return ExperimentOut.model_validate(experiment)


@router.delete(
    "/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)) -> None:
    if not experiment_service.delete_experiment(db, experiment_id):
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
