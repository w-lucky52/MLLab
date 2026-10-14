"""Model catalog endpoints (backed by the ml_engine model registry)."""

from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas import ErrorResponse, ModelInfo
from backend.services import ml_bridge

router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=list[ModelInfo],
    responses={503: {"model": ErrorResponse}},
)
def list_models(
    task_type: str | None = Query(default=None, description="'classification' or 'regression'."),
) -> list[ModelInfo]:
    try:
        return [ModelInfo(**item) for item in ml_bridge.list_models(task_type)]
    except ml_bridge.MlEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/models/{name}",
    response_model=ModelInfo,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_model(name: str) -> ModelInfo:
    try:
        return ModelInfo(**ml_bridge.model_detail(name))
    except ml_bridge.MlEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}") from exc
