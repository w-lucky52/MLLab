"""Dataset catalog endpoints (backed by the ml_engine dataset registry)."""

from fastapi import APIRouter, HTTPException

from backend.api.schemas import DatasetDetail, DatasetInfo, ErrorResponse
from backend.services import ml_bridge

router = APIRouter(tags=["datasets"])


@router.get(
    "/datasets",
    response_model=list[DatasetInfo],
    responses={503: {"model": ErrorResponse}},
)
def list_datasets() -> list[DatasetInfo]:
    try:
        return [DatasetInfo(**item) for item in ml_bridge.list_datasets()]
    except ml_bridge.MlEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/datasets/{name}",
    response_model=DatasetDetail,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_dataset(name: str) -> DatasetDetail:
    try:
        return DatasetDetail(**ml_bridge.dataset_detail(name))
    except ml_bridge.MlEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {name!r}") from exc
