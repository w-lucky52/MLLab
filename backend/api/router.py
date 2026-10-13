from fastapi import APIRouter

from backend.api.routes import datasets, experiments, models
from backend.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(datasets.router)
api_router.include_router(models.router)
api_router.include_router(experiments.router)
