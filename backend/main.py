from fastapi import FastAPI

from backend.api.router import api_router
from backend.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        version="0.1.0",
        description="Extensible machine-learning experiment platform API.",
    )
    application.include_router(api_router, prefix=settings.api_prefix)

    @application.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "status": "running",
            "docs": "/docs",
        }

    return application


app = create_app()

