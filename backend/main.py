from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import database
from backend.api.router import api_router
from backend.core.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.init_db()
    yield


def _format_validation_error(exc: RequestValidationError) -> str:
    """Flatten FastAPI's validation error list into a single message string.

    FastAPI returns ``{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}``
    by default. The frontend expects ``{"detail": "<message>"}``.
    """
    messages = []
    for error in exc.errors():
        loc = " -> ".join(str(part) for part in error.get("loc", []) if part != "body")
        msg = error.get("msg", "")
        messages.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(messages) if messages else "Request validation failed"


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        version="0.1.0",
        description="Extensible machine-learning experiment platform API.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _format_validation_error(exc)})

    application.include_router(api_router, prefix=settings.api_prefix)

    # Serve experiment artifacts (chart PNGs, summary.json) so the frontend
    # can render them. The directory is created by ``init_db``.
    artifacts_dir = Path(settings.experiments_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/artifacts",
        StaticFiles(directory=str(artifacts_dir)),
        name="artifacts",
    )

    @application.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "status": "running",
            "docs": "/docs",
        }

    return application


app = create_app()
