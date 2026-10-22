from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root — backend/core/config.py → backend/ → <root>/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "MLLab"
    environment: str = "development"
    debug: bool = True
    api_prefix: str = "/api/v1"
    database_url: str = f"sqlite:///{_PROJECT_ROOT / 'mllab.db'}"
    experiments_dir: str = str(_PROJECT_ROOT / "experiments")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MLLAB_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

