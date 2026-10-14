from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MLLab"
    environment: str = "development"
    debug: bool = True
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./mllab.db"
    experiments_dir: str = "experiments"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MLLAB_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

