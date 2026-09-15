from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MLLab"
    environment: str = "development"
    debug: bool = True
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./mllab.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MLLAB_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

