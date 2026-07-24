"""Application settings, loaded from environment variables and/or a .env file.

Priority (highest wins): real environment variable > .env file > default here.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Moon Classifier API"
    app_version: str = "1.0.0"
    api_key: str = "change-me"            # override via API_KEY env var / .env
    rate_limit_per_minute: int = 60       # override via RATE_LIMIT_PER_MINUTE


settings = Settings()