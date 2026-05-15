from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional at import time
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    app_name: str = os.getenv("APP_NAME", "autoData API")
    api_prefix: str = os.getenv("API_PREFIX", "/api")
    environment: str = os.getenv("ENVIRONMENT", "development")

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./autodata.db")
    auto_create_tables: bool = os.getenv("AUTO_CREATE_TABLES", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    cors_origins: list[str] = _split_csv(
        os.getenv(
            "BACKEND_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        )
    )

    model_dir: Path = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models" / "prod"))
    processed_data_path: Path = Path(
        os.getenv("PROCESSED_DATA_PATH", ROOT_DIR / "data" / "processed" / "processed_data.csv")
    )
    raw_data_path: Path = Path(os.getenv("RAW_DATA_PATH", ROOT_DIR / "data" / "raw" / "stock_data.csv"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
