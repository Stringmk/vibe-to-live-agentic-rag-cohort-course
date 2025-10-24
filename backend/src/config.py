from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Optional

def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _parse_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str
    debug: bool
    host: str
    port: int
    database_url: Optional[str]
    openai_api_key: Optional[str]
    openai_api_endpoint: Optional[str]
    qdrant_api_key: Optional[str]
    qdrant_api_endpoint: Optional[str]
    phoenix_api_key: Optional[str]
    phoenix_api_endpoint: Optional[str]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load application settings from environment variables and cache the result.

    Environment variables:
    - APP_NAME (str)
    - DEBUG (bool-like: 1/true/yes/on)
    - HOST (str)
    - PORT (int)
    - DATABASE_URL (str)
    - OPENAI_API_KEY (str)
    - OPENAI_BASE_URL / OPENAI_API_ENDPOINT (str)
    - QDRANT_API_KEY (str)
    - QDRANT_API_ENDPOINT (str)
    - PHOENIX_API_KEY (str)
    - PHOENIX_API_ENDPOINT (str)
    """
    env = os.environ
    return Settings(
        app_name=env.get("APP_NAME", "OpenAI Agentic RAG API"),
        debug=_parse_bool(env.get("DEBUG"), False),
        host=env.get("HOST", "0.0.0.0"),
        port=_parse_int(env.get("PORT"), 8000),
        database_url=env.get("DATABASE_URL"),
        openai_api_key=env.get("OPENAI_API_KEY"),
        openai_api_endpoint=env.get("OPENAI_BASE_URL"),
        qdrant_api_key=env.get("QDRANT_API_KEY"),
        qdrant_api_endpoint=env.get("QDRANT_API_ENDPOINT"),
        phoenix_api_key=env.get("PHOENIX_API_KEY"),
        phoenix_api_endpoint=env.get("PHOENIX_API_ENDPOINT"),
    )