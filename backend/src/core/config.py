from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings loaded from codespaces environment variables.
    Would need a .env if not running codespaces
    """
    QDRANT_URL: str
    QDRANT_API_KEY: str
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str

def get_settings() -> Settings:
    """
    Returns a singleton instance of Settings.
    Loads codespaces env variables if present.
    """
    return Settings()