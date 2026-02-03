from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq (Free LLM)
    groq_api_key: str = ""

    # Qdrant Cloud
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "erse_regulations"

    # Optional: OpenAI (for production)
    openai_api_key: str = ""

    # Embedding settings (OpenAI)
    embedding_model: str = "text-embedding-3-small"  # Fast, high quality
    embedding_dim: int = 1536

    # LLM settings
    llm_provider: str = "openai"  # "openai" or "groq"
    llm_model: str = "gpt-4o-mini"  # Fast and good quality

    # App settings
    app_version: str = "2.0.0"
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
