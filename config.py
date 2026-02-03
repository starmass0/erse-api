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

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Free, local
    embedding_dim: int = 384

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
