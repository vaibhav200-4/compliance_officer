from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from .env
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    MISTRAL_API_KEY: str = ""
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    OPENROUTER_API_KEY: str = ""

    
    # Pinecone
    PINECONE_INDEX_NAME: str
    PINECONE_NAMESPACE: str = "company-policy"

    # Models
    LLM_MODEL: str = ""
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    EMBEDDING_DIMENSION: int = Field(default=3072, gt=0)

    # Retrieval
    TOP_K: int = Field(default=5)

    # Chunking
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # Application
    APP_NAME: str = "ComplianceIQ"
    APP_VERSION: str = "0.1.0"
    # APP_DEBUG avoids collisions with generic host environment variables such as DEBUG=release.
    DEBUG: bool = Field(default=False, validation_alias="APP_DEBUG")

    # LLM Configuration
    LLM_PROVIDER: str = ""  # Empty = all configured providers; "nvidia" = NVIDIA only, "openrouter" = OpenRouter only
    LLM_REQUEST_TIMEOUT: int = Field(default=30, gt=0)  # seconds
    LLM_MAX_RETRIES: int = Field(default=3, gt=0)  # maximum retry attempts
    MAX_CONCURRENT_LLM_REQUESTS: int = Field(default=4, gt=0)  # global max active LLM calls

    # Orchestration Configuration
    ARTICLE_WORKERS: int = Field(default=5, gt=0)  # article-level concurrency
    GROUP_WORKERS: int = Field(default=1, gt=0)  # group-level concurrency within each article


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()


settings = get_settings()
