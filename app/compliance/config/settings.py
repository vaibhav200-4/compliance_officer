"""Application settings."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------
    app_name: str = "ComplianceIQ"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False

    # ---------------------------------------------------------
    # Mistral
    # ---------------------------------------------------------
    mistral_api_key: str = Field(..., alias="MISTRAL_API_KEY")
    mistral_ocr_model: str = "mistral-ocr-latest"

    # ---------------------------------------------------------
    # Google
    # ---------------------------------------------------------
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    google_embedding_model: str = "gemini-embedding-001"
    google_embedding_dimensions: int = 3072

    # ---------------------------------------------------------
    # Pinecone
    # ---------------------------------------------------------
    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index_name: str = "complianceiq"
    pinecone_dimension: int = 3072
    pinecone_metric: str = "cosine"

    # Separate knowledge bases
    pinecone_gdpr_namespace: str = "gdpr"
    pinecone_policy_namespace: str = "company-policy"

    # ---------------------------------------------------------
    # OpenRouter
    # ---------------------------------------------------------
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")

    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    openrouter_model: str = Field(
        default="",
        alias="OPENROUTER_MODEL",
    )

    # ---------------------------------------------------------
    # RAG
    # ---------------------------------------------------------
    retrieval_top_k: int = 5

    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.40

    # ---------------------------------------------------------
    # Project paths
    # ---------------------------------------------------------
    gdpr_data_path: str = "data/gdpr"
    policy_data_path: str = "data/policies"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()