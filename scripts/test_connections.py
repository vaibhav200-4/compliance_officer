"""Verify Mistral, Google embeddings, and Pinecone without ingesting a file."""

from __future__ import annotations

import sys

from mistralai.client import Mistral
from pinecone import Pinecone

def _is_configured(value: str) -> bool:
    return bool(value and not value.startswith("your_"))


def main() -> int:
    print("=" * 50)
    print("COMPLIANCEIQ CONNECTION TEST")
    print("=" * 50)

    try:
        from app.core.config import settings
        from app.ingestion.embedder import Embedder
    except Exception as exc:
        print(f"\n[CONFIGURATION]\n[FAIL] {type(exc).__name__}: {exc}")
        print("\n" + "=" * 50)
        print("CONNECTION TEST FAILED: configuration")
        print("=" * 50)
        return 1

    failures: list[str] = []
    embedding_dimension: int | None = None

    print("\n[MISTRAL]")
    try:
        if not _is_configured(settings.MISTRAL_API_KEY):
            raise ValueError("MISTRAL_API_KEY is missing or still a placeholder.")
        Mistral(api_key=settings.MISTRAL_API_KEY).models.list()
        print("[PASS] API connection successful")
    except Exception as exc:
        failures.append("Mistral")
        print(f"[FAIL] {type(exc).__name__}: {exc}")

    print("\n[GOOGLE EMBEDDINGS]")
    try:
        if not _is_configured(settings.GOOGLE_API_KEY):
            raise ValueError("GOOGLE_API_KEY is missing or still a placeholder.")
        embedder = Embedder()
        print("[PASS] Model initialized")
        vector = embedder.embed_query("Connection test")
        embedding_dimension = len(vector)
        print("[PASS] Embedding generated")
        print(f"[PASS] Dimension: {embedding_dimension}")
    except Exception as exc:
        failures.append("Google embeddings")
        print(f"[FAIL] {type(exc).__name__}: {exc}")

    print("\n[PINECONE]")
    try:
        if not _is_configured(settings.PINECONE_API_KEY):
            raise ValueError("PINECONE_API_KEY is missing or still a placeholder.")
        client = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_names = client.list_indexes().names()
        if settings.PINECONE_INDEX_NAME not in index_names:
            raise ValueError(f"Index does not exist: {settings.PINECONE_INDEX_NAME}")
        description = client.describe_index(settings.PINECONE_INDEX_NAME)
        dimension = description.dimension
        print("[PASS] API connection successful")
        print(f"[PASS] Index: {settings.PINECONE_INDEX_NAME}")
        print(f"[PASS] Dimension: {dimension}")
        if embedding_dimension is not None and dimension != embedding_dimension:
            raise ValueError(
                f"Index dimension ({dimension}) does not match embedding dimension "
                f"({embedding_dimension})."
            )
    except Exception as exc:
        failures.append("Pinecone")
        print(f"[FAIL] {type(exc).__name__}: {exc}")

    print("\n" + "=" * 50)
    if failures:
        print("CONNECTION TEST FAILED: " + ", ".join(failures))
        print("=" * 50)
        return 1

    print("ALL CONNECTIONS PASSED")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
