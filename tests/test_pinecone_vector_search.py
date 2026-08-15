"""Integration test for vector-based Pinecone search."""
from __future__ import annotations

import json
from pathlib import Path

from app.vectorstore.pinecone import PineconeManager


GDPR_EMBEDDINGS_PATH = Path("data/gdpr/gdpr_group_embeddings.json")
GDPR_GROUP_KEY = "5:5.1.f"


def test_similarity_search_by_cached_gdpr_vector_returns_results() -> None:
    manager = PineconeManager()

    with GDPR_EMBEDDINGS_PATH.open(encoding="utf-8") as file:
        gdpr_embeddings = json.load(file)

    vector = gdpr_embeddings["groups"][GDPR_GROUP_KEY]["embedding"]
    response = manager.similarity_search_by_vector(
        vector,
        namespace="company-policy",
        top_k=5,
    )
    matches = response.get("matches", []) if isinstance(response, dict) else response.matches

    for match in matches:
        match_id = match.get("id") if isinstance(match, dict) else match.id
        score = match.get("score") if isinstance(match, dict) else match.score
        metadata = match.get("metadata", {}) if isinstance(match, dict) else match.metadata
        print(f"id={match_id} score={score} text={metadata.get('text', '')}")

    assert matches
