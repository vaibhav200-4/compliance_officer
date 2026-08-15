"""Integration test for cached GDPR group retrieval."""
from __future__ import annotations

from app.compliance.group_retriever import ComplianceGroupRetriever


def test_article_5_group_5_1_f_retrieves_with_cached_embedding() -> None:
    retriever = ComplianceGroupRetriever()

    result = retriever.retrieve_group_by_cached_embedding(
        article_number=5,
        group_id="5.1.f",
        top_k=5,
    )

    cache = retriever.embedding_cache
    assert cache is not None

    vector = cache.get_embedding(
        5,
        "5.1.f",
    )

    print(f"group id={result.group_id}")
    print(f"evidence count={result.evidence_count}")

    for evidence in result.evidence:
        print(
            f"chunk id={evidence.chunk_id} "
            f"score={evidence.score} "
            f"text={evidence.text}"
        )

    assert result.evidence
    assert len(vector) == retriever.retriever.vector_store.index_dimension
