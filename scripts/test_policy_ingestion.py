"""Ingest a real company privacy-policy PDF into only ``company-policy``."""

from __future__ import annotations

import sys
from pathlib import Path

POLICY_PATH = Path("data/policies/privacy_policy.pdf")
TEST_QUERY = "What personal data does the company collect?"
COMPANY_POLICY_NAMESPACE = "company-policy"


def _namespace_vector_count(stats) -> int:
    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})
    namespace_stats = (namespaces or {}).get(COMPANY_POLICY_NAMESPACE)
    if namespace_stats is None:
        return 0
    if hasattr(namespace_stats, "vector_count"):
        return namespace_stats.vector_count or 0
    return namespace_stats.get("vector_count", 0)


def main() -> int:
    print("=" * 50)
    print("COMPANY POLICY INGESTION TEST")
    print("=" * 50)

    if not POLICY_PATH.is_file():
        print(f"[FAIL] File not found: {POLICY_PATH}")
        print("Add a real company policy PDF; no test document was created.")
        return 1

    try:
        from app.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        result = pipeline.ingest(POLICY_PATH)
        document_id = result["document_ids"][0]
        print(f"\nFile:\n{POLICY_PATH.name}")
        print(f"\nDocument ID:\n{document_id}")
        print(f"\nDocuments:\n{result['documents']}")
        print(f"\nChunks:\n{result['chunks']}")
        print(f"\nNamespace:\n{result['namespace']}")
        print(f"\nVector IDs:\n{len(result['vector_ids'])}")

        manager = pipeline.vector_store
        vector_count = _namespace_vector_count(manager.describe_index_stats())
        if vector_count <= 0:
            raise RuntimeError("Pinecone has no vectors in namespace 'company-policy'.")
        print(f"\nVerified namespace vector count: {vector_count}")

        matches = manager.similarity_search(TEST_QUERY, namespace=COMPANY_POLICY_NAMESPACE)
        if not getattr(matches, "matches", None):
            raise RuntimeError("Company-policy similarity search returned no matches.")
        print("[PASS] Company-policy-only similarity search returned matches")
    except Exception as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        return 1

    print("\n" + "=" * 50)
    print("INGESTION SUCCESS")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
