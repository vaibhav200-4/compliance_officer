from pathlib import Path

from app.core.logger import get_logger
from app.ingestion.pipeline import IngestionPipeline
from app.vectorstore.pinecone import (
    COMPANY_POLICY_NAMESPACE,
    PineconeManager,
)

logger = get_logger()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Change this to your sample policy file
SAMPLE_POLICY = Path("data/SampleInput_CompanyPolicy.pdf")


def main():
    print("=" * 70)
    print("RESET COMPANY POLICY VECTOR STORE + SAMPLE INGESTION")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Check sample file
    # -----------------------------------------------------

    print("\n[1] Checking sample policy...")

    if not SAMPLE_POLICY.exists():
        raise FileNotFoundError(
            f"Sample policy not found: {SAMPLE_POLICY.resolve()}"
        )

    print(f"File: {SAMPLE_POLICY.resolve()}")
    print("SUCCESS")

    # -----------------------------------------------------
    # 2. Connect to Pinecone
    # -----------------------------------------------------

    print("\n[2] Connecting to Pinecone...")

    vector_store = PineconeManager()

    print("SUCCESS")
    print(f"Index     : {vector_store.index_description.name}")
    print(f"Namespace : {COMPANY_POLICY_NAMESPACE}")

    # -----------------------------------------------------
    # 3. Show current namespace statistics
    # -----------------------------------------------------

    print("\n[3] Current vector store statistics...")
    print("-" * 70)

    stats_before = vector_store.describe_index_stats()

    print(stats_before)

    # -----------------------------------------------------
    # 4. Clean company-policy namespace
    # -----------------------------------------------------

    print("\n[4] Cleaning company-policy namespace...")

    vector_store.index.delete(
        delete_all=True,
        namespace=COMPANY_POLICY_NAMESPACE,
    )

    print("DELETE REQUEST SENT")

    # -----------------------------------------------------
    # 5. Verify namespace is empty
    # -----------------------------------------------------

    print("\n[5] Verifying namespace cleanup...")

    stats_after_delete = vector_store.describe_index_stats()

    print(stats_after_delete)

    # -----------------------------------------------------
    # 6. Ingest sample policy
    # -----------------------------------------------------

    print("\n[6] Ingesting sample policy...")

    pipeline = IngestionPipeline()

    result = pipeline.ingest(
        SAMPLE_POLICY
    )

    print("\nINGESTION RESULT")
    print("-" * 70)

    print(f"Files       : {result['files']}")
    print(f"Documents   : {result['documents']}")
    print(f"Chunks      : {result['chunks']}")
    print(f"Namespace   : {result['namespace']}")
    print(f"Document IDs: {result['document_ids']}")

    # -----------------------------------------------------
    # 7. Final Pinecone statistics
    # -----------------------------------------------------

    print("\n[7] Final vector store statistics...")
    print("-" * 70)

    final_stats = vector_store.describe_index_stats()

    print(final_stats)

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RESET + SAMPLE INGESTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()