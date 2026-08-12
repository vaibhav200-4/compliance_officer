from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.compliance.retriever import ComplianceRetriever


def main():
    print("=" * 70)
    print("COMPLIANCE RETRIEVER TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load GDPR knowledge base
    # ---------------------------------------------------------

    print("\n[1] Loading GDPR knowledge base...")

    kb = GDPRKnowledgeBase()

    obligation = kb.require_sub_obligation(
        "5.1.e.1"
    )

    print("SUCCESS")
    print(f"Obligation ID: {obligation.id}")

    # ---------------------------------------------------------
    # 2. Initialize retriever
    # ---------------------------------------------------------

    print("\n[2] Initializing compliance retriever...")

    retriever = ComplianceRetriever()

    print("SUCCESS")

    # ---------------------------------------------------------
    # 3. Retrieve evidence
    # ---------------------------------------------------------

    print("\n[3] Retrieving policy evidence...")
    print("-" * 70)

    print("Evidence prompt:")
    print(obligation.evidence_prompt)

    results = retriever.retrieve(
        query=obligation.evidence_prompt,
        top_k=5,
    )

    # ---------------------------------------------------------
    # 4. Display results
    # ---------------------------------------------------------

    print("\n[4] Retrieved chunks")
    print("-" * 70)

    if not results:
        print("NO RESULTS FOUND")
    else:

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(f"\nResult #{index}")
            print(f"Chunk ID : {result.chunk_id}")
            print(f"Score    : {result.score:.4f}")

            print("\nMetadata:")
            print(result.metadata)

            print("\nText:")
            print(result.text[:1000])

    # ---------------------------------------------------------
    # 5. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)

    if results:
        print(
            f"RETRIEVER TEST PASSED "
            f"({len(results)} chunks retrieved)"
        )
    else:
        print(
            "RETRIEVER TEST COMPLETED "
            "(Pinecone returned no usable chunks)"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()