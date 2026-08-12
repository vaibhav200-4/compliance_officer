from app.compliance.group_retriever import (
    ComplianceGroupRetriever,
)


def main():

    print("=" * 70)
    print("COMPLIANCE GROUP RETRIEVER TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Initialize
    # ---------------------------------------------------------

    print("\n[1] Initializing group retriever...")

    retriever = ComplianceGroupRetriever()

    print("SUCCESS")

    # ---------------------------------------------------------
    # 2. Get Article 5 groups
    # ---------------------------------------------------------

    print("\n[2] Article 5 groups")
    print("-" * 70)

    groups = retriever.get_groups(5)

    for group in groups:

        print(
            f"{group.group_id:<10} "
            f"→ {group.obligation_count} obligations"
        )

    # ---------------------------------------------------------
    # 3. Select security group
    # ---------------------------------------------------------

    print("\n[3] Loading group 5.1.f")
    print("-" * 70)

    group = retriever.get_group(
        5,
        "5.1.f",
    )

    print(
        f"Article       : {group.article_number}"
    )

    print(
        f"Group         : {group.group_id}"
    )

    print(
        f"Obligations   : {group.obligation_count}"
    )

    for obligation in group.obligations:

        print(
            f"\n{obligation.id}"
        )

        print(
            f"Summary: {obligation.plain_summary}"
        )

        print(
            f"Evidence: {obligation.evidence_prompt}"
        )

    # ---------------------------------------------------------
    # 4. Build combined query
    # ---------------------------------------------------------

    print("\n[4] Combined retrieval query")
    print("-" * 70)

    query = retriever.build_group_query(
        group
    )

    print(query)

    # ---------------------------------------------------------
    # 5. Retrieve shared evidence
    # ---------------------------------------------------------

    print("\n[5] Retrieving shared evidence")
    print("-" * 70)

    result = retriever.retrieve_group(
        article_number=5,
        group_id="5.1.f",
        top_k=5,
    )

    print(
        f"\nEvidence chunks: "
        f"{result.evidence_count}"
    )

    # ---------------------------------------------------------
    # 6. Display evidence
    # ---------------------------------------------------------

    for index, evidence in enumerate(
        result.evidence,
        start=1,
    ):

        print(
            f"\nResult #{index}"
        )

        print(
            f"Chunk ID : {evidence.chunk_id}"
        )

        print(
            f"Score    : {evidence.score:.4f}"
        )

        print(
            f"Page     : "
            f"{evidence.metadata.get('page')}"
        )

        print("\nText:")
        print(
            evidence.text[:1200]
        )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GROUP RETRIEVER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()