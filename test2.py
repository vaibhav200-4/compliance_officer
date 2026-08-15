from app.vectorstore.pinecone import (
    PineconeManager,
    COMPANY_POLICY_NAMESPACE,
)


def main():

    print("=" * 75)
    print("PINECONE DIAGNOSTIC TEST")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Connect
    # ---------------------------------------------------------

    print("\n[1] Connecting to Pinecone...")

    manager = PineconeManager()

    print("Connected.")

    # ---------------------------------------------------------
    # 2. Index statistics
    # ---------------------------------------------------------

    print("\n[2] Index statistics...")

    stats = manager.describe_index_stats()

    print(stats)

    # ---------------------------------------------------------
    # 3. Check company-policy namespace
    # ---------------------------------------------------------

    print("\n[3] Company-policy namespace...")

    try:
        namespace_stats = stats["namespaces"].get(
            COMPANY_POLICY_NAMESPACE
        )

        print(
            f"Namespace: {COMPANY_POLICY_NAMESPACE}"
        )

        print(
            f"Stats: {namespace_stats}"
        )

    except Exception:

        print(
            "Could not extract namespace statistics."
        )

    # ---------------------------------------------------------
    # 4. Run one direct query
    # ---------------------------------------------------------

    print("\n[4] Running direct Pinecone query...")

    query = """
    Does the privacy policy explain how personal data
    is collected, used, processed, protected and retained?
    """

    response = manager.similarity_search(
        query=query,
        namespace=COMPANY_POLICY_NAMESPACE,
        top_k=5,
    )

    # ---------------------------------------------------------
    # 5. Print RAW response
    # ---------------------------------------------------------

    print("\n[5] RAW PINECONE RESPONSE")
    print("-" * 75)

    print(response)

    # ---------------------------------------------------------
    # 6. Extract matches manually
    # ---------------------------------------------------------

    print("\n[6] MATCH DETAILS")
    print("-" * 75)

    matches = getattr(
        response,
        "matches",
        [],
    )

    print(
        f"Number of matches: {len(matches)}"
    )

    for index, match in enumerate(
        matches,
        start=1,
    ):

        print(
            f"\nMATCH #{index}"
        )

        print(
            f"Type: {type(match)}"
        )

        print(
            f"ID: {getattr(match, 'id', None)}"
        )

        print(
            f"Score: {getattr(match, 'score', None)}"
        )

        metadata = getattr(
            match,
            "metadata",
            None,
        )

        print(
            f"Metadata type: {type(metadata)}"
        )

        print(
            f"Metadata: {metadata}"
        )

        if metadata:

            print(
                f"Metadata keys: "
                f"{list(metadata.keys())}"
            )

            print(
                f"chunk_id: "
                f"{metadata.get('chunk_id')}"
            )

            print(
                f"text exists: "
                f"{'text' in metadata}"
            )

            if "text" in metadata:

                text = metadata["text"]

                print(
                    f"text length: "
                    f"{len(str(text))}"
                )

                print(
                    f"text preview:\n"
                    f"{str(text)[:500]}"
                )

    print("\n" + "=" * 75)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()