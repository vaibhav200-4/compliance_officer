"""
delete_all_vectors.py
----------------------
Deletes ALL vectors from a Pinecone index (optionally scoped to one namespace).
Run this before re-uploading with a clean, namespaced upload script.
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "generic-rag")

# Leave as None to wipe the default namespace ("__default__").
# Set to a string (e.g. "policies") to wipe only that namespace instead.
NAMESPACE = None


def main():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not found in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    before = index.describe_index_stats()
    print("Before delete:", before)

    if NAMESPACE == "__ALL__":
        # Wipe every namespace that currently exists in the index.
        namespaces = list(before.get("namespaces", {}).keys())
        if not namespaces:
            print("No namespaces found, nothing to delete.")
        for ns in namespaces:
            index.delete(delete_all=True, namespace=ns)
            print(f"Deleted all vectors in namespace '{ns}'.")
    elif NAMESPACE:
        index.delete(delete_all=True, namespace=NAMESPACE)
        print(f"Deleted all vectors in namespace '{NAMESPACE}'.")
    else:
        # Deletes everything in the default namespace only.
        index.delete(delete_all=True)
        print("Deleted all vectors in the default namespace.")

    after = index.describe_index_stats()
    print("After delete:", after)


if __name__ == "__main__":
    main()