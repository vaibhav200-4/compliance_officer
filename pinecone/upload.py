import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "generic-rag")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

BATCH_SIZE = 100  # Pinecone recommends batching upserts


def get_or_create_index(pc: Pinecone, index_name: str, dimension: int):
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Index '{index_name}' not found. Creating with dimension={dimension} ...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
    else:
        print(f"Using existing index: {index_name}")
    return pc.Index(index_name)


def upsert_chunks(index, chunks: list, namespace: str):
    vectors = [
        {"id": c["id"], "values": c["values"], "metadata": c["metadata"]}
        for c in chunks
    ]

    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc=f"Upserting to '{namespace}'"):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)


def main():
    parser = argparse.ArgumentParser(description="Upload embedded policy chunks to Pinecone")
    parser.add_argument("--chunks", required=True, help="Path to chunks_embedded.json")
    parser.add_argument("--dim", type=int, required=True,
                         help="Embedding dimension (printed by generate_embeddings.py)")
    parser.add_argument("--namespace", required=True,
                         help="Pinecone namespace to upsert into, e.g. 'policies' or 'gdpr_articles'")
    args = parser.parse_args()

    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY not found. Copy .env.example to .env and fill in your key."
        )

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc, INDEX_NAME, args.dim)
    upsert_chunks(index, chunks, namespace=args.namespace)

    stats = index.describe_index_stats()
    ns_count = stats["namespaces"].get(args.namespace, {}).get("vector_count", 0)
    print(f"Done. Namespace '{args.namespace}' now has {ns_count} vectors "
          f"(index total: {stats['total_vector_count']}).")


if __name__ == "__main__":
    main()