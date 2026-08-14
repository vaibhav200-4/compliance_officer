"""
upload_to_pinecone.py
------------------------
Step 4: Embedded chunks ko Pinecone index mein upsert karta hai.

IMPORTANT: API keys .env se load ho rahi hain, hardcode NAHI karni
(jaisa maine KisanSaathi mein bhi suggest kiya tha).

Input : chunks_embedded.json (from generate_embeddings.py)
Output: chunks Pinecone index mein upload ho jaate hain

Usage:
    python upload_to_pinecone.py --chunks data/chunks_embedded.json --dim 384
"""

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


def upsert_chunks(index, chunks: list):
    vectors = [
        {"id": c["id"], "values": c["values"], "metadata": c["metadata"]}
        for c in chunks
    ]

    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting to Pinecone"):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)


def main():
    parser = argparse.ArgumentParser(description="Upload embedded policy chunks to Pinecone")
    parser.add_argument("--chunks", required=True, help="Path to chunks_embedded.json")
    parser.add_argument("--dim", type=int, required=True,
                         help="Embedding dimension (printed by generate_embeddings.py)")
    args = parser.parse_args()

    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY not found. Copy .env.example to .env and fill in your key."
        )

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc, INDEX_NAME, args.dim)
    upsert_chunks(index, chunks)

    stats = index.describe_index_stats()
    print(f"Done. Index '{INDEX_NAME}' now has {stats['total_vector_count']} total vectors.")


if __name__ == "__main__":
    main()