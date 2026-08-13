"""
embedding_gemini.py
-------------------

Generates Gemini embeddings for GDPR/RAG chunks.

Input:
    gdpr_chunks.json

Output:
    gdpr_chunks_embedded_gemini.json

Model:
    gemini-embedding-001

Embedding dimension:
    3072

Usage:
    python embedding_gemini.py \
        --chunks gdpr_chunks.json \
        --out gdpr_chunks_embedded_gemini.json
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072

# Number of texts sent in one API request
BATCH_SIZE = 20


def generate_embeddings(chunks: list):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not found in .env file."
        )

    print(f"Loading Gemini embedding model: {MODEL_NAME}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")

    client = genai.Client(api_key=GEMINI_API_KEY)

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")

    all_vectors = []

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Generating Gemini embeddings"
    ):
        batch = texts[start:start + BATCH_SIZE]

        result = client.models.embed_content(
            model=MODEL_NAME,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )

        for embedding in result.embeddings:
            vector = embedding.values

            if len(vector) != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Unexpected embedding dimension: {len(vector)} "
                    f"(expected {EMBEDDING_DIMENSION})"
                )

            all_vectors.append(vector)

        # Small delay between batches
        time.sleep(0.2)

    if len(all_vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: "
            f"{len(all_vectors)} vectors for {len(chunks)} chunks"
        )

    for chunk, vector in zip(chunks, all_vectors):
        chunk["values"] = vector

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gemini embeddings for GDPR chunks"
    )

    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to input chunks JSON"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Path to output embedded chunks JSON"
    )

    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    out_path = Path(args.out)

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {chunks_path}"
        )

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    embedded_chunks = generate_embeddings(chunks)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path.write_text(
        json.dumps(
            embedded_chunks,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("Gemini embedding generation completed")
    print("=" * 60)
    print(f"Model:      {MODEL_NAME}")
    print(f"Dimension:  {EMBEDDING_DIMENSION}")
    print(f"Chunks:     {len(embedded_chunks)}")
    print(f"Output:     {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()