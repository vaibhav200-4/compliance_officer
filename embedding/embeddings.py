"""

Input : policy_chunks.json (from chunk_policy.py)
Output: chunks_with_embeddings.json (same chunks + "values" key added)

Usage:
    python generate_embeddings.py --chunks data/policy_chunks.json --out data/chunks_embedded.json
"""

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "jinaai/jina-embeddings-v3"  # change to multilingual model if needed


def generate_embeddings(chunks: list, model_name: str = MODEL_NAME):
    print(f"Loading embedding model: {model_name} ...")

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True
    )

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")

    vectors = model.encode(
        texts,
        task="retrieval.passage",
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )

    for chunk, vector in zip(chunks, vectors):
        chunk["values"] = vector.tolist()

    return chunks, model.get_sentence_embedding_dimension()

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for policy chunks")
    parser.add_argument("--chunks", required=True, help="Path to policy_chunks.json (from chunk_policy.py)")
    parser.add_argument("--out", required=True, help="Path to save chunks + embeddings as .json")
    parser.add_argument("--model", default=MODEL_NAME, help="sentence-transformers model name")
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    embedded_chunks, dim = generate_embeddings(chunks, args.model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(embedded_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Embedding dimension: {dim}  <-- use this when creating your Pinecone index")
    print(f"Saved {len(embedded_chunks)} embedded chunks to: {out_path}")


if __name__ == "__main__":
    main()