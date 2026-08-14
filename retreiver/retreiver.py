"""
retrieve_evidence_for_obligations.py
--------------------------------------
For EACH GDPR obligation (article_embeddings.npy + article_metadata.json),
finds the top-k most relevant policy chunks from Pinecone.

This is the reverse of compare_policy_with_articles.py: that script went
policy_chunk -> top articles. This one goes obligation -> top policy_chunks
(evidence), which is what the judge step needs.

Output: evidence_for_obligations.json
  [
    {
      "id": "gdpr_article_1_Article_1_1_",
      "text": "...",
      "article_number": 1,
      "article_name": "...",
      "chapter_number": 1,
      "section_name": "Article 1(1)",
      "severity": "MEDIUM",
      "evidence": [
        {"chunk_id": "xyz_chunk_3", "text": "...", "similarity": 0.81},
        ...
      ]
    },
    ...
  ]
"""

import json
import numpy as np
from pinecone import Pinecone

# ---------- CONFIG ----------
PINECONE_API_KEY = "pcsk_5VSGjr_9A67zH5nY6MRDFoWMuicMuBbBtcb5MgWiQ3cxTGH6DXnyz5aTtKjb6dUyg6piLy"
INDEX_NAME = "generic-rag"
COMPANY_NAME = "TechStartup Pvt Ltd"
ARTICLE_EMBEDDINGS_PATH = "../dataa/article_embeddings1.npy"
ARTICLE_METADATA_PATH = "../dataa/article_metadata1.json"
EMBEDDING_DIM = 1024
TOP_K_EVIDENCE = 3
OUTPUT_PATH = "../dataa/evidence_for_obligations.json"
# -----------------------------


def load_obligations(embeddings_path, metadata_path):
    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Mismatch: {embeddings.shape[0]} embeddings but {len(metadata)} metadata entries."
        )
    return embeddings, metadata


def fetch_policy_chunks_from_pinecone(index, company_name, dim):
    dummy_vector = [0.0] * dim
    results = index.query(
        vector=dummy_vector,
        top_k=1000,
        filter={"company": {"$eq": company_name}},
        include_values=True,
        include_metadata=True,
    )
    chunks = []
    for match in results["matches"]:
        chunks.append({
            "chunk_id": match["id"],
            "embedding": np.array(match["values"], dtype=np.float32),
            "text": match["metadata"].get("text", ""),
        })
    return chunks


def compute_cosine_similarity(query_vec, matrix):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def main():
    obligation_embeddings, obligation_metadata = load_obligations(
        ARTICLE_EMBEDDINGS_PATH, ARTICLE_METADATA_PATH
    )
    print(f"Loaded {len(obligation_metadata)} GDPR obligations.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    policy_chunks = fetch_policy_chunks_from_pinecone(index, COMPANY_NAME, EMBEDDING_DIM)
    print(f"Fetched {len(policy_chunks)} policy chunks for {COMPANY_NAME}.")

    if not policy_chunks:
        print("No policy chunks found. Check company name / index name / Pinecone config.")
        return

    # Build a matrix of all policy chunk embeddings once (reused for every obligation)
    policy_matrix = np.array([c["embedding"] for c in policy_chunks], dtype=np.float32)

    output = []
    for i, obligation_vec in enumerate(obligation_embeddings):
        similarities = compute_cosine_similarity(obligation_vec, policy_matrix)
        top_indices = np.argsort(similarities)[::-1][:TOP_K_EVIDENCE]

        evidence = [
            {
                "chunk_id": policy_chunks[idx]["chunk_id"],
                "text": policy_chunks[idx]["text"],
                "similarity": round(float(similarities[idx]), 4),
            }
            for idx in top_indices
        ]

        obligation_entry = dict(obligation_metadata[i])  # copy so we don't mutate original
        obligation_entry.pop("embedding", None)  # don't need to carry this forward
        obligation_entry["evidence"] = evidence
        output.append(obligation_entry)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. Evidence retrieved for {len(output)} obligations -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()