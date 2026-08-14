"""
Step 3: compare_policy_with_articles.py
----------------------------------------
Compares uploaded company policy chunks (stored in Pinecone) against
GDPR article embeddings (stored locally in gdpr_chunks.json) using
cosine similarity, and flags potential violations.

Input:
  - article_embeddings.npy   (from Step 1: generate_article_embeddings.py)
  - article_metadata.json    (from Step 1: generate_article_embeddings.py)
  - Pinecone index           (policy chunks uploaded in Step 2)

Output:
  - comparison_results.json  --> feeds directly into Step 4 (generate_report.py)
"""

import json
import numpy as np
from pinecone import Pinecone

# ---------- CONFIG (edit these) ----------
PINECONE_API_KEY = "pcsk_5VSGjr_9A67zH5nY6MRDFoWMuicMuBbBtcb5MgWiQ3cxTGH6DXnyz5aTtKjb6dUyg6piLy"
INDEX_NAME = "generic-rag"      # your Pinecone index name
COMPANY_NAME = "TechStartup Pvt Ltd"               # which company's policy to check
ARTICLE_EMBEDDINGS_PATH = "../sample_chunking/data/article_embeddings.npy"
ARTICLE_METADATA_PATH = "../sample_chunking/data/article_metadata.json"
EMBEDDING_DIM = 1024                    # must match your embedding model (e.g. all-MiniLM-L6-v2)
TOP_K = 3
SIMILARITY_THRESHOLD = 0.65            # tune this based on testing
# ------------------------------------------


def load_gdpr_articles(embeddings_path, metadata_path):
    """
    Loads the pre-computed article embeddings (fast, binary .npy)
    and their matching metadata (id, title) from a separate .json file.

    Order matters: row i in the .npy array must correspond to
    metadata[i] in the .json file. This is guaranteed as long as
    generate_article_embeddings.py wrote them together in the same loop.
    """
    embeddings = np.load(embeddings_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Mismatch: {embeddings.shape[0]} embeddings but {len(metadata)} "
            f"metadata entries. Re-run generate_article_embeddings.py."
        )

    return embeddings, metadata


def fetch_policy_chunks_from_pinecone(index, company_name, dim):
    """
    Fetches all policy chunks for a given company from Pinecone.

    Pinecone doesn't have a plain "give me everything matching this filter"
    call for pod-based indexes, so the standard workaround is:
    query with a dummy (zero) vector, top_k set high, and rely entirely
    on the metadata filter.
    """
    dummy_vector = [0.0] * dim

    results = index.query(
        vector=dummy_vector,
        top_k=1000,  # high enough to catch all chunks for one company
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
    """
    query_vec: shape (dim,)
    matrix: shape (n_articles, dim)
    Returns: array of shape (n_articles,) with similarity scores
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def get_top_k_matches(similarities, metadata, k):
    top_indices = np.argsort(similarities)[::-1][:k]
    return [
        {
            "article_id": metadata[i]["id"],
            "title": metadata[i]["title"],
            "similarity_score": round(float(similarities[i]), 4),
        }
        for i in top_indices
    ]


def main():
    # 1. Load GDPR article embeddings (local, fast .npy load)
    article_embeddings, article_metadata = load_gdpr_articles(
        ARTICLE_EMBEDDINGS_PATH, ARTICLE_METADATA_PATH
    )
    print(f"Loaded {len(article_metadata)} GDPR articles.")

    # 2. Connect to Pinecone and fetch policy chunks for this company
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    policy_chunks = fetch_policy_chunks_from_pinecone(index, COMPANY_NAME, EMBEDDING_DIM)
    print(f"Fetched {len(policy_chunks)} policy chunks for {COMPANY_NAME}.")

    if not policy_chunks:
        print("No policy chunks found. Check company name / index name.")
        return

    # 3. Compare each policy chunk against all GDPR articles
    comparison_results = []

    for chunk in policy_chunks:
        similarities = compute_cosine_similarity(chunk["embedding"], article_embeddings)
        top_matches = get_top_k_matches(similarities, article_metadata, TOP_K)

        best_score = top_matches[0]["similarity_score"]
        flag = "potential_violation" if best_score >= SIMILARITY_THRESHOLD else "ok"

        comparison_results.append({
            "policy_chunk": chunk["text"],
            "matched_articles": top_matches,
            "flag": flag,
        })

    # 4. Save results -> this file feeds directly into generate_report.py
    with open("../sample_chunking/comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(comparison_results)} chunks compared -> comparison_results.json saved.")
    flagged = sum(1 for r in comparison_results if r["flag"] == "potential_violation")
    print(f"{flagged} chunk(s) flagged as potential_violation.")


if __name__ == "__main__":
    main()