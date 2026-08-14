"""
Step 1: generate_article_embeddings.py
----------------------------------------
Produces the two files the comparison engine (Step 3) actually needs:
  - article_embeddings.npy   -> numpy array, shape (n_articles, dim)
  - article_metadata.json    -> [{"id": ..., "title": ...}, ...]

This script works in TWO MODES:

MODE A (fast path - USE THIS if you already have gdpr_chunks.json):
  Just splits your existing gdpr_chunks.json into the .npy + .json pair.
  No re-embedding needed.

MODE B (fresh path - use only if starting from raw articles.json):
  Embeds articles.json from scratch using sentence-transformers.

Run this once, or whenever your GDPR articles change.
"""

import json
import numpy as np

# ---------- CONFIG ----------
MODE = "A"  # "A" = convert existing gdpr_chunks.json, "B" = embed articles.json fresh
GDPR_CHUNKS_PATH = "data/gdpr_chunks_embedded.json"       # used in MODE A
ARTICLES_JSON_PATH = "data/gdpr_articles.json"        # used in MODE B
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"   # used in MODE B (must match policy embedding model)
OUTPUT_EMBEDDINGS_PATH = "data/article_embeddings1.npy"
OUTPUT_METADATA_PATH = "data/article_metadata1.json"
# -----------------------------


def mode_a_convert_existing(path):
    """
    Splits an already-embedded gdpr_chunks.json into a separate .npy array
    and metadata.json.

    IMPORTANT: preserves the FULL metadata block (chapter_number, article_number,
    article_name, section_name, text, etc.) -- not just id/title. The judge step
    later needs chapter_number/article_number/article_name/severity to build
    valid judge_results entries, so losing this metadata breaks the pipeline
    downstream even though embeddings/similarity still "work".

    Expected input shape per item:
    {
      "id": "gdpr_article_1_Article_1_1_",
      "text": "...",
      "embedding": [...],
      "metadata": {
        "article_number": 1,
        "article_name": "Subject-matter and objectives",
        "chapter_number": 1,
        "section_name": "Article 1(1)",
        ...
      }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    metadata = []

    for item in data:
        emb = item.get("embedding") or item.get("vector") or item.get("values")
        if emb is None:
            raise ValueError(f"No embedding found for article {item.get('id')}. "
                              f"Check the key name in gdpr_chunks.json.")
        embeddings.append(emb)

        item_meta = item.get("metadata", {})
        metadata.append({
            "id": item.get("id"),
            "text": item.get("text") or item.get("content"),
            "article_number": item_meta.get("article_number"),
            "article_name": item_meta.get("article_name"),
            "chapter_number": item_meta.get("chapter_number"),
            "section_name": item_meta.get("section_name"),
            # severity isn't in the source data -- default placeholder for now,
            # can be manually refined later per-article if needed.
            "severity": item_meta.get("severity", "MEDIUM"),
        })

    embeddings_array = np.array(embeddings, dtype=np.float32)
    return embeddings_array, metadata


def mode_b_generate_fresh(path, model_name):
    """
    Embeds articles.json from scratch. Only needed if you don't already
    have embeddings, or if you're re-generating after changing the model.
    """
    from sentence_transformers import SentenceTransformer

    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    model = SentenceTransformer(model_name)

    texts = [a["content"] for a in articles]
    embeddings_array = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    metadata = [{"id": a["id"], "title": a["title"]} for a in articles]

    return embeddings_array.astype(np.float32), metadata


def main():
    if MODE == "A":
        print("Mode A: converting existing gdpr_chunks.json ...")
        embeddings_array, metadata = mode_a_convert_existing(GDPR_CHUNKS_PATH)
    elif MODE == "B":
        print("Mode B: embedding articles.json from scratch ...")
        embeddings_array, metadata = mode_b_generate_fresh(ARTICLES_JSON_PATH, EMBEDDING_MODEL_NAME)
    else:
        raise ValueError("MODE must be 'A' or 'B'")

    np.save(OUTPUT_EMBEDDINGS_PATH, embeddings_array)
    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved {embeddings_array.shape[0]} article embeddings, dim={embeddings_array.shape[1]}")
    print(f"  -> {OUTPUT_EMBEDDINGS_PATH}")
    print(f"  -> {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()