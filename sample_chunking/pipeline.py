"""
pipeline.py
-------------
End-to-end orchestrator: PDF -> clean text -> section-wise chunks -> embeddings -> Pinecone.

Usage:
    python pipeline.py --pdf data/SampleInput_CompanyPolicy.pdf --company "TechStartup Pvt Ltd"

This just calls the 4 steps in sequence and reuses their functions directly,
so you don't have to run 4 separate commands every time. Intermediate files
(clean text, chunks, embeddings) are still saved to data/ for debugging.
"""

import argparse
from pathlib import Path

from loading.extract_text import extract_raw_text, clean_text
from chunking.policy_chunking import build_chunks
from embedding.embeddings import generate_embeddings
from pinecone.upload import get_or_create_index, upsert_chunks, PINECONE_API_KEY, INDEX_NAME
from pinecone import Pinecone


def run_pipeline(pdf_path: str, company: str, data_dir: str = "data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    # Step 1: extract + clean
    print("\n[1/4] Extracting text from PDF...")
    raw = extract_raw_text(pdf_path)
    cleaned = clean_text(raw)
    (data_dir / "policy_clean.txt").write_text(cleaned, encoding="utf-8")
    print(f"  -> {len(cleaned)} cleaned characters")

    # Step 2: section-wise chunking
    print("\n[2/4] Chunking section-wise...")
    chunks = build_chunks(cleaned, company)
    print(f"  -> {len(chunks)} chunks created")

    # Step 3: embeddings
    print("\n[3/4] Generating embeddings...")
    chunks, dim = generate_embeddings(chunks)
    print(f"  -> embedding dimension: {dim}")

    # Step 4: upload to Pinecone
    print("\n[4/4] Uploading to Pinecone...")
    if not PINECONE_API_KEY:
        print("  SKIPPED: PINECONE_API_KEY not set in .env. "
              "Chunks + embeddings are saved locally; run upload_to_pinecone.py "
              "separately once your key is set.")
    else:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = get_or_create_index(pc, INDEX_NAME, dim)
        upsert_chunks(index, chunks)
        print("  -> upload complete")

    out_path = data_dir / "chunks_embedded.json"
    import json
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll intermediate files saved in: {data_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full policy RAG ingestion pipeline")
    parser.add_argument("--pdf", required=True, help="Path to policy PDF")
    parser.add_argument("--company", required=True, help="Company name for metadata/IDs")
    parser.add_argument("--data_dir", default="data", help="Where to save intermediate files")
    args = parser.parse_args()

    run_pipeline(args.pdf, args.company, args.data_dir)