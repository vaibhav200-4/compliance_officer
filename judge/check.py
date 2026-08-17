"""
Quick diagnostic: paste this in your report/ dir and run.
Checks: (1) dims match, (2) norms are sane, (3) whether obligation<->policy
similarity scores are near-random (mean/std close to 0, ~uniform) which
confirms an embedding-model mismatch.
"""
import numpy as np, json

obligation_embeddings = np.load("../dataa/article_embeddings1.npy")
with open("dataa/evidence_for_obligations.json", encoding="utf-8") as f:
    evidence = json.load(f)

print("Obligation embeddings shape:", obligation_embeddings.shape)
print("Obligation vector norms (first 5):", np.linalg.norm(obligation_embeddings[:5], axis=1))

all_sims = [e["similarity"] for ob in evidence for e in ob["evidence"]]
all_sims = np.array(all_sims)
print(f"\nSimilarity stats across {len(all_sims)} evidence pairs:")
print("  mean:", all_sims.mean(), "  std:", all_sims.std())
print("  min:", all_sims.min(), "  max:", all_sims.max())

# how many DISTINCT top-1 chunk_ids are being reused across obligations?
top1_chunks = [ob["evidence"][0]["chunk_id"] for ob in evidence if ob["evidence"]]
from collections import Counter
c = Counter(top1_chunks)
print(f"\nDistinct top-1 chunks used: {len(c)} out of {len(top1_chunks)} obligations")
print("Most reused chunks:", c.most_common(5))