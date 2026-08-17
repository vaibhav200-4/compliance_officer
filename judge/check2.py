"""
diagnose_obligation_text.py
-----------------------------
Checks two things that would explain "everything matches the same chunk":
1. Is the 'text' field used to build article_embeddings1.npy actually the
   real GDPR legal text, or a generic placeholder repeated for every article?
2. Are obligation embeddings suspiciously similar TO EACH OTHER (not just
   to policy chunks)? If two totally different articles (e.g. Art 5 vs
   Art 33) have >0.9 cosine similarity with each other, their source text
   was near-identical -> confirms the placeholder-text hypothesis.
"""
import json
import numpy as np

ARTICLE_EMBEDDINGS_PATH = "../dataa/article_embeddings1.npy"
ARTICLE_METADATA_PATH = "../dataa/article_metadata1.json"

with open(ARTICLE_METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

embeddings = np.load(ARTICLE_EMBEDDINGS_PATH)
print(f"Loaded {len(metadata)} obligations, embeddings shape {embeddings.shape}\n")

# ---- Check 1: inspect the actual text field for a few random, DIFFERENT articles ----
sample_idxs = [0, len(metadata)//4, len(metadata)//2, 3*len(metadata)//4, len(metadata)-1]
print("=== Sample 'text' fields (should differ meaningfully across articles) ===")
for i in sample_idxs:
    m = metadata[i]
    txt = m.get("text", "")
    print(f"[{i}] id={m.get('id')}  text[:120]={txt[:120]!r}")
print()

# ---- Check 2: pairwise cosine similarity between DIFFERENT articles' embeddings ----
def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

print("=== Pairwise similarity between unrelated obligations (should be LOW, <0.5 ideally) ===")
pairs_checked = 0
sims = []
for i in sample_idxs:
    for j in sample_idxs:
        if i >= j:
            continue
        s = cos(embeddings[i], embeddings[j])
        sims.append(s)
        print(f"  {metadata[i].get('id')}  vs  {metadata[j].get('id')}  ->  similarity={s:.4f}")
        pairs_checked += 1

sims = np.array(sims)
print(f"\nMean pairwise similarity across {pairs_checked} unrelated-article pairs: {sims.mean():.4f}")
if sims.mean() > 0.85:
    print(">>> HIGH mean similarity between UNRELATED articles. Strong sign the embedded")
    print(">>> 'text' field is generic/placeholder rather than real distinct legal text.")
else:
    print(">>> Similarity looks reasonably distinct -- placeholder-text theory is NOT confirmed,")
    print(">>> look elsewhere (e.g. chunking/upsert bug on the policy side).")