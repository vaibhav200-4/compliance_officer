import json

with open("../dataa/evidence_for_obligations.json") as f:
    data = json.load(f)

# sort by best similarity ascending
data_with_best = [
    (max((e["similarity"] for e in o["evidence"]), default=0), o)
    for o in data
]
data_with_best.sort(key=lambda x: x[0])

print("=== 10 LOWEST scoring obligations ===")
for score, o in data_with_best[:10]:
    print(f"\n[{score:.3f}] {o['id']}")
    print(f"  Obligation: {o['text'][:120]}")
    if o["evidence"]:
        print(f"  Best match: {o['evidence'][0]['text'][:120]}")

print("\n\n=== 10 obligations right around median (0.30-0.40) ===")
mid = [x for x in data_with_best if 0.30 <= x[0] <= 0.40][:10]
for score, o in mid:
    print(f"\n[{score:.3f}] {o['id']}")
    print(f"  Obligation: {o['text'][:120]}")
    if o["evidence"]:
        print(f"  Best match: {o['evidence'][0]['text'][:120]}")