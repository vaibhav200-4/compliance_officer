import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # folder this script lives in


with open("sample_report/judge/judge_input.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print("Total judged:", len(results))
from collections import Counter
print("Verdict breakdown:", Counter(r["verdict"] for r in results))
print("Chapters covered:", sorted(set(r["chapter_number"] for r in results)))