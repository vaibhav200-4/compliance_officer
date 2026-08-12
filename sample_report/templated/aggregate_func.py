"""
Chapter Aggregator Layer
------------------------
Takes obligation-level assessment records (the output of the compliance-checking
layer) and rolls them up into per-chapter score reports.

Input : list[dict]  -> obligation-level records with keys:
    article_number, article_name, chapter_number, obligation_id,
    severity, verdict, confidence, reason, gap, evidence

Output: list[dict]  -> one object per chapter (chapter_range, default 1-11):
    {
      "overall_score": float | None,
      "summary": {"total", "fully_met", "partially_met", "not_met", "conflicting"},
      "chapter_scores": [{"chapter", "score", "status"}],
      "requirements": [
          {"sub_id", "verdict", "confidence", "evidence", "gap", "fix_required"}
      ]
    }
"""

from collections import defaultdict

# Weight applied to each verdict when computing the chapter completion score.
VERDICT_WEIGHTS = {
    "FULLY_MET": 1.0,
    "PARTIALLY_MET": 0.5,
    "NOT_MET": 0.0,
    "CONFLICTING": 0.0,
}

# Score thresholds -> status label. Checked top-down, first match wins.
STATUS_BANDS = [
    (80.0, "Good"),
    (60.0, "Needs Improvement"),
    (0.0, "High Risk"),
]


def _status_for_score(score: float) -> str:
    for threshold, label in STATUS_BANDS:
        if score >= threshold:
            return label
    return "High Risk"


def aggregate_chapters(obligations: list[dict], chapter_range=range(1, 12)) -> list[dict]:
    """
    Group obligation-level records by chapter_number and compute a
    weighted completion score + status for each chapter.

    chapter_range: which chapter numbers to emit reports for, in order.
                   Chapters with no matching obligations get a null score
                   and "Not Assessed" status (so the report shape stays
                   consistent even for un-assessed chapters).
    """
    grouped = defaultdict(list)
    for record in obligations:
        grouped[record["chapter_number"]].append(record)

    reports = []

    for chapter in chapter_range:
        items = grouped.get(chapter, [])
        total = len(items)

        counts = {"fully_met": 0, "partially_met": 0, "not_met": 0, "conflicting": 0}
        weighted_sum = 0.0
        requirements = []

        for record in items:
            verdict = record["verdict"]
            count_key = verdict.lower()
            if count_key in counts:
                counts[count_key] += 1
            weighted_sum += VERDICT_WEIGHTS.get(verdict, 0.0)

            requirements.append({
                "sub_id": record["obligation_id"],
                "verdict": verdict,
                "confidence": record.get("confidence"),
                "evidence": [],          # evidence detail intentionally dropped at this layer
                "gap": record.get("gap"),
                "fix_required": None,    # populated by the downstream fix/recommendation layer
            })

        if total == 0:
            score = None
            status = "Not Assessed"
        else:
            score = round((weighted_sum / total) * 100, 2)
            status = _status_for_score(score)

        reports.append({
            "overall_score": score,
            "summary": {
                "total": total,
                "fully_met": counts["fully_met"],
                "partially_met": counts["partially_met"],
                "not_met": counts["not_met"],
                "conflicting": counts["conflicting"],
            },
            "chapter_scores": [{
                "chapter": chapter,
                "score": score,
                "status": status,
            }],
            "requirements": requirements,
        })

    return reports


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        obligations = json.load(f)

    result = aggregate_chapters(obligations)
    print(json.dumps(result, indent=2))