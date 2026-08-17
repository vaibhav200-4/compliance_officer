from collections import Counter, defaultdict


# =========================================================
# CONFIGURATION
# =========================================================

VERDICT_SCORE = {
    "FULLY_MET": 1.0,
    "PARTIALLY_MET": 0.5,
    "NOT_MET": 0.0,
    "CONFLICTING": 0.0
}

CHAPTERS = range(1, 12)


# =========================================================
# STATUS
# =========================================================

def get_status(score):
    if score is None:
        return "Not Assessed"

    if score >= 80:
        return "Compliant"

    if score >= 60:
        return "Needs Improvement"

    return "High Risk"


# =========================================================
# VALIDATION
# =========================================================

def validate_judge_result(result):
    required_fields = [
        "article_number",
        "article_name",
        "chapter_number",
        "obligation_id",
        "severity",
        "verdict",
        "confidence",
        "reason",
        "gap",
        "evidence"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(
                f"Missing field '{field}' "
                f"in obligation {result.get('obligation_id')}"
            )

    chapter = result["chapter_number"]

    if chapter not in CHAPTERS:
        raise ValueError(
            f"Invalid chapter number: {chapter}"
        )

    verdict = result["verdict"]

    if verdict not in VERDICT_SCORE and verdict != "NOT_APPLICABLE":
        raise ValueError(
            f"Invalid verdict '{verdict}' "
            f"for obligation {result['obligation_id']}"
        )


# =========================================================
# REQUIREMENT TRANSFORMATION
# =========================================================

def transform_requirement(result):
    return {
        "sub_id": result["obligation_id"],
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "evidence": result["evidence"],
        "gap": result["gap"],
        "fix_required": result.get("fix_required")
    }


# =========================================================
# AGGREGATE ONE CHAPTER
# =========================================================

def aggregate_chapter(chapter_number, chapter_results):

    # -----------------------------------------------------
    # Split out NOT_APPLICABLE -- shown in report, never scored
    # -----------------------------------------------------
    applicable_results = [
        r for r in chapter_results if r["verdict"] != "NOT_APPLICABLE"
    ]
    not_applicable_results = [
        r for r in chapter_results if r["verdict"] == "NOT_APPLICABLE"
    ]

    total = len(applicable_results)

    # -----------------------------------------------------
    # No SCORABLE obligations for this chapter
    # (either truly empty, or everything was not_applicable)
    # -----------------------------------------------------

    if total == 0:

        return {
            "overall_score": None,

            "summary": {
                "total": 0,
                "fully_met": 0,
                "partially_met": 0,
                "not_met": 0,
                "conflicting": 0,
                "not_applicable": len(not_applicable_results),
            },

            "chapter_scores": [
                {
                    "chapter": chapter_number,
                    "score": None,
                    "status": "Not Assessed",
                }
            ],

            "requirements": [
                transform_requirement(r) for r in not_applicable_results
            ],
        }

    # -----------------------------------------------------
    # Count verdicts (scorable only)
    # -----------------------------------------------------

    verdict_counts = Counter(
        result["verdict"]
        for result in applicable_results
    )

    fully_met = verdict_counts.get("FULLY_MET", 0)
    partially_met = verdict_counts.get("PARTIALLY_MET", 0)
    not_met = verdict_counts.get("NOT_MET", 0)
    conflicting = verdict_counts.get("CONFLICTING", 0)

    # -----------------------------------------------------
    # Calculate score (denominator excludes not_applicable)
    # -----------------------------------------------------

    total_points = sum(
        VERDICT_SCORE[result["verdict"]]
        for result in applicable_results
    )

    score = round(
        (total_points / total) * 100,
        2
    )

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    status = get_status(score)

    # -----------------------------------------------------
    # Requirement-level output -- includes not_applicable too,
    # so the report can still list them (for audit transparency)
    # -----------------------------------------------------

    requirements = [
        transform_requirement(result)
        for result in chapter_results
    ]

    # -----------------------------------------------------
    # Final chapter object
    # -----------------------------------------------------

    return {
        "overall_score": score,

        "summary": {
            "total": total,
            "fully_met": fully_met,
            "partially_met": partially_met,
            "not_met": not_met,
            "conflicting": conflicting,
            "not_applicable": len(not_applicable_results),
        },

        "chapter_scores": [
            {
                "chapter": chapter_number,
                "score": score,
                "status": status,
            }
        ],

        "requirements": requirements,
    }
# =========================================================
# AGGREGATE ALL 11 CHAPTERS
# =========================================================

def aggregate_results(judge_results):

    # -----------------------------------------------------
    # Validate all Judge outputs
    # -----------------------------------------------------

    for result in judge_results:
        validate_judge_result(result)

    # -----------------------------------------------------
    # Group Judge results by chapter
    # -----------------------------------------------------

    chapter_groups = defaultdict(list)

    for result in judge_results:
        chapter_number = result["chapter_number"]

        chapter_groups[chapter_number].append(result)

    # -----------------------------------------------------
    # Generate one object for every GDPR chapter
    # -----------------------------------------------------

    aggregated_results = []

    for chapter_number in CHAPTERS:

        chapter_results = chapter_groups.get(
            chapter_number,
            []
        )

        chapter_output = aggregate_chapter(
            chapter_number,
            chapter_results
        )

        aggregated_results.append(chapter_output)

    return aggregated_results