from __future__ import annotations

import json
from pathlib import Path

from app.agents.analyzer_agent import AnalyzerAgent
from app.compliance.gdpr_kb import GDPRKnowledgeBase


# ============================================================
# CONFIG
# ============================================================

KB_PATH = Path(
    "Data/new_json_gdpr.json"
)

ARTICLE_NUMBER = 5

TOP_K = 5

MIN_SCORE = None

OUTPUT_PATH = Path(
    "Data/article_5_analysis.json"
)


# ============================================================
# HELPERS
# ============================================================

def print_separator(
    title: str = "",
):
    print(
        "\n" + "=" * 75
    )

    if title:
        print(title)

    print(
        "=" * 75
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator(
        "ARTICLE 5 END-TO-END ANALYZER TEST"
    )

    # ========================================================
    # 1. LOAD GDPR KNOWLEDGE BASE
    # ========================================================

    print(
        "\n[1] Loading GDPR knowledge base..."
    )

    kb = GDPRKnowledgeBase(
        KB_PATH
    )

    print(
        f"Articles     : "
        f"{kb.article_count()}"
    )

    print(
        f"Groups       : "
        f"{kb.group_count()}"
    )

    print(
        f"Obligations  : "
        f"{kb.obligation_count()}"
    )

    # ========================================================
    # 2. LOAD ARTICLE
    # ========================================================

    print(
        f"\n[2] Loading Article "
        f"{ARTICLE_NUMBER}..."
    )

    article = kb.get_article(
        ARTICLE_NUMBER
    )

    print(
        f"Article      : "
        f"{article.article_number}"
    )

    print(
        f"Title        : "
        f"{article.article_name}"
    )

    print(
        f"Checkability : "
        f"{article.checkability}"
    )

    groups = kb.get_groups(
        ARTICLE_NUMBER
    )

    print(
        f"Groups       : "
        f"{len(groups)}"
    )

    for group in groups:

        print(
            f"  - "
            f"{group.group_id} | "
            f"{group.principle} | "
            f"{group.condition_logic} | "
            f"{len(group.obligations)} obligations"
        )

    # ========================================================
    # 3. INITIALIZE ANALYZER
    # ========================================================

    print(
        "\n[3] Initializing AnalyzerAgent..."
    )

    analyzer = AnalyzerAgent(
        knowledge_base=kb,
        top_k=TOP_K,
        min_score=MIN_SCORE,
    )

    print(
        "AnalyzerAgent initialized."
    )

    print(
        f"Top-K       : {TOP_K}"
    )

    print(
        f"Min score   : {MIN_SCORE}"
    )

    # ========================================================
    # 4. ANALYZE ARTICLE
    # ========================================================

    print_separator(
        f"ANALYZING ARTICLE {ARTICLE_NUMBER}"
    )

    result = analyzer.analyze_article(
        ARTICLE_NUMBER
    )

    # ========================================================
    # 5. PRINT ARTICLE SUMMARY
    # ========================================================

    print_separator(
        "ARTICLE RESULT"
    )

    print(
        f"Article          : "
        f"{result['article_number']}"
    )

    print(
        f"Title            : "
        f"{result['article_title']}"
    )

    print(
        f"Status           : "
        f"{result['status']}"
    )

    print(
        f"Confidence       : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Groups expected  : "
        f"{result['group_count']}"
    )

    print(
        f"Groups completed : "
        f"{result['completed_groups']}"
    )

    # ========================================================
    # 6. PRINT GROUP RESULTS
    # ========================================================

    print_separator(
        "GROUP RESULTS"
    )

    for group in result["groups"]:

        print(
            f"\nGroup: "
            f"{group['group_id']}"
        )

        print(
            f"Principle  : "
            f"{group['principle']}"
        )

        print(
            f"Logic      : "
            f"{group['condition_logic']}"
        )

        print(
            f"Status     : "
            f"{group['status']}"
        )

        print(
            f"Confidence : "
            f"{group['confidence']:.4f}"
        )

        print(
            f"Evidence   : "
            f"{group['evidence_count']} chunks"
        )

        print(
            f"Reason     : "
            f"{group['reason']}"
        )

        if group.get("gap"):

            print(
                f"Gap        : "
                f"{group['gap']}"
            )

        print(
            "Sub-obligations:"
        )

        for obligation in group[
            "sub_obligations"
        ]:

            print(
                f"  {obligation.get('obligation_id')}"
            )

            print(
                f"    Status     : "
                f"{obligation.get('status')}"
            )

            print(
                f"    Confidence : "
                f"{obligation.get('confidence')}"
            )

            print(
                f"    Reason     : "
                f"{obligation.get('reason')}"
            )

            evidence = (
                obligation.get(
                    "evidence",
                    [],
                )
            )

            for reference in evidence:

                print(
                    f"    Evidence   : "
                    f"{reference.get('chunk_id')}"
                )

                print(
                    f"    Quote      : "
                    f"{reference.get('quote')}"
                )

    # ========================================================
    # 7. VALIDATION
    # ========================================================

    print_separator(
        "VALIDATION"
    )

    expected_groups = len(groups)

    actual_groups = len(
        result["groups"]
    )

    if actual_groups != expected_groups:

        raise RuntimeError(
            f"Expected {expected_groups} "
            f"groups but received "
            f"{actual_groups}."
        )

    print(
        f"Group count check : PASS "
        f"({actual_groups}/{expected_groups})"
    )

    if result["completed_groups"] != expected_groups:

        raise RuntimeError(
            "Not all groups completed."
        )

    print(
        "All groups completed : PASS"
    )

    valid_statuses = {
        "MET",
        "PARTIALLY_MET",
        "NOT_MET",
        "CONFLICTING",
        "INSUFFICIENT_EVIDENCE",
        "NOT_APPLICABLE",
    }

    for group in result["groups"]:

        if group["status"] not in valid_statuses:

            raise RuntimeError(
                f"Invalid group status: "
                f"{group['status']}"
            )

        confidence = float(
            group["confidence"]
        )

        if not 0.0 <= confidence <= 1.0:

            raise RuntimeError(
                f"Invalid confidence for "
                f"{group['group_id']}: "
                f"{confidence}"
            )

    print(
        "Group status validation : PASS"
    )

    print(
        "Confidence validation    : PASS"
    )

    # ========================================================
    # 8. SAVE RESULT
    # ========================================================

    print(
        "\n[8] Saving Article 5 result..."
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved to:"
        f"\n{OUTPUT_PATH}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print_separator(
        "ARTICLE 5 ANALYZER TEST COMPLETED"
    )

    print(
        f"Final Article Status : "
        f"{result['status']}"
    )

    print(
        f"Final Confidence     : "
        f"{result['confidence']:.4f}"
    )


if __name__ == "__main__":
    main()