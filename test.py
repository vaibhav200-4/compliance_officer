from pathlib import Path

from app.agents.orchestrator import ComplianceOrchestrator
from app.compliance.gdpr_kb import GDPRKnowledgeBase


KB_PATH = Path(
    "Data/new_json_gdpr.json"
)


def main():

    print("=" * 75)
    print("ORCHESTRATOR TEST")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load GDPR KB
    # ---------------------------------------------------------

    print("\n[1] Loading GDPR knowledge base...")

    kb = GDPRKnowledgeBase(
        KB_PATH
    )

    print(
        f"Articles    : {kb.article_count()}"
    )

    print(
        f"Groups      : {kb.group_count()}"
    )

    print(
        f"Obligations : {kb.obligation_count()}"
    )

    # ---------------------------------------------------------
    # 2. Initialize orchestrator
    # ---------------------------------------------------------

    print(
        "\n[2] Initializing orchestrator..."
    )

    orchestrator = ComplianceOrchestrator(
        knowledge_base=kb,

        # Start conservatively.
        max_workers=3,

        # NEW: bounded group-level concurrency inside each
        # article. 3 articles x 2 groups = 6 concurrent LLM
        # calls at peak. Drop to 1 if you see 429 rate-limit
        # errors from OpenRouter on the free-tier model.
        max_group_workers=2,

        top_k=5,

        min_score=None,

        output_dir=Path(
            "Data/analysis_results"
        ),
    )

    print(
        "Orchestrator initialized."
    )

    # ---------------------------------------------------------
    # 3. TEST ONLY ARTICLES 1-5
    # ---------------------------------------------------------

    print(
        "\n[3] Running Articles 1-5..."
    )

    summary = orchestrator.run(
        start_article=1,
        end_article=5,
        resume=True,
    )

    # ---------------------------------------------------------
    # 4. Summary
    # ---------------------------------------------------------

    print(
        "\n[4] FINAL SUMMARY"
    )

    print(
        f"Status   : "
        f"{summary['status']}"
    )

    print(
        f"Total    : "
        f"{summary['total']}"
    )

    print(
        f"Success  : "
        f"{summary['completed']}"
    )

    print(
        f"Failed   : "
        f"{summary['failed']}"
    )

    print(
        "\nORCHESTRATOR TEST COMPLETED."
    )


if __name__ == "__main__":
    main()