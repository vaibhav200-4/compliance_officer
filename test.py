"""
Article 5 end-to-end compliance test.

Tests:
    GDPR KB
        ↓
    Pinecone retrieval
        ↓
    AnalyzerAgent
        ↓
    ComplianceJudge
        ↓
    NVIDIA NIM
        ↓
    Article 5 result

IMPORTANT:
- Runs ONLY Article 5.
- Does NOT run all 99 articles.
- Uses the existing application components.
- Uses 5 article workers and 1 group worker.
- Uses NVIDIA provider through existing configuration.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# IMPORT EXISTING PROJECT COMPONENTS
# ============================================================

from app.agents.orchestrator import ComplianceOrchestrator
from app.compliance.gdpr_kb import GDPRKnowledgeBase


# ============================================================
# CONFIGURATION
# ============================================================

ARTICLE_NUMBER = 5

ARTICLE_WORKERS = int(
    os.getenv("ARTICLE_WORKERS", "5")
)

GROUP_WORKERS = int(
    os.getenv("GROUP_WORKERS", "1")
)

OUTPUT_DIR = Path(
    os.getenv(
        "ANALYSIS_OUTPUT_DIR",
        str(PROJECT_ROOT / "Data" / "analysis_results"),
    )
)


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def inspect_result(result: dict) -> None:
    """
    Print important information from the Article 5 result.
    Does not modify the result.
    """

    print_header("ARTICLE 5 RESULT")

    print(f"Article       : {result.get('article_number')}")
    print(f"Status        : {result.get('status')}")
    print(f"Confidence    : {result.get('confidence')}")

    groups = result.get("groups", [])

    print(f"Groups        : {len(groups)}")

    total_obligations = 0

    status_counts: dict[str, int] = {}

    for group in groups:

        group_id = group.get("group_id", "UNKNOWN")
        group_status = group.get("status", "UNKNOWN")

        status_counts[group_status] = (
            status_counts.get(group_status, 0) + 1
        )

        obligations = group.get(
            "sub_obligations",
            []
        )

        total_obligations += len(obligations)

        print(
            f"  Group {group_id:<15} "
            f"{group_status:<25} "
            f"obligations={len(obligations)}"
        )

    print()
    print(f"Total obligations : {total_obligations}")

    print()
    print("Group status summary:")

    for status, count in sorted(status_counts.items()):
        print(f"  {status:<25}: {count}")


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print_header(
        "GDPR ARTICLE 5 — END-TO-END NVIDIA TEST"
    )

    print("Configuration:")
    print(f"  Article workers : {ARTICLE_WORKERS}")
    print(f"  Group workers   : {GROUP_WORKERS}")
    print(f"  Article         : {ARTICLE_NUMBER}")
    print(f"  Output directory: {OUTPUT_DIR}")

    provider = os.getenv(
        "LLM_PROVIDER",
        ""
    ).strip().lower()

    model = os.getenv(
        "NVIDIA_NIM_MODEL",
        ""
    ).strip()

    timeout = os.getenv(
        "LLM_REQUEST_TIMEOUT",
        "30"
    )

    retries = os.getenv(
        "LLM_MAX_RETRIES",
        "3"
    )

    print()
    print("LLM configuration:")
    print(f"  Provider : {provider or '<not explicitly configured>'}")
    print(f"  NVIDIA model : {model or '<not configured>'}")
    print(f"  Timeout  : {timeout}s")
    print(f"  Retries  : {retries}")

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if provider and provider != "nvidia":

        print()
        print(
            "ERROR: LLM_PROVIDER is not set to NVIDIA."
        )
        print(
            f"Current provider: {provider}"
        )
        print()
        print(
            "Set this in .env before running:"
        )
        print(
            "LLM_PROVIDER=nvidia"
        )

        return 1

    if not model:

        print()
        print(
            "WARNING: NVIDIA_NIM_MODEL is not configured."
        )

        return 1

    # --------------------------------------------------------
    # Create orchestrator
    # --------------------------------------------------------

    print_header(
        "INITIALIZING COMPLIANCE ORCHESTRATOR"
    )

    try:

        # Initialize GDPR Knowledge Base
        knowledge_base = GDPRKnowledgeBase()

        orchestrator = ComplianceOrchestrator(
            knowledge_base=knowledge_base,
            max_workers=ARTICLE_WORKERS,
            max_group_workers=GROUP_WORKERS,
            output_dir=str(OUTPUT_DIR),
            top_k=5,
        )

    except Exception as exc:

        print()
        print(
            "ORCHESTRATOR INITIALIZATION FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    print(
        "Orchestrator initialized successfully."
    )

    # --------------------------------------------------------
    # Run Article 5 only
    # --------------------------------------------------------

    print_header(
        "RUNNING ARTICLE 5"
    )

    print(
        "Only Article 5 will be processed."
    )

    print(
        "No other GDPR articles will be executed."
    )

    print()

    start_time = time.perf_counter()

    try:

        results = orchestrator.run(
            start_article=ARTICLE_NUMBER,
            end_article=ARTICLE_NUMBER,
            resume=False,
        )

    except Exception as exc:

        elapsed = time.perf_counter() - start_time

        print()
        print(
            "ARTICLE 5 EXECUTION FAILED"
        )

        print(
            f"Elapsed time: {elapsed:.2f} seconds"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    elapsed = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Check returned result
    # --------------------------------------------------------

    if not results:

        print()
        print(
            "ERROR: Orchestrator returned no results."
        )

        return 1

    result = results.get(ARTICLE_NUMBER)

    if result is None:

        # Some implementations may use string keys.
        result = results.get(
            str(ARTICLE_NUMBER)
        )

    if result is None:

        print()
        print(
            "ERROR: Article 5 result was not returned."
        )

        print(
            f"Returned result keys: {list(results.keys())}"
        )

        return 1

    # --------------------------------------------------------
    # Save/inspect result
    # --------------------------------------------------------

    inspect_result(result)

    output_file = (
        OUTPUT_DIR
        / f"article_{ARTICLE_NUMBER}.json"
    )

    print()
    print(
        f"Expected output: {output_file}"
    )

    if output_file.exists():

        print(
            "Output file      : EXISTS"
        )

        try:

            with output_file.open(
                "r",
                encoding="utf-8",
            ) as file:

                saved_result = json.load(file)

            print(
                "JSON validation  : VALID"
            )

            if saved_result.get(
                "article_number"
            ) != ARTICLE_NUMBER:

                print(
                    "WARNING: article_number in saved "
                    "JSON does not match Article 5."
                )

        except Exception as exc:

            print(
                "JSON validation  : FAILED"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            return 1

    else:

        print(
            "WARNING: Expected Article 5 JSON "
            "file was not found."
        )

    # --------------------------------------------------------
    # Final diagnostic
    # --------------------------------------------------------

    print_header(
        "ARTICLE 5 TEST COMPLETE"
    )

    print(
        f"Execution time : {elapsed:.2f} seconds"
    )

    print(
        f"Article status : {result.get('status')}"
    )

    print(
        f"Output file    : {output_file}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This test processed ONLY Article 5."
    )
    print(
        "The full 99-article analysis was NOT started."
    )

    print()
    print(
        "ARTICLE 5 NVIDIA TEST: PASSED"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())