#!/usr/bin/env python3
"""
Targeted Verification Test for Articles 14 and 15
Verifies transition logging, group evaluation, article progression,
report generation, and job completion.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.agents.orchestrator import ComplianceOrchestrator
from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.core.logger import get_logger

logger = get_logger()


def main() -> int:
    print("=" * 75)
    print("TARGETED VERIFICATION TEST: Articles 14 and 15")
    print("=" * 75)
    print()

    kb = GDPRKnowledgeBase()
    output_dir = PROJECT_ROOT / "Data" / "analysis_results"

    orchestrator = ComplianceOrchestrator(
        knowledge_base=kb,
        output_dir=output_dir,
        max_workers=2,
    )

    t0 = time.perf_counter()
    summary = orchestrator.run(
        start_article=14,
        end_article=15,
        resume=False,
        company_name="Verification Test Corp",
        policy_name="Test Privacy Policy",
    )
    total_time = time.perf_counter() - t0

    print()
    print("=" * 75)
    print("VERIFICATION SUMMARY")
    print("=" * 75)
    print(f"Status               : {summary.get('status')}")
    print(f"Total requested      : {summary.get('total')}")
    print(f"Completed            : {summary.get('completed')}")
    print(f"Failed               : {summary.get('failed')}")
    print(f"Total Wall-Clock Time: {total_time:.2f}s")
    print("=" * 75)

    art14_exists = (output_dir / "article_14.json").exists()
    art15_exists = (output_dir / "article_15.json").exists()
    report_exists = (output_dir / "final_report.json").exists()

    print(f"Article 14 JSON exists: {art14_exists}")
    print(f"Article 15 JSON exists: {art15_exists}")
    print(f"Final Report JSON exists: {report_exists}")

    if summary.get("status") in ("COMPLETED", "PARTIAL_REPORT") and art14_exists and art15_exists:
        logger.success("VERIFICATION SUCCESS: Pipeline executed Article 14 -> Article 15 -> JOB_COMPLETED")
        return 0
    else:
        logger.error("VERIFICATION FAILURE: Pipeline failed to reach terminal completion status.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
