#!/usr/bin/env python3
"""
Complete GDPR Compliance Analysis Runner

Orchestrates the full GDPR compliance analysis across all 99 articles.
Includes smoke test (articles 1-2) before proceeding to full run.

Usage:
    python run_compliance_analysis.py [--smoke-only]
    python run_compliance_analysis.py [--skip-smoke]
    python run_compliance_analysis.py [--start 5] [--end 10]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure the project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.orchestrator import ComplianceOrchestrator
from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.core.logger import get_logger


logger = get_logger()

# Configuration
OUTPUT_DIR = PROJECT_ROOT / "Data" / "analysis_results"
DEFAULT_ARTICLE_WORKERS = 2
DEFAULT_GROUP_WORKERS = 2
DEFAULT_TOP_K = 5
SMOKE_TEST_START = 1
SMOKE_TEST_END = 2


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    logger.info("Environment variables loaded.")


def initialize_knowledge_base() -> GDPRKnowledgeBase:
    """
    Initialize the GDPR Knowledge Base.
    
    Returns:
        GDPRKnowledgeBase instance
    """
    logger.info("Initializing GDPR Knowledge Base...")
    kb = GDPRKnowledgeBase()
    logger.success(f"Knowledge Base loaded: {kb.article_count()} articles")
    return kb


def run_smoke_test(
    orchestrator: ComplianceOrchestrator,
) -> bool:
    """
    Run smoke test on articles 1-2.
    
    Parameters:
        orchestrator: ComplianceOrchestrator instance
    
    Returns:
        True if smoke test passed, False otherwise
    """
    print()
    print("=" * 75)
    print("SMOKE TEST: Running Articles 1-2")
    print("=" * 75)
    print()
    
    start_time = time.time()
    
    try:
        result = orchestrator.run(
            start_article=SMOKE_TEST_START,
            end_article=SMOKE_TEST_END,
            resume=False,
        )
        
        elapsed = time.time() - start_time
        
        print()
        print("=" * 75)
        print("SMOKE TEST RESULT")
        print("=" * 75)
        print(f"Status: {result.get('status')}")
        print(f"Completed: {result.get('completed')}/{result.get('total')}")
        print(f"Failed: {result.get('failed')}")
        print(f"Time: {elapsed:.1f}s")
        print("=" * 75)
        print()
        
        # Check if smoke test passed
        if result.get("failed", 0) > 0:
            logger.error(f"Smoke test failed with {result.get('failed')} article(s)")
            return False
        
        # Verify output files exist
        for article_num in range(SMOKE_TEST_START, SMOKE_TEST_END + 1):
            result_file = OUTPUT_DIR / f"article_{article_num}.json"
            if not result_file.exists():
                logger.error(f"Expected result file not found: {result_file}")
                return False
        
        logger.success("Smoke test PASSED")
        return True
    
    except Exception as exc:
        logger.exception(f"Smoke test failed with exception: {exc}")
        return False


def run_full_analysis(
    orchestrator: ComplianceOrchestrator,
    start_article: int = 1,
    end_article: int | None = None,
) -> dict[str, Any]:
    """
    Run full compliance analysis.
    
    Parameters:
        orchestrator: ComplianceOrchestrator instance
        start_article: Starting article number
        end_article: Ending article number (None = all articles)
    
    Returns:
        Orchestration summary dictionary
    """
    print()
    print("=" * 75)
    print("FULL ANALYSIS: Running All Articles")
    print("=" * 75)
    print()
    
    start_time = time.time()
    
    result = orchestrator.run(
        start_article=start_article,
        end_article=end_article,
        resume=True,
    )
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 75)
    print("FULL ANALYSIS COMPLETE")
    print("=" * 75)
    print(f"Time: {elapsed:.1f}s")
    print("=" * 75)
    print()
    
    return result


def validate_results(output_dir: Path) -> dict[str, Any]:
    """
    Validate result files in the output directory.
    
    Parameters:
        output_dir: Directory containing result files
    
    Returns:
        Validation summary
    """
    import json
    
    logger.info("Validating results...")
    
    successful_articles = []
    failed_articles = []
    corrupt_files = []
    
    # Find all result files
    result_files = sorted(output_dir.glob("article_*.json"))
    
    for result_file in result_files:
        
        # Skip failure files
        if "_failure.json" in result_file.name:
            continue
        
        article_num_str = (
            result_file.name
            .replace("article_", "")
            .replace(".json", "")
        )
        
        try:
            article_num = int(article_num_str)
        except ValueError:
            continue
        
        try:
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate structure
            required_fields = {
                "article_number",
                "status",
                "confidence",
                "groups",
            }
            
            if not required_fields.issubset(data.keys()):
                corrupt_files.append(result_file.name)
                continue
            
            if data.get("status") == "FAILED":
                failed_articles.append(article_num)
                continue
            
            successful_articles.append(article_num)
        
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Corrupt file {result_file.name}: {exc}")
            corrupt_files.append(result_file.name)
    
    summary = {
        "successful": len(successful_articles),
        "failed": len(failed_articles),
        "corrupt": len(corrupt_files),
        "successful_articles": successful_articles,
        "failed_articles": failed_articles,
        "corrupt_files": corrupt_files,
    }
    
    logger.success(
        f"Validation complete: "
        f"{len(successful_articles)} successful, "
        f"{len(failed_articles)} failed, "
        f"{len(corrupt_files)} corrupt"
    )
    
    return summary


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    parser = argparse.ArgumentParser(
        description="GDPR Compliance Analysis Runner"
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only smoke test (articles 1-2)",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip smoke test and run full analysis",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting article number",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending article number",
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_environment()
    
    # Initialize components
    logger.info("Initializing components...")
    kb = initialize_knowledge_base()
    
    orchestrator = ComplianceOrchestrator(
        knowledge_base=kb,
        output_dir=OUTPUT_DIR,
        max_workers=DEFAULT_ARTICLE_WORKERS,
        max_group_workers=DEFAULT_GROUP_WORKERS,
        top_k=DEFAULT_TOP_K,
    )
    
    # Run smoke test unless explicitly skipped
    if not args.skip_smoke:
        smoke_result = run_smoke_test(orchestrator)
        
        if not smoke_result:
            logger.error(
                "Smoke test failed. "
                "Not proceeding to full analysis."
            )
            return 1
    
    # Exit if smoke-only mode
    if args.smoke_only:
        logger.info("Smoke-only mode: exiting.")
        return 0
    
    # Run full analysis
    logger.info("Proceeding to full analysis...")
    full_result = run_full_analysis(
        orchestrator,
        start_article=args.start,
        end_article=args.end,
    )
    
    # Validate results
    validation = validate_results(OUTPUT_DIR)
    
    # Print final diagnostic
    print_final_diagnostic(
        kb=kb,
        orchestration_result=full_result,
        validation_result=validation,
        output_dir=OUTPUT_DIR,
    )
    
    return 0 if full_result.get("failed", 0) == 0 else 1


def print_final_diagnostic(
    kb: GDPRKnowledgeBase,
    orchestration_result: dict[str, Any],
    validation_result: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Print final diagnostic report.
    
    Parameters:
        kb: Knowledge base instance
        orchestration_result: Orchestration run result
        validation_result: Validation results
        output_dir: Output directory path
    """
    print()
    print("=" * 75)
    print("GDPR COMPLIANCE ANALYSIS COMPLETE")
    print("=" * 75)
    print()
    
    total_articles = kb.article_count()
    successful = validation_result.get("successful", 0)
    failed = validation_result.get("failed", 0)
    corrupt = validation_result.get("corrupt", 0)
    
    print(f"Total articles in KB    : {total_articles}")
    print(f"Successful analyses     : {successful}")
    print(f"Failed analyses         : {failed}")
    print(f"Corrupt files           : {corrupt}")
    print()
    
    if failed > 0:
        failed_list = validation_result.get("failed_articles", [])
        if failed_list:
            print(f"Failed article numbers  : {sorted(failed_list)}")
    
    print()
    print(f"Output directory        : {output_dir}")
    print()
    print("=" * 75)
    print()


if __name__ == "__main__":
    sys.exit(main())
