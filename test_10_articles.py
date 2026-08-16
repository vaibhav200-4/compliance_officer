#!/usr/bin/env python3
"""
10-Article Representative Benchmark (Articles 5-14)
Measures global group queue performance across 10 articles.
"""

from __future__ import annotations

import json
import os
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
from app.core.config import get_settings


def main() -> int:
    print("=" * 75)
    print("10-ARTICLE BENCHMARK (Articles 5 to 14)")
    print("=" * 75)

    kb = GDPRKnowledgeBase()
    settings = get_settings()

    print(f"Max Concurrent LLM Limit : {settings.MAX_CONCURRENT_LLM_REQUESTS}")
    print(f"Article Range             : 5 to 14 (10 articles)")

    output_dir = PROJECT_ROOT / "Data" / "analysis_results"

    orchestrator = ComplianceOrchestrator(
        knowledge_base=kb,
        output_dir=output_dir,
        max_workers=settings.MAX_CONCURRENT_LLM_REQUESTS,
    )

    t0 = time.perf_counter()
    summary = orchestrator.run(
        start_article=5,
        end_article=14,
        resume=False,
    )
    total_wall_clock = time.perf_counter() - t0

    # Collect metrics across all 10 articles
    total_groups = 0
    total_obligations = 0
    total_attempts = 0
    total_retrieval_time = 0.0
    total_llm_time = 0.0
    total_429 = 0
    total_5xx = 0
    total_malformed = 0
    total_validation_failures = 0
    fallback_groups_count = 0

    articles_processed = 0
    articles_skipped = summary.get("resumed", 0)

    for art_num in range(5, 15):
        result_file = output_dir / f"article_{art_num}.json"
        if result_file.exists():
            articles_processed += 1
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            groups = data.get("groups", [])
            total_groups += len(groups)

            perf = data.get("performance", {})
            total_retrieval_time += perf.get("total_retrieval_time", 0.0)
            total_llm_time += perf.get("total_llm_time", 0.0)
            total_attempts += perf.get("total_attempts", 0)
            total_429 += perf.get("count_429", 0)
            total_5xx += perf.get("count_5xx", 0)
            total_malformed += perf.get("count_malformed_json", 0)
            total_validation_failures += perf.get("count_validation_failures", 0)

            for g in groups:
                total_obligations += len(g.get("sub_obligations", []))
                if g.get("status") == "INSUFFICIENT_EVIDENCE" and "error" in g:
                    fallback_groups_count += 1

    effective_concurrency = total_llm_time / total_wall_clock if total_wall_clock > 0 else 0.0
    avg_group_latency = total_wall_clock / total_groups if total_groups > 0 else 0.0

    print()
    print("=" * 75)
    print("10-ARTICLE BENCHMARK REPORT (Global Group Queue)")
    print("=" * 75)
    print(f"Articles requested         : 10 (Articles 5-14)")
    print(f"Articles processed         : {articles_processed}")
    print(f"Articles skipped           : {articles_skipped}")
    print(f"Requirement groups         : {total_groups}")
    print(f"Sub-obligations evaluated : {total_obligations}")
    print(f"Initial LLM calls          : {total_groups}")
    print(f"Total LLM calls            : {total_attempts}")
    print(f"Retries                    : {total_attempts - total_groups}")
    print(f"429 Count                  : {total_429}")
    print(f"Timeout / 5xx Count        : {total_5xx}")
    print(f"Malformed JSON Count       : {total_malformed}")
    print(f"Validation Failures        : {total_validation_failures}")
    print(f"Fallback Groups            : {fallback_groups_count}")
    print(f"Total Retrieval Time       : {total_retrieval_time:.2f}s")
    print(f"Total LLM Time             : {total_llm_time:.2f}s")
    print(f"Total Wall-Clock Time      : {total_wall_clock:.2f}s")
    print(f"Effective LLM Concurrency  : {effective_concurrency:.2f}x")
    print(f"Average Group Latency      : {avg_group_latency:.2f}s")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
