# app/agents/orchestrator.py

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agents.analyzer_agent import AnalyzerAgent
from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.core.config import get_settings
from app.core.logger import get_logger


logger = get_logger()


@dataclass(frozen=True)
class GroupTask:
    """Represents a single ComplianceGroup task in the global queue."""
    article_number: int
    group: Any
    index: int
    total_groups: int


class ComplianceOrchestrator:
    """
    Orchestrates GDPR article-level compliance analysis using a Global Group-Task Queue.

    Responsibilities
    ----------------
    - Filter completed articles for resume capability.
    - Instantly process 0-group articles.
    - Enqueue all pending requirement groups across articles into a global task queue.
    - Process group tasks using a worker pool matching MAX_CONCURRENT_LLM_REQUESTS.
    - Aggregate article verdicts deterministically when all groups for an article finish.
    - Save article results atomically.
    """

    def __init__(
        self,
        knowledge_base: GDPRKnowledgeBase,
        *,
        output_dir: str | Path = "Data/analysis_results",
        max_workers: int | None = None,
        max_group_workers: int | None = None,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> None:

        settings = get_settings()

        # By default, max_workers matches global MAX_CONCURRENT_LLM_REQUESTS
        if max_workers is None:
            max_workers = settings.MAX_CONCURRENT_LLM_REQUESTS
        
        if max_group_workers is None:
            max_group_workers = settings.GROUP_WORKERS

        if max_workers <= 0:
            raise ValueError(
                "max_workers must be greater than 0."
            )

        if max_group_workers <= 0:
            raise ValueError(
                "max_group_workers must be greater than 0."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        self.knowledge_base = knowledge_base

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.max_workers = max_workers
        self.max_group_workers = max_group_workers
        self.top_k = top_k
        self.min_score = min_score

        self.analyzer = AnalyzerAgent(
            knowledge_base=self.knowledge_base,
            top_k=self.top_k,
            min_score=self.min_score,
            max_group_workers=self.max_group_workers,
        )

        # Protects file writes and shared state counters
        self._lock = threading.Lock()

        logger.success(
            "ComplianceOrchestrator initialized."
        )
        logger.info(
            f"Global LLM Worker Limit : {self.max_workers}"
        )
        logger.info(
            f"Top-K                  : {self.top_k}"
        )
        logger.info(
            f"Output dir             : {self.output_dir}"
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def run(
        self,
        *,
        start_article: int = 1,
        end_article: int | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze a range of GDPR articles using a global group queue.
        """

        total_articles = self.knowledge_base.article_count()

        if end_article is None:
            end_article = total_articles

        if start_article <= 0:
            raise ValueError("start_article must be greater than 0.")

        if end_article > total_articles:
            raise ValueError(
                f"end_article={end_article} exceeds knowledge base article count ({total_articles})."
            )

        if start_article > end_article:
            raise ValueError("start_article cannot be greater than end_article.")

        article_numbers = list(range(start_article, end_article + 1))

        # -----------------------------------------------------
        # 1. Filter completed articles (Resume behavior)
        # -----------------------------------------------------
        completed = []

        if resume:
            for article_number in article_numbers:
                if self._is_completed(article_number):
                    completed.append(article_number)

        pending = [
            article for article in article_numbers if article not in completed
        ]

        self._print_header(
            article_numbers=article_numbers,
            completed=completed,
            pending=pending,
        )

        if not pending:
            logger.success("All requested articles are already completed.")
            return {
                "status": "COMPLETED",
                "total": len(article_numbers),
                "completed": len(completed),
                "successful": len(completed),
                "failed": 0,
                "skipped": len(completed),
                "failed_articles": [],
                "output_dir": str(self.output_dir),
            }

        # -----------------------------------------------------
        # 2. Extract group tasks and process 0-group articles
        # -----------------------------------------------------
        successful = []
        failed = []
        article_results: dict[int, dict[str, Any]] = {}

        pending_group_counts: dict[int, int] = {}
        group_results_by_article: dict[int, dict[str, dict[str, Any]]] = {}
        article_start_times: dict[int, float] = {}

        group_tasks: list[GroupTask] = []

        for article_number in pending:
            article = self.knowledge_base.get_article(article_number)
            groups = self.analyzer.group_retriever.get_groups(article_number)

            if not groups:
                # Zero-group articles complete immediately
                result = {
                    "article_number": article_number,
                    "article_title": article.article_name,
                    "checkability": article.checkability,
                    "status": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.0,
                    "group_count": 0,
                    "completed_groups": 0,
                    "groups": [],
                    "performance": {"wall_clock_time": 0.0},
                    "orchestration": {
                        "worker_managed": True,
                        "article_number": article_number,
                        "execution_mode": "zero_groups",
                    },
                }
                self._save_result(article_number, result)
                successful.append(article_number)
                article_results[article_number] = result
                self._print_completion(
                    article_number=article_number,
                    status="INSUFFICIENT_EVIDENCE",
                    confidence=0.0,
                    completed_count=len(completed) + len(successful),
                    total_count=len(article_numbers),
                )
            else:
                pending_group_counts[article_number] = len(groups)
                group_results_by_article[article_number] = {}
                article_start_times[article_number] = time.perf_counter()

                for index, group in enumerate(groups, start=1):
                    group_tasks.append(
                        GroupTask(
                            article_number=article_number,
                            group=group,
                            index=index,
                            total_groups=len(groups),
                        )
                    )

        logger.info(
            f"Global Group Queue: {len(pending_group_counts)} article(s) with groups | "
            f"{len(group_tasks)} group tasks | "
            f"concurrency limit = {self.max_workers}"
        )

        # -----------------------------------------------------
        # 3. Process Global Group Queue
        # -----------------------------------------------------
        if group_tasks:
            worker_count = min(self.max_workers, len(group_tasks))

            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="gdpr-group-queue",
            ) as executor:

                future_to_task = {
                    executor.submit(self._execute_group_task, task): task
                    for task in group_tasks
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]

                    try:
                        art_num, group_result = future.result()
                    except Exception as exc:
                        logger.exception(
                            f"Article {task.article_number} | Group {task.group.group_id} worker crashed."
                        )
                        art_num = task.article_number
                        group_result = {
                            "group_id": task.group.group_id,
                            "principle": task.group.principle,
                            "condition_logic": task.group.condition_logic,
                            "status": "INSUFFICIENT_EVIDENCE",
                            "confidence": 0.0,
                            "reason": "Group worker crashed.",
                            "gap": str(exc),
                            "evidence_count": 0,
                            "sub_obligations": [],
                            "error": str(exc),
                        }

                    # Check if all groups for this article have completed
                    with self._lock:
                        group_results_by_article[art_num][group_result["group_id"]] = group_result
                        pending_group_counts[art_num] -= 1

                        if pending_group_counts[art_num] == 0:
                            art_obj = self.knowledge_base.get_article(art_num)
                            canonical_groups = self.analyzer.group_retriever.get_groups(art_num)
                            ordered_group_results = [
                                group_results_by_article[art_num][g.group_id]
                                for g in canonical_groups
                            ]

                            art_status = self.analyzer._aggregate_article_status(ordered_group_results)
                            art_confidence = self.analyzer._calculate_article_confidence(ordered_group_results)
                            art_duration = time.perf_counter() - article_start_times[art_num]

                            total_retrieval = sum(g.get("performance", {}).get("retrieval_time", 0.0) for g in ordered_group_results)
                            total_llm = sum(g.get("performance", {}).get("llm_time", 0.0) for g in ordered_group_results)
                            total_val = sum(g.get("performance", {}).get("validation_time", 0.0) for g in ordered_group_results)
                            total_backoff = sum(g.get("performance", {}).get("backoff_time", 0.0) for g in ordered_group_results)
                            total_attempts = sum(g.get("performance", {}).get("attempts", 1) for g in ordered_group_results)
                            total_429 = sum(g.get("performance", {}).get("count_429", 0) for g in ordered_group_results)
                            total_5xx = sum(g.get("performance", {}).get("count_5xx", 0) for g in ordered_group_results)
                            total_malformed = sum(g.get("performance", {}).get("count_malformed_json", 0) for g in ordered_group_results)
                            total_val_failures = sum(g.get("performance", {}).get("count_validation_failures", 0) for g in ordered_group_results)

                            art_result = {
                                "article_number": art_num,
                                "article_title": art_obj.article_name,
                                "checkability": art_obj.checkability,
                                "status": art_status,
                                "confidence": art_confidence,
                                "group_count": len(canonical_groups),
                                "completed_groups": len(ordered_group_results),
                                "groups": ordered_group_results,
                                "performance": {
                                    "wall_clock_time": round(art_duration, 4),
                                    "total_retrieval_time": round(total_retrieval, 4),
                                    "total_llm_time": round(total_llm, 4),
                                    "total_validation_time": round(total_val, 4),
                                    "total_backoff_time": round(total_backoff, 4),
                                    "total_attempts": total_attempts,
                                    "count_429": total_429,
                                    "count_5xx": total_5xx,
                                    "count_malformed_json": total_malformed,
                                    "count_validation_failures": total_val_failures,
                                },
                                "orchestration": {
                                    "worker_managed": True,
                                    "article_number": art_num,
                                    "execution_mode": "global_group_queue",
                                },
                            }

                            self._save_result(art_num, art_result)
                            successful.append(art_num)
                            article_results[art_num] = art_result

                            self._print_completion(
                                article_number=art_num,
                                status=art_status,
                                confidence=art_confidence,
                                completed_count=len(completed) + len(successful),
                                total_count=len(article_numbers),
                            )

        # -----------------------------------------------------
        # 4. Final Summary
        # -----------------------------------------------------
        total_successful = len(completed) + len(successful)

        summary = {
            "status": (
                "COMPLETED"
                if not failed
                else "COMPLETED_WITH_FAILURES"
            ),
            "total": len(article_numbers),
            "completed": total_successful,
            "successful_this_run": len(successful),
            "resumed": len(completed),
            "failed": len(failed),
            "failed_articles": sorted(failed),
            "output_dir": str(self.output_dir),
            "article_results": article_results,
        }

        # Include direct article mapping for test runners (e.g. results.get(5))
        for art_num, art_res in article_results.items():
            summary[art_num] = art_res
            summary[str(art_num)] = art_res

        self._print_final_summary(summary)
        return summary

    def _execute_group_task(self, task: GroupTask) -> tuple[int, dict[str, Any]]:
        """Process one group task using AnalyzerAgent."""
        result = self.analyzer._process_group(
            article_number=task.article_number,
            group=task.group,
            index=task.index,
            total=task.total_groups,
        )
        return task.article_number, result

    # =========================================================
    # ARTICLE WORKER (Backward Compatibility helper)
    # =========================================================

    def _process_article(
        self,
        article_number: int,
    ) -> dict[str, Any]:
        """Backward compatible single article processing."""
        result = self.analyzer.analyze_article(article_number)
        result["orchestration"] = {
            "worker_managed": True,
            "article_number": article_number,
        }
        self._save_result(article_number, result)
        return result

    # =========================================================
    # RESULT STORAGE
    # =========================================================

    def _result_path(
        self,
        article_number: int,
    ) -> Path:
        return self.output_dir / f"article_{article_number}.json"

    def _save_result(
        self,
        article_number: int,
        result: dict[str, Any],
    ) -> None:
        """
        Atomically save an article result using a temporary file.
        """
        final_path = self._result_path(article_number)
        temp_path = final_path.with_suffix(".tmp")

        with self._lock:
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(
                    result,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
                file.flush()

            temp_path.replace(final_path)

        logger.info(
            f"Article {article_number} saved to {final_path}"
        )

    def _save_failure(
        self,
        article_number: int,
        exc: Exception,
    ) -> None:
        failure = {
            "article_number": article_number,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

        failure_path = self.output_dir / f"article_{article_number}_failure.json"

        with self._lock:
            with failure_path.open("w", encoding="utf-8") as file:
                json.dump(
                    failure,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

    # =========================================================
    # RESUME
    # =========================================================

    def _is_completed(
        self,
        article_number: int,
    ) -> bool:
        path = self._result_path(article_number)
        if not path.exists():
            return False

        try:
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)

            if not isinstance(result, dict):
                return False

            required_fields = {
                "article_number",
                "status",
                "confidence",
                "groups",
            }

            if not required_fields.issubset(result.keys()):
                return False

            if result.get("status") == "FAILED":
                return False

            return True

        except (json.JSONDecodeError, OSError):
            logger.warning(
                f"Invalid result file for Article {article_number}; it will be reprocessed."
            )
            return False

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def _print_header(
        *,
        article_numbers: list[int],
        completed: list[int],
        pending: list[int],
    ) -> None:

        print()
        print("=" * 75)
        print("GDPR COMPLIANCE ORCHESTRATOR (GLOBAL GROUP QUEUE)")
        print("=" * 75)
        print(f"Total requested : {len(article_numbers)}")
        print(f"Already done    : {len(completed)}")
        print(f"Remaining       : {len(pending)}")

        if pending:
            print(f"Articles        : {pending[0]} -> {pending[-1]}")

        print("=" * 75)
        print()

    @staticmethod
    def _print_completion(
        *,
        article_number: int,
        status: str,
        confidence: float,
        completed_count: int,
        total_count: int,
    ) -> None:

        print(
            f"[{completed_count}/{total_count}] "
            f"Article {article_number:<3} "
            f"-> {status:<22} "
            f"confidence={confidence:.4f}"
        )

    @staticmethod
    def _print_final_summary(
        summary: dict[str, Any],
    ) -> None:

        print()
        print("=" * 75)
        print("ORCHESTRATION COMPLETED")
        print("=" * 75)
        print(f"Status              : {summary['status']}")
        print(f"Total articles      : {summary['total']}")
        print(f"Successful           : {summary['completed']}")
        print(f"Processed this run  : {summary['successful_this_run']}")
        print(f"Resumed/skipped      : {summary['resumed']}")
        print(f"Failed               : {summary['failed']}")

        if summary["failed_articles"]:
            print(f"Failed articles     : {summary['failed_articles']}")

        print(f"Output directory     : {summary['output_dir']}")
        print("=" * 75)