# app/agents/orchestrator.py

from __future__ import annotations

import json
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

from app.agents.analyzer_agent import AnalyzerAgent
from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.core.logger import get_logger


logger = get_logger()


class ComplianceOrchestrator:
    """
    Orchestrates GDPR article-level compliance analysis.

    Responsibilities
    ----------------
    - Select articles for analysis.
    - Run AnalyzerAgent instances in controlled parallelism.
    - Save each article result immediately.
    - Resume from previously completed articles.
    - Isolate article-level failures.
    - Maintain execution progress.

    Architecture
    ------------
        Orchestrator
             |
             +-- Article workers (max_workers)
             |        |
             |        +-- AnalyzerAgent.analyze_article()
             |                |
             |                +-- Group workers (max_group_workers)
             |                        |
             |                        +-- retrieve -> judge -> aggregate
             |
             +-- incremental result files

    Two-level bounded parallelism:
        max_workers * max_group_workers = peak concurrent LLM calls.
        Defaults (3 * 2 = 6) are intentionally conservative relative
        to launching one call per group across all articles at once.
    """

    def __init__(
        self,
        knowledge_base: GDPRKnowledgeBase,
        *,
        output_dir: str | Path = "Data/analysis_results",
        max_workers: int = 3,
        max_group_workers: int = 2,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> None:

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

        # -----------------------------------------------------
        # One AnalyzerAgent is shared by all article workers.
        #
        # The Analyzer already owns:
        #   GroupRetriever
        #   ComplianceRetriever
        #   Pinecone
        #   ComplianceJudge
        #
        # These components hold no mutable per-request state,
        # so sharing one Analyzer graph across threads is safe:
        #   - ComplianceJudge._call_llm() builds a fresh
        #     request per call (no shared session/state).
        #   - PineconeManager queries are per-call network
        #     calls against a client documented as safe for
        #     concurrent reads.
        #   - ComplianceGroupRetriever's only mutable field
        #     (embedding_cache) is lazily built behind a lock.
        #
        # Do NOT recreate those components here, and do NOT
        # switch to thread-local Analyzer instances unless a
        # future component introduces real unguarded mutable
        # state.
        # -----------------------------------------------------

        self.analyzer = AnalyzerAgent(
            knowledge_base=self.knowledge_base,
            top_k=self.top_k,
            min_score=self.min_score,
            max_group_workers=self.max_group_workers,
        )

        # Protects file writes and progress counters.
        self._lock = threading.Lock()

        logger.success(
            "ComplianceOrchestrator initialized."
        )

        logger.info(
            f"Article workers : {self.max_workers}"
        )

        logger.info(
            f"Group workers   : {self.max_group_workers}"
        )

        logger.info(
            f"Top-K           : {self.top_k}"
        )

        logger.info(
            f"Output dir      : {self.output_dir}"
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
        Analyze a range of GDPR articles.

        Parameters
        ----------
        start_article:
            First article to process.

        end_article:
            Last article to process.
            If None, all articles in the KB are processed.

        resume:
            If True, existing successful article result files
            are skipped.

        Returns
        -------
        dict
            Overall orchestration summary.
        """

        total_articles = (
            self.knowledge_base.article_count()
        )

        if end_article is None:
            end_article = total_articles

        if start_article <= 0:
            raise ValueError(
                "start_article must be greater than 0."
            )

        if end_article > total_articles:
            raise ValueError(
                f"end_article={end_article} exceeds "
                f"knowledge base article count "
                f"({total_articles})."
            )

        if start_article > end_article:
            raise ValueError(
                "start_article cannot be greater than "
                "end_article."
            )

        article_numbers = list(
            range(
                start_article,
                end_article + 1,
            )
        )

        # -----------------------------------------------------
        # Find already completed articles
        # -----------------------------------------------------

        completed = []

        if resume:

            for article_number in article_numbers:

                if self._is_completed(
                    article_number
                ):
                    completed.append(
                        article_number
                    )

        pending = [
            article
            for article in article_numbers
            if article not in completed
        ]

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        self._print_header(
            article_numbers=article_numbers,
            completed=completed,
            pending=pending,
        )

        if not pending:

            logger.success(
                "All requested articles are already completed."
            )

            return {
                "status": "COMPLETED",
                "total": len(article_numbers),
                "completed": len(completed),
                "successful": len(completed),
                "failed": 0,
                "skipped": len(completed),
                "failed_articles": [],
                "output_dir": str(
                    self.output_dir
                ),
            }

        # -----------------------------------------------------
        # Run articles in parallel
        # -----------------------------------------------------

        successful = []
        failed = []

        logger.info(
            f"Starting {len(pending)} articles "
            f"with {self.max_workers} article workers "
            f"x {self.max_group_workers} group workers."
        )

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="gdpr-worker",
        ) as executor:

            future_to_article = {
                executor.submit(
                    self._process_article,
                    article_number,
                ): article_number
                for article_number in pending
            }

            for future in as_completed(
                future_to_article
            ):

                article_number = (
                    future_to_article[future]
                )

                try:

                    result = future.result()

                    successful.append(
                        article_number
                    )

                    status = result.get(
                        "status",
                        "UNKNOWN",
                    )

                    confidence = result.get(
                        "confidence",
                        0.0,
                    )

                    self._print_completion(
                        article_number=article_number,
                        status=status,
                        confidence=confidence,
                        completed_count=(
                            len(completed)
                            + len(successful)
                        ),
                        total_count=len(
                            article_numbers
                        ),
                    )

                except Exception as exc:

                    failed.append(
                        article_number
                    )

                    logger.exception(
                        f"Article {article_number} "
                        f"failed."
                    )

                    self._save_failure(
                        article_number,
                        exc,
                    )

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------

        total_successful = (
            len(completed)
            + len(successful)
        )

        summary = {
            "status": (
                "COMPLETED"
                if not failed
                else "COMPLETED_WITH_FAILURES"
            ),
            "total": len(article_numbers),
            "completed": total_successful,
            "successful_this_run": len(
                successful
            ),
            "resumed": len(completed),
            "failed": len(failed),
            "failed_articles": sorted(
                failed
            ),
            "output_dir": str(
                self.output_dir
            ),
        }

        self._print_final_summary(
            summary
        )

        return summary

    # =========================================================
    # ARTICLE WORKER
    # =========================================================

    def _process_article(
        self,
        article_number: int,
    ) -> dict[str, Any]:
        """
        Process exactly one article.

        The Analyzer remains responsible for the actual
        compliance analysis, including its own internal
        bounded group-level concurrency.
        """

        logger.info(
            f"[Article {article_number}] "
            f"Worker started."
        )

        try:

            # -------------------------------------------------
            # Analyzer does:
            #
            # Article
            #   -> Groups
            #   -> Retrieval (per group, possibly concurrent)
            #   -> Judge (per group, possibly concurrent)
            #   -> Group aggregation
            #   -> Article aggregation
            # -------------------------------------------------

            result = (
                self.analyzer.analyze_article(
                    article_number
                )
            )

            if not isinstance(
                result,
                dict,
            ):
                raise TypeError(
                    "AnalyzerAgent must return "
                    "a dictionary."
                )

            # -------------------------------------------------
            # Add orchestration metadata
            # -------------------------------------------------

            result = dict(result)

            result[
                "orchestration"
            ] = {
                "worker_managed": True,
                "article_number": article_number,
            }

            # -------------------------------------------------
            # Save immediately
            # -------------------------------------------------

            self._save_result(
                article_number,
                result,
            )

            logger.success(
                f"[Article {article_number}] "
                f"Worker completed."
            )

            return result

        except Exception as exc:

            logger.exception(
                f"[Article {article_number}] "
                f"Worker failed."
            )

            raise

    # =========================================================
    # RESULT STORAGE
    # =========================================================

    def _result_path(
        self,
        article_number: int,
    ) -> Path:

        return (
            self.output_dir
            / f"article_{article_number}.json"
        )

    def _save_result(
        self,
        article_number: int,
        result: dict[str, Any],
    ) -> None:
        """
        Atomically save an article result.

        We write to a temporary file first and then replace
        the final file. This prevents a partially written JSON
        file from being considered a completed article.
        """

        final_path = self._result_path(
            article_number
        )

        temp_path = final_path.with_suffix(
            ".tmp"
        )

        with self._lock:

            with temp_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

                file.flush()

            temp_path.replace(
                final_path
            )

        logger.info(
            f"Article {article_number} "
            f"saved to {final_path}"
        )

    def _save_failure(
        self,
        article_number: int,
        exc: Exception,
    ) -> None:
        """
        Save a failure record without marking the article
        as successfully completed.
        """

        failure = {
            "article_number": article_number,
            "status": "FAILED",
            "error_type": type(
                exc
            ).__name__,
            "error": str(exc),
        }

        failure_path = (
            self.output_dir
            / f"article_{article_number}_failure.json"
        )

        with self._lock:

            with failure_path.open(
                "w",
                encoding="utf-8",
            ) as file:

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
        """
        Determine whether an article has a valid completed
        result file.
        """

        path = self._result_path(
            article_number
        )

        if not path.exists():
            return False

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                result = json.load(
                    file
                )

            if not isinstance(
                result,
                dict,
            ):
                return False

            # -------------------------------------------------
            # A valid Analyzer result must contain these.
            # -------------------------------------------------

            required_fields = {
                "article_number",
                "status",
                "confidence",
                "groups",
            }

            if not required_fields.issubset(
                result.keys()
            ):
                return False

            if result.get(
                "status"
            ) == "FAILED":
                return False

            return True

        except (
            json.JSONDecodeError,
            OSError,
        ):

            logger.warning(
                f"Invalid result file for "
                f"Article {article_number}; "
                f"it will be reprocessed."
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
        print(
            "GDPR COMPLIANCE ORCHESTRATOR"
        )
        print("=" * 75)

        print(
            f"Total requested : "
            f"{len(article_numbers)}"
        )

        print(
            f"Already done    : "
            f"{len(completed)}"
        )

        print(
            f"Remaining       : "
            f"{len(pending)}"
        )

        if pending:

            print(
                f"Articles        : "
                f"{pending[0]} → {pending[-1]}"
            )

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
            f"→ {status:<22} "
            f"confidence={confidence:.4f}"
        )

    @staticmethod
    def _print_final_summary(
        summary: dict[str, Any],
    ) -> None:

        print()
        print("=" * 75)
        print(
            "ORCHESTRATION COMPLETED"
        )
        print("=" * 75)

        print(
            f"Status              : "
            f"{summary['status']}"
        )

        print(
            f"Total articles      : "
            f"{summary['total']}"
        )

        print(
            f"Successful           : "
            f"{summary['completed']}"
        )

        print(
            f"Processed this run  : "
            f"{summary['successful_this_run']}"
        )

        print(
            f"Resumed/skipped      : "
            f"{summary['resumed']}"
        )

        print(
            f"Failed               : "
            f"{summary['failed']}"
        )

        if summary["failed_articles"]:

            print(
                f"Failed articles     : "
                f"{summary['failed_articles']}"
            )

        print(
            f"Output directory     : "
            f"{summary['output_dir']}"
        )

        print("=" * 75)