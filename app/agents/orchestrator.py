# app/agents/orchestrator.py

from __future__ import annotations

import concurrent.futures
import json
import queue
import threading
import time
import traceback
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.report_agent import ReportAgent
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

        self.report_agent = ReportAgent(output_dir=self.output_dir)

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
        company_name: str = "Target Organization",
        policy_name: str = "Company Privacy Policy",
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a range of GDPR articles using a global group queue.
        """

        import os
        import queue
        import threading

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

        is_fast_test = os.getenv("FAST_TEST_MODE", "").lower() in ("true", "1")
        if is_fast_test:
            resume = False
            msg_res_chk = "FAST_TEST_RESUME_CHECK | resume_forced_false=True"
            logger.info(msg_res_chk)
            print(msg_res_chk, flush=True)

            fast_start = os.getenv("FAST_TEST_START_ARTICLE")
            fast_count = os.getenv("FAST_TEST_ARTICLE_COUNT")
            if fast_start and fast_start.isdigit():
                start_article = int(fast_start)
            if fast_count and fast_count.isdigit():
                end_article = min(total_articles, start_article + int(fast_count) - 1)

        article_numbers = list(range(start_article, end_article + 1))
        job_start_time = time.perf_counter()

        msg_job_start = f"JOB_START | Range: Articles {start_article} to {end_article}"
        msg_art_req = f"ANALYSIS_ARTICLES | requested={len(article_numbers)} | first={start_article} | last={end_article}"
        logger.info(msg_job_start)
        print(msg_job_start, flush=True)
        logger.info(msg_art_req)
        print(msg_art_req, flush=True)

        if is_fast_test:
            msg_fast = "FAST_TEST_MODE active | Limiting to 1 group per article, 1 attempt max, 120s timeout."
            logger.info(msg_fast)
            print(msg_fast, flush=True)

        try:
            # -----------------------------------------------------
            # Parallel Incremental Consumer Queue (ReportAgent)
            # -----------------------------------------------------
            result_queue: queue.Queue[Any] = queue.Queue()

            def _report_consumer():
                while True:
                    try:
                        item = result_queue.get()
                        if item is None:
                            result_queue.task_done()
                            break
                        self.report_agent.consume_result(item)
                    except Exception as exc:
                        logger.error(f"REPORT_AGENT_ERROR | Error consuming result: {exc}")
                    finally:
                        result_queue.task_done()

            consumer_thread = threading.Thread(
                target=_report_consumer,
                name="report-agent-consumer",
                daemon=True,
            )
            consumer_thread.start()

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

            article_results: dict[int, dict[str, Any]] = {}
            for art_num in completed:
                path = self._result_path(art_num)
                if path.exists():
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            res = json.load(f)
                            article_results[art_num] = res
                            result_queue.put(res)
                    except Exception:
                        pass

            if not pending:
                msg_all_done = "ALL_ARTICLES_COMPLETE | All requested articles are already completed."
                logger.success(msg_all_done)
                print(msg_all_done, flush=True)
                logger.info("RESULT_AGGREGATION_COMPLETE")
                print("RESULT_AGGREGATION_COMPLETE", flush=True)
                logger.info("REPORT_AGENT_START | Finalizing report...")
                print("REPORT_AGENT_START", flush=True)

                result_queue.put(None)
                consumer_thread.join(timeout=10.0)
                try:
                    final_report = self.report_agent.finalize_report(
                        company_name=company_name,
                        policy_name=policy_name,
                    )
                    logger.success("REPORT_AGENT_COMPLETE | Final report generated.")
                    print("REPORT_AGENT_COMPLETE", flush=True)
                except Exception as exc:
                    logger.error(f"REPORT_AGENT_ERROR | Exception during report generation: {exc}")
                    final_report = None

                logger.info("JOB_COMPLETED")
                print("JOB_COMPLETED", flush=True)
                return {
                    "status": "COMPLETED",
                    "total": len(article_numbers),
                    "completed": len(completed),
                    "successful": len(completed),
                    "failed": 0,
                    "skipped": len(completed),
                    "failed_articles": [],
                    "output_dir": str(self.output_dir),
                    "article_results": article_results,
                    "final_report": final_report,
                    "report_path": str(self.output_dir / "final_report.json"),
                    "report_md_path": str(self.output_dir / "report.md"),
                }

            # -----------------------------------------------------
            # 2. Extract group tasks and process 0-group articles
            # -----------------------------------------------------
            successful = []
            failed = []

            pending_group_counts: dict[int, int] = {}
            group_results_by_article: dict[int, dict[str, dict[str, Any]]] = {}
            article_start_times: dict[int, float] = {}

            group_tasks: list[GroupTask] = []

            for article_number in pending:
                msg_art_start = f"ARTICLE_START | Article={article_number}"
                logger.info(msg_art_start)
                print(msg_art_start, flush=True)

                article = self.knowledge_base.get_article(article_number)
                groups = self.analyzer.group_retriever.get_groups(article_number)

                if is_fast_test and groups:
                    groups = groups[:1]
                    msg_lim = f"FAST_TEST_GROUP_LIMIT | Article={article_number} | groups={len(groups)}"
                    logger.info(msg_lim)
                    print(msg_lim, flush=True)

                if not groups:
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
                    result_queue.put(result)
                    logger.info(f"ARTICLE_FINALIZE_COMPLETE | Article={article_number}")
                    print(f"ARTICLE_FINALIZE_COMPLETE | Article={article_number}", flush=True)
                    self._print_completion(
                        article_number=article_number,
                        status="INSUFFICIENT_EVIDENCE",
                        confidence=0.0,
                        completed_count=len(completed) + len(successful),
                        total_count=len(article_numbers),
                    )
                    if progress_callback:
                        try:
                            progress_callback(
                                len(completed) + len(successful),
                                len(article_numbers),
                                article_number,
                                "N/A",
                            )
                        except Exception:
                            pass
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

            msg_queue = (
                f"Global Group Queue: {len(pending_group_counts)} article(s) with groups | "
                f"{len(group_tasks)} group tasks | "
                f"concurrency limit = {self.max_workers}"
            )
            logger.info(msg_queue)
            print(msg_queue, flush=True)

            # -----------------------------------------------------
            # 3. Process Global Group Queue with Deadline Timeout
            # -----------------------------------------------------
            if group_tasks:
                worker_count = min(self.max_workers, len(group_tasks))
                executor = ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="gdpr-group-queue",
                )

                future_to_task = {}
                for task in group_tasks:
                    fut = executor.submit(self._execute_group_task, task)
                    future_to_task[fut] = task
                    msg_sub = f"GROUP_SUBMITTED | Article={task.article_number} | Group={task.group.group_id}"
                    logger.info(msg_sub)
                    print(msg_sub, flush=True)

                fast_test_timeout = float(os.getenv("FAST_TEST_TIMEOUT", "120.0"))
                remaining_timeout = (
                    max(0.1, fast_test_timeout - (time.perf_counter() - job_start_time))
                    if is_fast_test
                    else None
                )

                try:
                    for future in as_completed(future_to_task, timeout=remaining_timeout):
                        task = future_to_task[future]
                        try:
                            art_num, group_result = future.result()
                        except Exception as exc:
                            msg_err = f"GROUP_ERROR | Article={task.article_number} | Group={task.group.group_id} | Exception: {exc}"
                            logger.error(msg_err)
                            print(msg_err, flush=True)
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

                        self._store_group_result(
                            art_num=art_num,
                            group_result=group_result,
                            group_results_by_article=group_results_by_article,
                            pending_group_counts=pending_group_counts,
                            article_start_times=article_start_times,
                            completed=completed,
                            successful=successful,
                            pending=pending,
                            article_numbers=article_numbers,
                            article_results=article_results,
                            result_queue=result_queue,
                            progress_callback=progress_callback,
                        )

                except concurrent.futures.TimeoutError:
                    msg_tout = f"JOB_TIMEOUT | Fast test mode reached {fast_test_timeout}s wall-clock deadline."
                    logger.error(msg_tout)
                    print(msg_tout, flush=True)

                    unfinished_tasks = [task for fut, task in future_to_task.items() if not fut.done()]
                    for task in unfinished_tasks:
                        msg_unf = f"UNFINISHED_GROUP | Article={task.article_number} | Group={task.group.group_id}"
                        logger.error(msg_unf)
                        print(msg_unf, flush=True)

                        timeout_result = {
                            "group_id": task.group.group_id,
                            "principle": task.group.principle,
                            "condition_logic": task.group.condition_logic,
                            "status": "INSUFFICIENT_EVIDENCE",
                            "confidence": 0.0,
                            "reason": f"Group worker exceeded FAST_TEST_MODE {fast_test_timeout}s timeout.",
                            "gap": "FAST_TEST_MODE timeout",
                            "evidence_count": 0,
                            "sub_obligations": [],
                            "error": "GROUP_TIMEOUT",
                        }
                        print(
                            f"GROUP_TIMEOUT_RESULT_CREATED | Article={task.article_number} | Group={task.group.group_id}",
                            flush=True,
                        )

                        self._store_group_result(
                            art_num=task.article_number,
                            group_result=timeout_result,
                            group_results_by_article=group_results_by_article,
                            pending_group_counts=pending_group_counts,
                            article_start_times=article_start_times,
                            completed=completed,
                            successful=successful,
                            pending=pending,
                            article_numbers=article_numbers,
                            article_results=article_results,
                            result_queue=result_queue,
                            progress_callback=progress_callback,
                        )

                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

            # -----------------------------------------------------
            # 4. Final Summary & Report Generation
            # -----------------------------------------------------
            total_successful = len(completed) + len(successful)
            msg_all_comp = f"ALL_ARTICLES_COMPLETE | Processed {total_successful}/{len(article_numbers)} articles."
            logger.info(msg_all_comp)
            print(msg_all_comp, flush=True)
            logger.info("RESULT_AGGREGATION_START | Aggregating final results...")
            logger.info("RESULT_AGGREGATION_COMPLETE")
            print("RESULT_AGGREGATION_COMPLETE", flush=True)

            # Stop ReportAgent consumer thread and trigger finalization
            result_queue.put(None)
            consumer_thread.join(timeout=10.0)
            if consumer_thread.is_alive():
                msg_rep_to = "REPORT_AGENT_CONSUMER_TIMEOUT"
                logger.warning(msg_rep_to)
                print(msg_rep_to, flush=True)
            else:
                msg_rep_jo = "REPORT_AGENT_CONSUMER_JOINED"
                logger.info(msg_rep_jo)
                print(msg_rep_jo, flush=True)

            logger.info("REPORT_AGENT_START | Triggering Report Agent finalization...")
            print("REPORT_AGENT_START", flush=True)
            try:
                final_report = self.report_agent.finalize_report(
                    company_name=company_name,
                    policy_name=policy_name,
                )
                logger.success("REPORT_AGENT_COMPLETE | Final report generated.")
                print("REPORT_AGENT_COMPLETE", flush=True)
            except Exception as exc:
                logger.error(f"REPORT_AGENT_ERROR | Report generation failed: {exc}")
                print(f"REPORT_AGENT_ERROR | {exc}", flush=True)
                final_report = None

            summary = {
                "status": (
                    "COMPLETED"
                    if not failed and final_report is not None
                    else ("PARTIAL_REPORT" if final_report is None else "COMPLETED_WITH_FAILURES")
                ),
                "total": len(article_numbers),
                "completed": total_successful,
                "successful_this_run": len(successful),
                "resumed": len(completed),
                "failed": len(failed),
                "failed_articles": sorted(failed),
                "output_dir": str(self.output_dir),
                "article_results": article_results,
                "final_report": final_report,
                "report_path": str(self.output_dir / "final_report.json"),
                "report_md_path": str(self.output_dir / "report.md"),
            }

            for art_num, art_res in article_results.items():
                summary[art_num] = art_res
                summary[str(art_num)] = art_res

            self._print_final_summary(summary)
            logger.info("JOB_COMPLETED")
            print("JOB_COMPLETED", flush=True)
            return summary

        except Exception as exc:
            msg_fail = f"JOB_FAILED | Exception during orchestration: {exc}"
            logger.exception(msg_fail)
            print(msg_fail, flush=True)
            raise

    def _store_group_result(
        self,
        art_num: int,
        group_result: dict[str, Any],
        group_results_by_article: dict[int, dict[str, dict[str, Any]]],
        pending_group_counts: dict[int, int],
        article_start_times: dict[int, float],
        completed: list[int],
        successful: list[int],
        pending: list[int],
        article_numbers: list[int],
        article_results: dict[int, dict[str, Any]],
        result_queue: queue.Queue,
        progress_callback: Any,
    ) -> None:
        """Store group result thread-safely, avoiding duplicate decrements."""
        grp_id = group_result.get("group_id", "unknown")
        should_finalize = False
        art_groups_copy: dict[str, dict[str, Any]] = {}

        with self._lock:
            if art_num not in group_results_by_article:
                group_results_by_article[art_num] = {}

            if grp_id in group_results_by_article[art_num]:
                return

            group_result["article_number"] = art_num
            group_results_by_article[art_num][grp_id] = group_result
            if art_num in pending_group_counts:
                pending_group_counts[art_num] -= 1
                rem = pending_group_counts[art_num]
            else:
                rem = 0

            msg_ret = f"GROUP_RESULT_RETURNED | Article={art_num} | Group={grp_id}"
            msg_sto = f"GROUP_RESULT_STORED | Article={art_num} | Group={grp_id} | remaining={rem}"
            logger.info(msg_ret)
            print(msg_ret, flush=True)
            logger.info(msg_sto)
            print(msg_sto, flush=True)

            result_queue.put(group_result)

            if pending_group_counts.get(art_num, 1) == 0:
                should_finalize = True
                art_groups_copy = dict(group_results_by_article[art_num])

        # Release lock BEFORE calling finalize_article to prevent self-deadlock in _save_result
        if should_finalize:
            total_grp_cnt = len(art_groups_copy)
            msg_complete = f"ARTICLE_GROUPS_COMPLETE | Article={art_num} | completed={total_grp_cnt}/{total_grp_cnt}"
            msg_fin_start = f"ARTICLE_FINALIZE_START | Article={art_num}"
            logger.info(msg_complete)
            print(msg_complete, flush=True)
            logger.info(msg_fin_start)
            print(msg_fin_start, flush=True)

            art_result = self.finalize_article(
                art_num=art_num,
                group_results_by_article=art_groups_copy,
                start_time=article_start_times.get(art_num, time.perf_counter()),
            )

            msg_fin_comp = f"ARTICLE_FINALIZE_COMPLETE | Article={art_num}"
            logger.info(msg_fin_comp)
            print(msg_fin_comp, flush=True)

            with self._lock:
                successful.append(art_num)
                article_results[art_num] = art_result
            result_queue.put(art_result)

            current_idx = pending.index(art_num) if art_num in pending else -1
            if current_idx >= 0 and current_idx + 1 < len(pending):
                next_art = pending[current_idx + 1]
                msg_next = f"NEXT_ARTICLE | from={art_num} | to={next_art}"
                logger.info(msg_next)
                print(msg_next, flush=True)

            self._print_completion(
                article_number=art_num,
                status=art_result["status"],
                confidence=art_result["confidence"],
                completed_count=len(completed) + len(successful),
                total_count=len(article_numbers),
            )

            if progress_callback:
                try:
                    progress_callback(
                        len(completed) + len(successful),
                        len(article_numbers),
                        art_num,
                        grp_id,
                    )
                except Exception:
                    pass

    def finalize_article(
        self,
        art_num: int,
        group_results_by_article: dict[str, dict[str, Any]],
        start_time: float,
    ) -> dict[str, Any]:
        """
        Pure in-memory aggregation of group results for an article.
        Saves result JSON to disk without calling LLM, Pinecone, or group_retriever.
        """
        art_obj = self.knowledge_base.get_article(art_num)
        ordered_group_results = list(group_results_by_article.values())

        msg_agg = f"ARTICLE_FINALIZE_AGGREGATING | Article={art_num} | groups={len(ordered_group_results)}"
        logger.info(msg_agg)
        print(msg_agg, flush=True)

        art_status = self.analyzer._aggregate_article_status(ordered_group_results)
        art_confidence = self.analyzer._calculate_article_confidence(ordered_group_results)
        art_duration = time.perf_counter() - start_time

        art_result = {
            "article_number": art_num,
            "article_title": art_obj.article_name,
            "checkability": art_obj.checkability,
            "status": art_status,
            "confidence": art_confidence,
            "group_count": len(ordered_group_results),
            "completed_groups": len(ordered_group_results),
            "groups": ordered_group_results,
            "performance": {
                "wall_clock_time": round(art_duration, 4),
            },
            "orchestration": {
                "worker_managed": True,
                "article_number": art_num,
                "execution_mode": "global_group_queue",
            },
        }

        self._save_result(art_num, art_result)
        msg_saved = f"ARTICLE_RESULT_SAVED | Article={art_num}"
        logger.info(msg_saved)
        print(msg_saved, flush=True)

        msg_art_comp = (
            f"ARTICLE_COMPLETE | Article {art_num} | "
            f"status={art_status} | "
            f"confidence={art_confidence:.4f} | "
            f"groups={len(ordered_group_results)} | "
            f"duration={art_duration:.2f}s"
        )
        logger.info(msg_art_comp)
        print(msg_art_comp, flush=True)
        return art_result

    _finalize_article = finalize_article

    def _execute_group_task(self, task: GroupTask) -> tuple[int, dict[str, Any]]:
        """Process one group task using AnalyzerAgent."""
        msg_start = f"GROUP_WORKER_START | Article={task.article_number} | Group={task.group.group_id}"
        logger.info(msg_start)
        print(msg_start, flush=True)

        result = self.analyzer._process_group(
            article_number=task.article_number,
            group=task.group,
            index=task.index,
            total=task.total_groups,
        )
        if isinstance(result, dict):
            result["article_number"] = task.article_number

        msg_ret = f"GROUP_WORKER_RETURNED | Article={task.article_number} | Group={task.group.group_id} | status={result.get('status')}"
        logger.info(msg_ret)
        print(msg_ret, flush=True)

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