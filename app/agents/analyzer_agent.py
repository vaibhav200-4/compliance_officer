# app/agents/analyzer_agent.py

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from typing import Any

from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.compliance.group_retriever import (
    ComplianceGroupRetriever,
)
from app.compliance.judge import ComplianceJudge
from app.core.config import get_settings
from app.core.logger import get_logger


logger = get_logger()


class AnalyzerAgent:
    """
    Analyze one complete GDPR article.

    Responsibilities:
        - Load one article's requirement groups.
        - Retrieve evidence using ComplianceGroupRetriever.
        - Judge each group using ComplianceJudge.
        - Deterministically aggregate sub-obligation verdicts.
        - Return one structured article result.

    The Analyzer does NOT:
        - connect to Pinecone directly
        - build retrieval queries
        - generate embeddings
        - perform keyword retrieval
        - make article-wide LLM calls
        - orchestrate multiple articles
        - generate reports

    Concurrency:
        Groups within one article are independent of each other
        (each has its own retrieval query and its own Judge call),
        so they can be processed concurrently up to
        `max_group_workers`. This is bounded, not unbounded,
        so that N article workers x max_group_workers doesn't
        overwhelm the LLM provider.
    """

    VALID_STATUSES = {
        "MET",
        "PARTIALLY_MET",
        "NOT_MET",
        "CONFLICTING",
        "INSUFFICIENT_EVIDENCE",
        "NOT_APPLICABLE",
    }

    def __init__(
        self,
        knowledge_base: GDPRKnowledgeBase,
        *,
        group_retriever: ComplianceGroupRetriever | None = None,
        judge: ComplianceJudge | None = None,
        top_k: int = 5,
        min_score: float | None = None,
        max_group_workers: int | None = None,
        document_id: str | None = None,
    ) -> None:

        # Load default from settings if not provided
        if max_group_workers is None:
            settings = get_settings()
            max_group_workers = settings.GROUP_WORKERS

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if max_group_workers <= 0:
            raise ValueError(
                "max_group_workers must be greater than 0."
            )

        self.knowledge_base = knowledge_base
        self.document_id = document_id

        # Reuse the existing retrieval architecture.
        self.group_retriever = (
            group_retriever
            or ComplianceGroupRetriever(
                knowledge_base=knowledge_base,
                document_id=document_id,
            )
        )

        # Reuse the working Judge.
        self.judge = (
            judge
            or ComplianceJudge()
        )

        self.top_k = top_k
        self.min_score = min_score
        self.max_group_workers = max_group_workers

        logger.success(
            f"AnalyzerAgent initialized "
            f"(max_group_workers={self.max_group_workers})."
        )

    # ============================================================
    # ARTICLE
    # ============================================================

    def analyze_article(
        self,
        article_number: int,
    ) -> dict[str, Any]:

        import time
        start_art_time = time.perf_counter()

        logger.info(
            f"ARTICLE_START | Article {article_number}"
        )

        # --------------------------------------------------------
        # 1. Load article
        # --------------------------------------------------------

        article = self.knowledge_base.get_article(
            article_number
        )

        if article is None:
            raise ValueError(
                f"Article {article_number} not found."
            )

        # --------------------------------------------------------
        # 2. Get groups
        # --------------------------------------------------------

        groups = self.group_retriever.get_groups(
            article_number
        )

        if not groups:

            return {
                "article_number": article_number,
                "article_title": article.article_name,
                "checkability": article.checkability,
                "status": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "groups": [],
            }

        logger.info(
            f"Article {article_number}: "
            f"{len(groups)} groups | "
            f"max_group_workers={self.max_group_workers}"
        )

        # --------------------------------------------------------
        # 3 & 4. Retrieve + judge each group, bounded concurrency.
        #
        # Each group is an independent pipeline:
        #     retrieve evidence -> judge -> aggregate group result
        #
        # A single group's retrieval/LLM failure never aborts the
        # article; _process_group() always returns a result dict,
        # it never raises.
        # --------------------------------------------------------

        group_results_by_id: dict[str, dict[str, Any]] = {}

        if self.max_group_workers <= 1 or len(groups) <= 1:

            # Sequential path. Used when concurrency is disabled
            # (max_group_workers=1) or there's only one group,
            # where a thread pool would just add overhead.
            for index, group in enumerate(groups, start=1):

                group_results_by_id[group.group_id] = (
                    self._process_group(
                        article_number,
                        group,
                        index,
                        len(groups),
                    )
                )

        else:

            with ThreadPoolExecutor(
                max_workers=min(
                    self.max_group_workers,
                    len(groups),
                ),
                thread_name_prefix=(
                    f"article-{article_number}-group"
                ),
            ) as executor:

                future_to_group = {
                    executor.submit(
                        self._process_group,
                        article_number,
                        group,
                        index,
                        len(groups),
                    ): group
                    for index, group in enumerate(
                        groups,
                        start=1,
                    )
                }

                for future in as_completed(
                    future_to_group
                ):

                    group = future_to_group[future]

                    try:
                        result = future.result()

                    except Exception as exc:

                        # _process_group() already catches its own
                        # exceptions internally; this is a
                        # last-resort guard against something
                        # unexpected (e.g. the thread pool itself
                        # misbehaving).
                        logger.exception(
                            f"Article {article_number} | "
                            f"Group {group.group_id} crashed "
                            f"unexpectedly."
                        )

                        result = {
                            "group_id": group.group_id,
                            "principle": group.principle,
                            "condition_logic": (
                                group.condition_logic
                            ),
                            "status": "INSUFFICIENT_EVIDENCE",
                            "confidence": 0.0,
                            "reason": "Group analysis crashed.",
                            "gap": str(exc),
                            "evidence_count": 0,
                            "sub_obligations": [],
                            "error": str(exc),
                        }

                    group_results_by_id[
                        group.group_id
                    ] = result

        # Reorder to match the article's canonical group order.
        # ThreadPoolExecutor completion order is nondeterministic,
        # but downstream output (JSON files, reports) must be
        # stable/reproducible run-to-run.
        group_results = [
            group_results_by_id[group.group_id]
            for group in groups
        ]

        # --------------------------------------------------------
        # 5. Article-level status
        # --------------------------------------------------------

        article_status = (
            self._aggregate_article_status(
                group_results
            )
        )

        article_confidence = (
            self._calculate_article_confidence(
                group_results
            )
        )

        # --------------------------------------------------------
        # 6. Return complete article result with performance metrics
        # --------------------------------------------------------

        art_duration = time.perf_counter() - start_art_time

        # Aggregate performance metrics from groups
        total_retrieval = sum(g.get("performance", {}).get("retrieval_time", 0.0) for g in group_results)
        total_llm = sum(g.get("performance", {}).get("llm_time", 0.0) for g in group_results)
        total_val = sum(g.get("performance", {}).get("validation_time", 0.0) for g in group_results)
        total_backoff = sum(g.get("performance", {}).get("backoff_time", 0.0) for g in group_results)
        total_attempts = sum(g.get("performance", {}).get("attempts", 1) for g in group_results)
        total_429 = sum(g.get("performance", {}).get("count_429", 0) for g in group_results)
        total_5xx = sum(g.get("performance", {}).get("count_5xx", 0) for g in group_results)
        total_malformed = sum(g.get("performance", {}).get("count_malformed_json", 0) for g in group_results)
        total_val_failures = sum(g.get("performance", {}).get("count_validation_failures", 0) for g in group_results)

        result = {
            "article_number": article_number,
            "article_title": article.article_name,
            "checkability": article.checkability,
            "status": article_status,
            "confidence": article_confidence,
            "group_count": len(groups),
            "completed_groups": len(
                group_results
            ),
            "groups": group_results,
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
        }

        logger.success(
            f"ARTICLE_COMPLETE | Article {article_number} | "
            f"status={article_status} | "
            f"time={art_duration:.2f}s (llm={total_llm:.2f}s, ret={total_retrieval:.2f}s)"
        )

        return result

    # ============================================================
    # SINGLE GROUP PIPELINE (retrieve -> judge -> aggregate)
    #
    # This runs either inline (sequential path) or inside a
    # ThreadPoolExecutor worker (concurrent path). It must never
    # raise -- all failure modes are converted into an
    # INSUFFICIENT_EVIDENCE group result so one bad group/LLM
    # response never destroys the whole article.
    # ============================================================

    def _process_group(
        self,
        article_number: int,
        group: Any,
        index: int,
        total: int,
    ) -> dict[str, Any]:

        import time

        msg_grp_start = f"GROUP_START | Article {article_number} | Group {index}/{total} | {group.group_id}"
        logger.info(msg_grp_start)
        print(msg_grp_start, flush=True)

        group_evidence = None

        try:
            msg_ret_start = f"RETRIEVAL_START | Article={article_number} | Group={group.group_id}"
            logger.info(msg_ret_start)
            print(msg_ret_start, flush=True)

            t0 = time.perf_counter()
            group_evidence = self.group_retriever.retrieve_group(
                article_number=article_number,
                group_id=group.group_id,
                top_k=self.top_k,
                min_score=self.min_score,
            )
            retrieval_dur = getattr(group_evidence, "retrieval_duration", time.perf_counter() - t0)

            msg_ret_comp = f"RETRIEVAL_COMPLETE | Article={article_number} | Group={group.group_id} | duration={retrieval_dur:.2f}s"
            logger.info(msg_ret_comp)
            print(msg_ret_comp, flush=True)

            msg_judge_start = f"JUDGE_START | Article={article_number} | Group={group.group_id}"
            logger.info(msg_judge_start)
            print(msg_judge_start, flush=True)

            t_j = time.perf_counter()
            sub_verdicts = self.judge.evaluate(
                group=group,
                group_evidence=group_evidence,
            )
            judge_dur = time.perf_counter() - t_j

            msg_judge_comp = f"JUDGE_COMPLETE | Article={article_number} | Group={group.group_id} | duration={judge_dur:.2f}s"
            logger.info(msg_judge_comp)
            print(msg_judge_comp, flush=True)

            group_status = self._aggregate_group_status(
                group.condition_logic,
                sub_verdicts,
            )

            group_confidence = self._calculate_confidence(
                sub_verdicts
            )

            metrics = getattr(sub_verdicts, "metrics", None)

            llm_sec = metrics.llm_time if metrics else 0.0
            val_sec = metrics.validation_time if metrics else 0.0
            attempt = metrics.attempts if metrics else 1
            provider = metrics.provider if metrics else ""
            masked_key = metrics.endpoint_masked_key if metrics else ""

            group_result = {
                "article_number": article_number,
                "group_id": group.group_id,
                "principle": group.principle,
                "condition_logic": group.condition_logic,
                "status": group_status,
                "confidence": group_confidence,
                "reason": self._build_reason(sub_verdicts),
                "gap": self._build_gap(group_status, sub_verdicts),
                "evidence_count": group_evidence.evidence_count,
                "sub_obligations": [
                    self._serialize(verdict)
                    for verdict in sub_verdicts
                ],
                "performance": {
                    "retrieval_time": round(retrieval_dur, 4),
                    "llm_time": round(llm_sec, 4),
                    "validation_time": round(val_sec, 4),
                    "backoff_time": round(metrics.backoff_time, 4) if metrics else 0.0,
                    "attempts": attempt,
                    "count_429": metrics.count_429 if metrics else 0,
                    "count_5xx": metrics.count_5xx if metrics else 0,
                    "count_malformed_json": metrics.count_malformed_json if metrics else 0,
                    "count_validation_failures": metrics.count_validation_failures if metrics else 0,
                    "provider": provider,
                    "endpoint_key": masked_key,
                },
            }

            msg_grp_comp = (
                f"GROUP_COMPLETE | Article {article_number} | Group {group.group_id} | "
                f"obligations={len(group.obligations)} | "
                f"retrieval={retrieval_dur:.2f}s | "
                f"llm={llm_sec:.2f}s | "
                f"status={group_status}"
            )
            logger.info(msg_grp_comp)
            print(msg_grp_comp, flush=True)

            return group_result

        except Exception as exc:

            logger.exception(
                f"Article {article_number} | "
                f"Group {group.group_id} failed."
            )

            evidence_count = (
                group_evidence.evidence_count
                if group_evidence is not None
                else 0
            )

            return {
                "group_id": group.group_id,
                "principle": group.principle,
                "condition_logic": group.condition_logic,
                "status": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "reason": "Group analysis failed.",
                "gap": str(exc),
                "evidence_count": evidence_count,
                "sub_obligations": [],
                "error": str(exc),
                "performance": {
                    "retrieval_time": 0.0,
                    "llm_time": 0.0,
                    "validation_time": 0.0,
                    "backoff_time": 0.0,
                    "attempts": 1,
                    "error": str(exc),
                },
            }

    # ============================================================
    # GROUP VERDICT
    # ============================================================

    def _aggregate_group_status(
        self,
        logic: str,
        verdicts: list[Any],
    ) -> str:

        if not verdicts:
            return "INSUFFICIENT_EVIDENCE"

        statuses = [
            str(
                getattr(
                    verdict,
                    "status",
                    "INSUFFICIENT_EVIDENCE",
                )
            ).upper()
            for verdict in verdicts
        ]

        if any(
            status == "CONFLICTING"
            for status in statuses
        ):
            return "CONFLICTING"

        logic = str(
            logic or "SINGLE"
        ).upper()

        # --------------------------------------------------------
        # SINGLE
        # --------------------------------------------------------

        if logic == "SINGLE":

            return statuses[0]

        # --------------------------------------------------------
        # ALL
        # --------------------------------------------------------

        if logic == "ALL":

            if all(
                status == "MET"
                for status in statuses
            ):
                return "MET"

            if any(
                status == "PARTIALLY_MET"
                for status in statuses
            ):
                return "PARTIALLY_MET"

            if any(
                status == "MET"
                for status in statuses
            ):
                return "PARTIALLY_MET"

            if any(
                status == "INSUFFICIENT_EVIDENCE"
                for status in statuses
            ):
                return "INSUFFICIENT_EVIDENCE"

            if all(
                status == "NOT_APPLICABLE"
                for status in statuses
            ):
                return "NOT_APPLICABLE"

            return "NOT_MET"

        # --------------------------------------------------------
        # ANY
        # --------------------------------------------------------

        if logic == "ANY":

            if "MET" in statuses:
                return "MET"

            if "PARTIALLY_MET" in statuses:
                return "PARTIALLY_MET"

            if any(
                status == "INSUFFICIENT_EVIDENCE"
                for status in statuses
            ):
                return "INSUFFICIENT_EVIDENCE"

            if all(
                status == "NOT_APPLICABLE"
                for status in statuses
            ):
                return "NOT_APPLICABLE"

            return "NOT_MET"

        raise ValueError(
            f"Unsupported condition logic: {logic}"
        )

    # ============================================================
    # ARTICLE STATUS
    # ============================================================

    @staticmethod
    def _aggregate_article_status(
        group_results: list[dict[str, Any]],
    ) -> str:

        if not group_results:
            return "INSUFFICIENT_EVIDENCE"

        statuses = [
            result["status"]
            for result in group_results
        ]

        if "CONFLICTING" in statuses:
            return "CONFLICTING"

        if "NOT_MET" in statuses:
            return "NOT_MET"

        if "PARTIALLY_MET" in statuses:
            return "PARTIALLY_MET"

        if "INSUFFICIENT_EVIDENCE" in statuses:
            return "INSUFFICIENT_EVIDENCE"

        if all(
            status == "NOT_APPLICABLE"
            for status in statuses
        ):
            return "NOT_APPLICABLE"

        return "MET"

    # ============================================================
    # CONFIDENCE
    # ============================================================

    @staticmethod
    def _calculate_confidence(
        verdicts: list[Any],
    ) -> float:

        if not verdicts:
            return 0.0

        values = []

        for verdict in verdicts:

            value = getattr(
                verdict,
                "confidence",
                0.0,
            )

            try:
                value = float(value)
            except (
                TypeError,
                ValueError,
            ):
                value = 0.0

            values.append(
                max(
                    0.0,
                    min(1.0, value),
                )
            )

        return round(
            sum(values) / len(values),
            4,
        )

    @staticmethod
    def _calculate_article_confidence(
        group_results: list[dict[str, Any]],
    ) -> float:

        if not group_results:
            return 0.0

        values = [
            float(
                result.get(
                    "confidence",
                    0.0,
                )
            )
            for result in group_results
        ]

        return round(
            sum(values) / len(values),
            4,
        )

    # ============================================================
    # REASON / GAP
    # ============================================================

    @staticmethod
    def _build_reason(
        verdicts: list[Any],
    ) -> str:

        reasons = []

        for verdict in verdicts:

            reason = getattr(
                verdict,
                "reason",
                None,
            )

            if reason:
                reasons.append(
                    str(reason)
                )

        if not reasons:
            return (
                "No detailed reason was provided."
            )

        return " ".join(reasons)

    @staticmethod
    def _build_gap(
        status: str,
        verdicts: list[Any],
    ) -> str | None:

        if status in {
            "MET",
            "NOT_APPLICABLE",
        }:
            return None

        reasons = []

        for verdict in verdicts:

            verdict_status = str(
                getattr(
                    verdict,
                    "status",
                    "",
                )
            ).upper()

            if verdict_status in {
                "PARTIALLY_MET",
                "NOT_MET",
                "CONFLICTING",
                "INSUFFICIENT_EVIDENCE",
            }:

                reason = getattr(
                    verdict,
                    "reason",
                    None,
                )

                if reason:
                    reasons.append(
                        str(reason)
                    )

        if not reasons:
            return (
                "The available policy evidence "
                "does not fully demonstrate compliance."
            )

        return " ".join(reasons)

    # ============================================================
    # SERIALIZATION
    # ============================================================

    @staticmethod
    def _serialize(
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if is_dataclass(value):

            return AnalyzerAgent._serialize(
                asdict(value)
            )

        if isinstance(value, dict):

            return {
                str(key): AnalyzerAgent._serialize(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):

            return [
                AnalyzerAgent._serialize(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        return str(value)