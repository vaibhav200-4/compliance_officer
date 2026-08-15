# app/agents/analyzer_agent.py

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.compliance.group_retriever import (
    ComplianceGroupRetriever,
)
from app.compliance.judge import ComplianceJudge
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
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        self.knowledge_base = knowledge_base

        # Reuse the existing retrieval architecture.
        self.group_retriever = (
            group_retriever
            or ComplianceGroupRetriever(
                knowledge_base=knowledge_base
            )
        )

        # Reuse the working Judge.
        self.judge = (
            judge
            or ComplianceJudge()
        )

        self.top_k = top_k
        self.min_score = min_score

        logger.success(
            "AnalyzerAgent initialized."
        )

    # ============================================================
    # ARTICLE
    # ============================================================

    def analyze_article(
        self,
        article_number: int,
    ) -> dict[str, Any]:

        logger.info(
            f"Starting Article {article_number} analysis."
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
            f"{len(groups)} groups."
        )

        # --------------------------------------------------------
        # 3. Retrieve evidence for the COMPLETE article
        #
        # Existing GroupRetriever handles:
        #
        # Article
        #   -> groups
        #   -> group query
        #   -> ComplianceRetriever
        #   -> Pinecone
        # --------------------------------------------------------

        evidence_results = (
            self.group_retriever.retrieve_article(
                article_number=article_number,
                top_k=self.top_k,
                min_score=self.min_score,
            )
        )

        if len(evidence_results) != len(groups):

            raise RuntimeError(
                f"Group/evidence count mismatch for "
                f"Article {article_number}: "
                f"{len(groups)} groups vs "
                f"{len(evidence_results)} evidence results."
            )

        # --------------------------------------------------------
        # 4. Judge each group
        # --------------------------------------------------------

        group_results = []

        for index, (
            group,
            group_evidence,
        ) in enumerate(
            zip(
                groups,
                evidence_results,
            ),
            start=1,
        ):

            logger.info(
                f"Article {article_number} | "
                f"Group {index}/{len(groups)} | "
                f"{group.group_id}"
            )

            try:

                sub_verdicts = (
                    self.judge.evaluate(
                        group=group,
                        group_evidence=group_evidence,
                    )
                )

                group_status = (
                    self._aggregate_group_status(
                        group.condition_logic,
                        sub_verdicts,
                    )
                )

                group_confidence = (
                    self._calculate_confidence(
                        sub_verdicts
                    )
                )

                group_result = {
                    "group_id": group.group_id,
                    "principle": group.principle,
                    "condition_logic": (
                        group.condition_logic
                    ),
                    "status": group_status,
                    "confidence": group_confidence,
                    "reason": (
                        self._build_reason(
                            sub_verdicts
                        )
                    ),
                    "gap": (
                        self._build_gap(
                            group_status,
                            sub_verdicts,
                        )
                    ),
                    "evidence_count": (
                        group_evidence.evidence_count
                    ),
                    "sub_obligations": [
                        self._serialize(
                            verdict
                        )
                        for verdict in sub_verdicts
                    ],
                }

                group_results.append(
                    group_result
                )

                logger.success(
                    f"Article {article_number} | "
                    f"Group {group.group_id} | "
                    f"{group_status}"
                )

            except Exception as exc:

                logger.exception(
                    f"Article {article_number} | "
                    f"Group {group.group_id} failed."
                )

                # Don't let one group destroy
                # the complete article result.
                group_results.append(
                    {
                        "group_id": group.group_id,
                        "principle": group.principle,
                        "condition_logic": (
                            group.condition_logic
                        ),
                        "status": (
                            "INSUFFICIENT_EVIDENCE"
                        ),
                        "confidence": 0.0,
                        "reason": (
                            "Group analysis failed."
                        ),
                        "gap": str(exc),
                        "evidence_count": (
                            group_evidence.evidence_count
                        ),
                        "sub_obligations": [],
                        "error": str(exc),
                    }
                )

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
        # 6. Return complete article result
        # --------------------------------------------------------

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
        }

        logger.success(
            f"Article {article_number} analysis completed | "
            f"status={article_status}"
        )

        return result

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