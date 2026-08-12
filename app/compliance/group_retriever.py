# app/compliance/group_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.compliance.gdpr_kb import (
    GDPRKnowledgeBase,
    SubObligation,
)
from app.compliance.retriever import (
    ComplianceRetriever,
    RetrievedEvidence,
)
from app.core.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class ComplianceGroup:
    """
    Represents one GDPR parent group and its child obligations.
    """

    article_number: int
    group_id: str
    obligations: tuple[SubObligation, ...]

    @property
    def obligation_count(self) -> int:
        return len(self.obligations)


@dataclass(frozen=True)
class GroupEvidence:
    """
    Evidence retrieved for a complete GDPR obligation group.
    """

    article_number: int
    group_id: str
    query: str
    evidence: tuple[RetrievedEvidence, ...]

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)


class ComplianceGroupRetriever:
    """
    Retrieves shared policy evidence for a GDPR parent group.

    Instead of performing one Pinecone search per atomic obligation,
    this class creates one combined query from all child obligations.

    Example:

        5.1.f
          ├── 5.1.f.1
          ├── 5.1.f.2
          ├── 5.1.f.3
          └── 5.1.f.4

    becomes:

        5.1.f
             ↓
        one combined query
             ↓
        Pinecone
             ↓
        shared evidence

    The individual obligations are still preserved and will be
    evaluated separately by the compliance judge later.
    """

    def __init__(
        self,
        knowledge_base: GDPRKnowledgeBase | None = None,
        retriever: ComplianceRetriever | None = None,
    ) -> None:

        self.knowledge_base = (
            knowledge_base
            or GDPRKnowledgeBase()
        )

        self.retriever = (
            retriever
            or ComplianceRetriever()
        )

        logger.success(
            "Compliance group retriever initialized."
        )

    # ------------------------------------------------------------------
    # Group discovery
    # ------------------------------------------------------------------

    def get_groups(
        self,
        article_number: int,
    ) -> list[ComplianceGroup]:
        """
        Get all parent groups for an article.

        Example for Article 5:

            5.1.a
            5.1.b
            5.1.c
            5.1.d
            5.1.e
            5.1.f
            5.2
        """

        obligations = self.knowledge_base.get_sub_obligations(
            article_number
        )

        grouped: dict[str, list[SubObligation]] = {}

        for obligation in obligations:

            grouped.setdefault(
                obligation.parent_group_id,
                [],
            ).append(obligation)

        groups = [
            ComplianceGroup(
                article_number=article_number,
                group_id=group_id,
                obligations=tuple(group_obligations),
            )
            for group_id, group_obligations
            in grouped.items()
        ]

        logger.info(
            f"Article {article_number}: "
            f"found {len(groups)} compliance groups."
        )

        return groups

    # ------------------------------------------------------------------
    # Group lookup
    # ------------------------------------------------------------------

    def get_group(
        self,
        article_number: int,
        group_id: str,
    ) -> ComplianceGroup:
        """
        Get one specific parent group.

        Example:

            get_group(5, "5.1.f")
        """

        groups = self.get_groups(article_number)

        for group in groups:

            if group.group_id == group_id:
                return group

        raise KeyError(
            f"GDPR group '{group_id}' "
            f"not found in Article {article_number}."
        )

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_group_query(
        group: ComplianceGroup,
    ) -> str:
        """
        Build one retrieval query from all child obligations.

        We use both plain_summary and evidence_prompt because:

            plain_summary → semantic description
            evidence_prompt → what evidence we actually want
        """

        lines: list[str] = []

        lines.append(
            f"GDPR Article {group.article_number}, "
            f"requirement group {group.group_id}."
        )

        lines.append(
            "Find company privacy policy evidence relevant "
            "to the following requirements:"
        )

        for obligation in group.obligations:

            lines.append(
                f"\nRequirement {obligation.id}:"
            )

            lines.append(
                f"Summary: {obligation.plain_summary}"
            )

            lines.append(
                f"Evidence question: "
                f"{obligation.evidence_prompt}"
            )

            # Conditional requirements should also contribute
            # their condition to retrieval context.
            if obligation.applicability_condition:

                lines.append(
                    f"Applicability condition: "
                    f"{obligation.applicability_condition}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Group retrieval
    # ------------------------------------------------------------------

    def retrieve_group(
        self,
        article_number: int,
        group_id: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> GroupEvidence:
        """
        Retrieve shared policy evidence for one GDPR group.
        """

        group = self.get_group(
            article_number,
            group_id,
        )

        query = self.build_group_query(
            group
        )

        logger.info(
            f"Retrieving group evidence | "
            f"article={article_number} | "
            f"group={group_id} | "
            f"obligations={group.obligation_count}"
        )

        evidence = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        logger.info(
            f"Group {group_id}: "
            f"retrieved {len(evidence)} evidence chunks."
        )

        return GroupEvidence(
            article_number=article_number,
            group_id=group_id,
            query=query,
            evidence=tuple(evidence),
        )

    # ------------------------------------------------------------------
    # Complete article retrieval
    # ------------------------------------------------------------------

    def retrieve_article(
        self,
        article_number: int,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[GroupEvidence]:
        """
        Retrieve shared evidence for every parent group
        in an article.

        This is the first level of article-wide analysis.

        Example:

            Article 5
              ↓
            7 groups
              ↓
            7 Pinecone searches
        """

        groups = self.get_groups(
            article_number
        )

        results: list[GroupEvidence] = []

        for group in groups:

            result = self.retrieve_group(
                article_number=article_number,
                group_id=group.group_id,
                top_k=top_k,
                min_score=min_score,
            )

            results.append(result)

        logger.success(
            f"Article {article_number}: "
            f"retrieved evidence for "
            f"{len(results)} groups."
        )

        return results
