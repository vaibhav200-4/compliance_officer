# app/compliance/group_retriever
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.compliance.gdpr_kb import (
    GDPRKnowledgeBase,
    RequirementGroup,
    SubObligation,
)

from app.compliance.retriever import (
    ComplianceRetriever,
    RetrievedEvidence,
)

from app.core.logger import get_logger

from app.vectorstore.pinecone import (
    COMPANY_POLICY_NAMESPACE,
)


if TYPE_CHECKING:
    from app.compliance.gdpr_embeddings import (
        GDPRGroupEmbeddingCache,
    )


logger = get_logger()


# ================================================================
# DATA CLASSES
# ================================================================

@dataclass(frozen=True)
class ComplianceGroup:

    article_number: int
    group_id: str
    condition_logic: str
    principle: str
    requirement_summary: str
    applicability_condition: str | None
    obligations: tuple[SubObligation, ...]
    keywords: tuple[str, ...]
    expected_evidence: tuple[str, ...]
    assessment_rules: dict[str, str]

    @property
    def obligation_count(self) -> int:
        return len(self.obligations)


@dataclass(frozen=True)
class GroupEvidence:

    article_number: int
    group_id: str
    query: str
    evidence: tuple[RetrievedEvidence, ...]

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)


# ================================================================
# RETRIEVER
# ================================================================

class ComplianceGroupRetriever:

    def __init__(
        self,
        knowledge_base: GDPRKnowledgeBase | None = None,
        retriever: ComplianceRetriever | None = None,
        embedding_cache: "GDPRGroupEmbeddingCache | None" = None,
    ) -> None:

        self.knowledge_base = (
            knowledge_base
            or GDPRKnowledgeBase()
        )

        self.retriever = (
            retriever
            or ComplianceRetriever()
        )

        self.embedding_cache = (
            embedding_cache
        )

        logger.success(
            "ComplianceGroupRetriever initialized."
        )

    # ============================================================
    # GET GROUPS
    # ============================================================

    def get_groups(
        self,
        article_number: int,
    ) -> list[ComplianceGroup]:

        raw_groups = (
            self.knowledge_base.get_groups(
                article_number
            )
        )

        groups = []

        for group in raw_groups:

            groups.append(
                ComplianceGroup(
                    article_number=group.article_number,
                    group_id=group.group_id,
                    condition_logic=group.condition_logic,
                    principle=group.principle,
                    requirement_summary=group.requirement_summary,
                    applicability_condition=group.applicability_condition,
                    obligations=group.obligations,
                    keywords=group.keywords,
                    expected_evidence=group.expected_evidence,
                    assessment_rules=group.assessment_rules,
                )
            )

        logger.info(
            f"Article {article_number}: "
            f"{len(groups)} requirement groups."
        )

        return groups

    # ============================================================
    # GET ONE GROUP
    # ============================================================

    def get_group(
        self,
        article_number: int,
        group_id: str,
    ) -> ComplianceGroup:

        groups = self.get_groups(
            article_number
        )

        for group in groups:

            if group.group_id == group_id:
                return group

        raise KeyError(
            f"Group '{group_id}' "
            f"not found in Article {article_number}."
        )

    # ============================================================
    # BUILD RETRIEVAL QUERY
    # ============================================================

    @staticmethod
    def build_group_query(
        group: ComplianceGroup,
    ) -> str:

        lines = []

        lines.append(
            f"GDPR Article {group.article_number}"
        )

        lines.append(
            f"Requirement Group: {group.group_id}"
        )

        lines.append(
            f"Principle: {group.principle}"
        )

        lines.append(
            f"Requirement: "
            f"{group.requirement_summary}"
        )

        if group.applicability_condition:

            lines.append(
                f"Applicability: "
                f"{group.applicability_condition}"
            )

        if group.keywords:

            lines.append(
                "Keywords: "
                + ", ".join(
                    group.keywords
                )
            )

        lines.append(
            "\nEvidence requirements:"
        )

        for obligation in group.obligations:

            lines.append(
                f"\n{obligation.id}"
            )

            lines.append(
                f"Summary: "
                f"{obligation.plain_summary}"
            )

            lines.append(
                f"Evidence question: "
                f"{obligation.evidence_prompt}"
            )

            if obligation.applicability_condition:

                lines.append(
                    f"Applicability: "
                    f"{obligation.applicability_condition}"
                )

        return "\n".join(lines)

    # ============================================================
    # RETRIEVE ONE GROUP
    # ============================================================

    def retrieve_group(
        self,
        article_number: int,
        group_id: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> GroupEvidence:

        group = self.get_group(
            article_number,
            group_id,
        )

        query = self.build_group_query(
            group
        )

        logger.info(
            f"Retrieving evidence | "
            f"Article={article_number} | "
            f"Group={group_id}"
        )

        evidence = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        return GroupEvidence(
            article_number=article_number,
            group_id=group_id,
            query=query,
            evidence=tuple(
                evidence
            ),
        )

    # ============================================================
    # RETRIEVE COMPLETE ARTICLE
    # ============================================================

    def retrieve_article(
        self,
        article_number: int,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[GroupEvidence]:

        groups = self.get_groups(
            article_number
        )

        results = []

        for group in groups:

            result = self.retrieve_group(
                article_number=article_number,
                group_id=group.group_id,
                top_k=top_k,
                min_score=min_score,
            )

            results.append(
                result
            )

        logger.success(
            f"Article {article_number}: "
            f"retrieved evidence for "
            f"{len(results)} groups."
        )

        return results

    # ============================================================
    # CACHED EMBEDDING RETRIEVAL
    # ============================================================

    def retrieve_group_by_cached_embedding(
        self,
        article_number: int,
        group_id: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> GroupEvidence:

        group = self.get_group(
            article_number,
            group_id,
        )

        query = self.build_group_query(
            group
        )

        cache = self._get_embedding_cache()

        vector = cache.get_embedding(
            article_number,
            group_id,
        )

        response = (
            self.retriever.vector_store
            .similarity_search_by_vector(
                vector,
                namespace=COMPANY_POLICY_NAMESPACE,
                top_k=top_k,
            )
        )

        matches = (
            self.retriever._extract_matches(
                response
            )
        )

        evidence = []

        for match in matches:

            score = float(
                match.get(
                    "score",
                    0.0,
                )
            )

            if (
                min_score is not None
                and score < min_score
            ):
                continue

            metadata = (
                match.get("metadata")
                or {}
            )

            chunk_id = str(
                match.get("id")
                or metadata.get(
                    "chunk_id"
                )
                or ""
            )

            text = str(
                metadata.get(
                    "text"
                )
                or ""
            )

            if not chunk_id:
                continue

            if not text.strip():
                continue

            evidence.append(
                RetrievedEvidence(
                    chunk_id=chunk_id,
                    score=score,
                    text=text,
                    metadata=metadata,
                )
            )

        evidence.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return GroupEvidence(
            article_number=article_number,
            group_id=group_id,
            query=query,
            evidence=tuple(
                evidence[:top_k]
            ),
        )

    # ============================================================
    # EMBEDDING CACHE
    # ============================================================

    def _get_embedding_cache(
        self,
    ):

        if self.embedding_cache is None:

            from app.compliance.gdpr_embeddings import (
                GDPRGroupEmbeddingCache,
            )

            self.embedding_cache = (
                GDPRGroupEmbeddingCache(
                    knowledge_base=self.knowledge_base,
                )
            )

        return self.embedding_cache