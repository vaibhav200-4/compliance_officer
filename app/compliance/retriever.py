from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.logger import get_logger
from app.vectorstore.pinecone import (
    COMPANY_POLICY_NAMESPACE,
    PineconeManager,
)

logger = get_logger()


@dataclass(frozen=True)
class RetrievedEvidence:
    """
    Represents one policy chunk retrieved from Pinecone.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class ComplianceRetriever:
    """
    Retrieves company-policy evidence for GDPR compliance obligations.

    This class is intentionally thin:
        - It does not generate embeddings itself.
        - It does not connect to Pinecone directly.
        - It does not judge compliance.

    It reuses the existing PineconeManager and Embedder.
    """

    def __init__(
        self,
        vector_store: PineconeManager | None = None,
    ) -> None:

        self.vector_store = vector_store or PineconeManager()

        logger.success(
            "Compliance retriever initialized."
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[RetrievedEvidence]:
        """
        Retrieve relevant company-policy chunks for a query.

        Parameters
        ----------
        query:
            Usually the GDPR sub-obligation evidence_prompt.

        top_k:
            Maximum number of Pinecone matches.

        min_score:
            Optional similarity threshold.

        Returns
        -------
        list[RetrievedEvidence]
            Normalized policy evidence sorted by score.
        """

        if not query or not query.strip():
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        logger.info(
            f"Retrieving policy evidence | "
            f"top_k={top_k} | query='{query[:120]}'"
        )

        # -----------------------------------------------------
        # Query Pinecone
        # -----------------------------------------------------

        response = self.vector_store.similarity_search(
            query=query,
            namespace=COMPANY_POLICY_NAMESPACE,
            top_k=top_k,
        )

        # -----------------------------------------------------
        # Extract matches
        # -----------------------------------------------------

        matches = self._extract_matches(response)

        results: list[RetrievedEvidence] = []

        # -----------------------------------------------------
        # Normalize results
        # -----------------------------------------------------

        for match in matches:

            score = float(
                match.get("score", 0.0)
            )

            # Optional similarity threshold
            if (
                min_score is not None
                and score < min_score
            ):
                continue

            metadata = match.get("metadata") or {}

            chunk_id = str(
                match.get("id")
                or metadata.get("chunk_id")
                or ""
            )

            text = str(
                metadata.get("text")
                or ""
            )

            # -------------------------------------------------
            # Validate chunk ID
            # -------------------------------------------------

            if not chunk_id:
                logger.warning(
                    "Skipping retrieved match without chunk_id."
                )
                continue

            # -------------------------------------------------
            # Validate text
            # -------------------------------------------------

            if not text.strip():
                logger.warning(
                    f"Skipping empty retrieved chunk: {chunk_id}"
                )
                continue

            # -------------------------------------------------
            # Create normalized evidence object
            # -------------------------------------------------

            results.append(
                RetrievedEvidence(
                    chunk_id=chunk_id,
                    score=score,
                    text=text,
                    metadata=metadata,
                )
            )

        # -----------------------------------------------------
        # Sort highest similarity first
        # -----------------------------------------------------

        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        logger.info(
            f"Retrieved {len(results)} usable evidence chunks."
        )

        # IMPORTANT:
        # list.sort() returns None.
        # Therefore sorting and returning must be separate.
        return results

    @staticmethod
    def _extract_matches(
        response: Any,
    ) -> list[dict[str, Any]]:
        """
        Extract Pinecone matches from the response.

        Pinecone SDK responses may expose matches either as
        dictionary-like objects or model objects.
        """

        if response is None:
            return []

        # -----------------------------------------------------
        # Dictionary-style response
        # -----------------------------------------------------

        if isinstance(response, dict):

            matches = response.get(
                "matches",
                [],
            )

            return [
                ComplianceRetriever._to_dict(match)
                for match in matches
            ]

        # -----------------------------------------------------
        # Pinecone SDK response object
        # -----------------------------------------------------

        matches = getattr(
            response,
            "matches",
            [],
        )

        return [
            ComplianceRetriever._to_dict(match)
            for match in matches
        ]

    @staticmethod
    def _to_dict(
        match: Any,
    ) -> dict[str, Any]:
        """
        Normalize a Pinecone match into a dictionary.
        """

        # Already a dictionary
        if isinstance(match, dict):
            return match

        result: dict[str, Any] = {}

        # -----------------------------------------------------
        # Match ID
        # -----------------------------------------------------

        match_id = getattr(
            match,
            "id",
            None,
        )

        if match_id is not None:
            result["id"] = match_id

        # -----------------------------------------------------
        # Similarity score
        # -----------------------------------------------------

        score = getattr(
            match,
            "score",
            None,
        )

        if score is not None:
            result["score"] = score

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        metadata = getattr(
            match,
            "metadata",
            None,
        )

        if metadata is not None:
            result["metadata"] = dict(metadata)

        return result