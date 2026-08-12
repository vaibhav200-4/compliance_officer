from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GDPR_PATH = (
    PROJECT_ROOT
    / "data"
    / "gdpr_articles.json"
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubObligation:
    """
    Represents one atomic GDPR compliance requirement.
    """

    id: str
    parent_group_id: str
    condition_logic: str
    legal_text: str
    plain_summary: str
    applicability_condition: str | None
    evidence_prompt: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubObligation":
        """
        Create a SubObligation from a JSON dictionary.
        """

        required_fields = [
            "id",
            "parent_group_id",
            "condition_logic",
            "legal_text",
            "plain_summary",
            "evidence_prompt",
        ]

        missing = [
            field
            for field in required_fields
            if field not in data
        ]

        if missing:
            raise ValueError(
                f"Sub-obligation is missing required fields: {missing}"
            )

        return cls(
            id=str(data["id"]),
            parent_group_id=str(data["parent_group_id"]),
            condition_logic=str(data["condition_logic"]),
            legal_text=str(data["legal_text"]),
            plain_summary=str(data["plain_summary"]),
            applicability_condition=(
                str(data["applicability_condition"])
                if data.get("applicability_condition") is not None
                else None
            ),
            evidence_prompt=str(data["evidence_prompt"]),
        )


@dataclass(frozen=True)
class GDPRArticle:
    """
    Represents one GDPR article and its decomposed obligations.
    """

    article_number: int
    article_name: str
    checkability: str
    contextual_reason: str | None
    sub_obligations: tuple[SubObligation, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GDPRArticle":
        """
        Create a GDPRArticle from a JSON dictionary.
        """

        required_fields = [
            "article_number",
            "article_name",
            "checkability",
            "sub_obligations",
        ]

        missing = [
            field
            for field in required_fields
            if field not in data
        ]

        if missing:
            raise ValueError(
                f"Article is missing required fields: {missing}"
            )

        sub_obligations = tuple(
            SubObligation.from_dict(item)
            for item in data["sub_obligations"]
        )

        return cls(
            article_number=int(data["article_number"]),
            article_name=str(data["article_name"]),
            checkability=str(data["checkability"]),
            contextual_reason=(
                str(data["contextual_reason"])
                if data.get("contextual_reason") is not None
                else None
            ),
            sub_obligations=sub_obligations,
        )


# ---------------------------------------------------------------------------
# GDPR Knowledge Base
# ---------------------------------------------------------------------------

class GDPRKnowledgeBase:
    """
    Loads and provides access to the pre-decomposed GDPR knowledge base.

    Responsibilities:
        - Load GDPR JSON
        - Validate article structure
        - Store articles in memory
        - Retrieve articles
        - Retrieve sub-obligations
        - Provide simple search helpers

    This class does NOT:
        - Generate embeddings
        - Call Pinecone
        - Call an LLM
        - Judge compliance
    """

    def __init__(
        self,
        json_path: str | Path = DEFAULT_GDPR_PATH,
    ) -> None:

        self.json_path = Path(json_path)

        if not self.json_path.is_absolute():
            self.json_path = PROJECT_ROOT / self.json_path

        self._articles: dict[int, GDPRArticle] = {}

        self._load()

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """
        Load and parse the GDPR JSON file.
        """

        if not self.json_path.exists():
            raise FileNotFoundError(
                f"GDPR knowledge base not found: {self.json_path}"
            )

        try:
            with self.json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                raw_data = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in GDPR knowledge base: "
                f"{self.json_path}"
            ) from exc

        article_data = self._normalize_root(raw_data)

        for item in article_data:
            article = GDPRArticle.from_dict(item)

            if article.article_number in self._articles:
                raise ValueError(
                    f"Duplicate GDPR article found: "
                    f"{article.article_number}"
                )

            self._articles[article.article_number] = article

    # -----------------------------------------------------------------------
    # Root normalization
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_root(
        raw_data: Any,
    ) -> list[dict[str, Any]]:
        """
        Normalize supported JSON structures.

        Supported:

        1. List of articles:
           [
               {...},
               {...}
           ]

        2. Object containing articles:
           {
               "articles": [
                   {...},
                   {...}
               ]
           }

        3. A single article:
           {
               "article_number": 5,
               ...
           }
        """

        if isinstance(raw_data, list):
            return raw_data

        if isinstance(raw_data, dict):

            if "articles" in raw_data:
                articles = raw_data["articles"]

                if not isinstance(articles, list):
                    raise ValueError(
                        "'articles' must contain a list"
                    )

                return articles

            if "article_number" in raw_data:
                return [raw_data]

        raise ValueError(
            "Unsupported GDPR knowledge base format. "
            "Expected a list of articles, an object containing "
            "'articles', or a single article object."
        )

    # -----------------------------------------------------------------------
    # Article Access
    # -----------------------------------------------------------------------

    def get_article(
        self,
        article_number: int,
    ) -> GDPRArticle | None:
        """
        Return a GDPR article by article number.

        Example:
            article = kb.get_article(5)
        """

        return self._articles.get(int(article_number))

    def require_article(
        self,
        article_number: int,
    ) -> GDPRArticle:
        """
        Return an article or raise an explicit error.
        """

        article = self.get_article(article_number)

        if article is None:
            raise KeyError(
                f"GDPR Article {article_number} not found "
                f"in knowledge base."
            )

        return article

    def get_all_articles(self) -> list[GDPRArticle]:
        """
        Return all loaded GDPR articles ordered by article number.
        """

        return [
            self._articles[number]
            for number in sorted(self._articles)
        ]

    # -----------------------------------------------------------------------
    # Sub-obligation Access
    # -----------------------------------------------------------------------

    def get_sub_obligations(
        self,
        article_number: int,
    ) -> list[SubObligation]:
        """
        Return all sub-obligations for an article.

        Example:
            obligations = kb.get_sub_obligations(5)
        """

        article = self.require_article(article_number)

        return list(article.sub_obligations)

    def get_sub_obligation(
        self,
        obligation_id: str,
    ) -> SubObligation | None:
        """
        Find a specific sub-obligation across the entire knowledge base.

        Example:
            obligation = kb.get_sub_obligation("5.1.e.1")
        """

        for article in self._articles.values():

            for obligation in article.sub_obligations:

                if obligation.id == obligation_id:
                    return obligation

        return None

    def require_sub_obligation(
        self,
        obligation_id: str,
    ) -> SubObligation:
        """
        Return a sub-obligation or raise an explicit error.
        """

        obligation = self.get_sub_obligation(obligation_id)

        if obligation is None:
            raise KeyError(
                f"GDPR sub-obligation '{obligation_id}' "
                f"not found in knowledge base."
            )

        return obligation

    # -----------------------------------------------------------------------
    # Group Access
    # -----------------------------------------------------------------------

    def get_group_obligations(
        self,
        article_number: int,
        parent_group_id: str,
    ) -> list[SubObligation]:
        """
        Return all obligations belonging to a parent group.

        Example:

            kb.get_group_obligations(5, "5.1")
        """

        obligations = self.get_sub_obligations(article_number)

        return [
            obligation
            for obligation in obligations
            if obligation.parent_group_id == parent_group_id
        ]

    # -----------------------------------------------------------------------
    # Applicability Helpers
    # -----------------------------------------------------------------------

    def get_conditional_obligations(
        self,
        article_number: int,
    ) -> list[SubObligation]:
        """
        Return obligations having an applicability condition.
        """

        obligations = self.get_sub_obligations(article_number)

        return [
            obligation
            for obligation in obligations
            if obligation.applicability_condition is not None
        ]

    def get_unconditional_obligations(
        self,
        article_number: int,
    ) -> list[SubObligation]:
        """
        Return obligations without an applicability condition.
        """

        obligations = self.get_sub_obligations(article_number)

        return [
            obligation
            for obligation in obligations
            if obligation.applicability_condition is None
        ]

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search_obligations(
        self,
        keyword: str,
    ) -> list[SubObligation]:
        """
        Simple keyword search across obligation text.

        This is NOT semantic search.

        It is only a helper for debugging/testing the knowledge base.
        """

        keyword = keyword.lower().strip()

        if not keyword:
            return []

        results: list[SubObligation] = []

        for article in self._articles.values():

            for obligation in article.sub_obligations:

                searchable_text = " ".join(
                    [
                        obligation.id,
                        obligation.legal_text,
                        obligation.plain_summary,
                        obligation.evidence_prompt,
                    ]
                ).lower()

                if keyword in searchable_text:
                    results.append(obligation)

        return results

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    @property
    def article_count(self) -> int:
        """Number of loaded GDPR articles."""

        return len(self._articles)

    @property
    def obligation_count(self) -> int:
        """Total number of loaded sub-obligations."""

        return sum(
            len(article.sub_obligations)
            for article in self._articles.values()
        )

    # -----------------------------------------------------------------------
    # Debugging
    # -----------------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """
        Return basic knowledge-base statistics.
        """

        return {
            "articles": self.article_count,
            "sub_obligations": self.obligation_count,
            "conditional_obligations": sum(
                len(self.get_conditional_obligations(article_number))
                for article_number in self._articles
            ),
            "unconditional_obligations": sum(
                len(self.get_unconditional_obligations(article_number))
                for article_number in self._articles
            ),
        }