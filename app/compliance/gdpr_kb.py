from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_KB_PATH = (
    PROJECT_ROOT
    / "Data"
    / "new_json_gdpr.json"
)


@dataclass(frozen=True)
class SubObligation:
    id: str
    parent_group_id: str
    article_number: int
    legal_text: str
    plain_summary: str
    evidence_prompt: str
    applicability_condition: str | None = None


@dataclass(frozen=True)
class RequirementGroup:
    article_number: int
    group_id: str
    condition_logic: str
    principle: str
    requirement_summary: str
    applicability_condition: str | None
    keywords: tuple[str, ...]
    expected_evidence: tuple[str, ...]
    assessment_rules: dict[str, str]
    obligations: tuple[SubObligation, ...]


@dataclass(frozen=True)
class GDPRArticle:
    article_number: int
    article_name: str
    checkability: str
    contextual_reason: str | None
    requirement_groups: tuple[RequirementGroup, ...]
    source: dict[str, Any]


class GDPRKnowledgeBase:

    def __init__(
        self,
        json_path: str | Path = DEFAULT_KB_PATH,
    ) -> None:

        self.json_path = Path(json_path)

        if not self.json_path.is_absolute():
            self.json_path = PROJECT_ROOT / self.json_path

        if not self.json_path.exists():
            raise FileNotFoundError(
                f"GDPR knowledge base not found:\n"
                f"{self.json_path}"
            )

        self._raw_data = self._load_json()

        self._articles: dict[int, GDPRArticle] = {}

        self._parse()

        print(
            f"GDPR Knowledge Base loaded: "
            f"{len(self._articles)} articles"
        )

    # ============================================================
    # LOAD JSON
    # ============================================================

    def _load_json(self) -> list[dict[str, Any]]:

        with self.json_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        # Expected format:
        #
        # [
        #   {...},
        #   {...}
        # ]

        if isinstance(data, list):
            return data

        # Also support:
        #
        # {
        #   "articles": [...]
        # }

        if isinstance(data, dict):

            if isinstance(
                data.get("articles"),
                list,
            ):
                return data["articles"]

        raise ValueError(
            "Unsupported GDPR JSON structure. "
            "Expected a list of article objects or "
            "{'articles': [...]}."
        )

    # ============================================================
    # PARSE
    # ============================================================

    def _parse(self) -> None:

        for index, item in enumerate(
            self._raw_data
        ):

            if not isinstance(item, dict):
                continue

            article_data = item.get("article")

            # This prevents the KeyError you encountered.
            if not isinstance(
                article_data,
                dict,
            ):
                print(
                    f"WARNING: Skipping JSON item "
                    f"{index}: missing 'article'."
                )
                continue

            number_text = str(
                article_data.get(
                    "number",
                    "",
                )
            )

            article_number = self._extract_article_number(
                number_text
            )

            if article_number is None:
                print(
                    f"WARNING: Could not parse article "
                    f"number from: {number_text}"
                )
                continue

            groups = self._parse_groups(
                article_number,
                item.get(
                    "requirement_groups",
                    [],
                ),
            )

            article = GDPRArticle(
                article_number=article_number,
                article_name=str(
                    article_data.get(
                        "title",
                        "",
                    )
                ),
                checkability=str(
                    article_data.get(
                        "checkability",
                        "direct",
                    )
                ),
                contextual_reason=article_data.get(
                    "contextual_reason"
                ),
                requirement_groups=tuple(groups),
                source=item.get(
                    "source",
                    {},
                ),
            )

            self._articles[
                article_number
            ] = article

    # ============================================================
    # PARSE GROUPS
    # ============================================================

    def _parse_groups(
        self,
        article_number: int,
        raw_groups: Any,
    ) -> list[RequirementGroup]:

        if not isinstance(
            raw_groups,
            list,
        ):
            return []

        groups = []

        for raw_group in raw_groups:

            if not isinstance(
                raw_group,
                dict,
            ):
                continue

            group_id = str(
                raw_group.get(
                    "group_id",
                    "",
                )
            )

            obligations = []

            raw_obligations = raw_group.get(
                "sub_obligations",
                [],
            )

            if isinstance(
                raw_obligations,
                list,
            ):

                for raw_obligation in raw_obligations:

                    if not isinstance(
                        raw_obligation,
                        dict,
                    ):
                        continue

                    obligation = SubObligation(
                        id=str(
                            raw_obligation.get(
                                "id",
                                "",
                            )
                        ),
                        parent_group_id=group_id,
                        article_number=article_number,
                        legal_text=str(
                            raw_obligation.get(
                                "legal_text",
                                "",
                            )
                        ),
                        plain_summary=str(
                            raw_obligation.get(
                                "plain_summary",
                                "",
                            )
                        ),
                        evidence_prompt=str(
                            raw_obligation.get(
                                "evidence_prompt",
                                "",
                            )
                        ),
                        applicability_condition=raw_group.get(
                            "applicability_condition"
                        ),
                    )

                    obligations.append(
                        obligation
                    )

            group = RequirementGroup(
                article_number=article_number,
                group_id=group_id,
                condition_logic=str(
                    raw_group.get(
                        "condition_logic",
                        "SINGLE",
                    )
                ),
                principle=str(
                    raw_group.get(
                        "principle",
                        "",
                    )
                ),
                requirement_summary=str(
                    raw_group.get(
                        "requirement_summary",
                        "",
                    )
                ),
                applicability_condition=raw_group.get(
                    "applicability_condition"
                ),
                keywords=tuple(
                    raw_group.get(
                        "keywords",
                        [],
                    )
                    or []
                ),
                expected_evidence=tuple(
                    raw_group.get(
                        "expected_evidence",
                        [],
                    )
                    or []
                ),
                assessment_rules=dict(
                    raw_group.get(
                        "assessment_rules",
                        {},
                    )
                    or {}
                ),
                obligations=tuple(
                    obligations
                ),
            )

            groups.append(group)

        return groups

    # ============================================================
    # ARTICLE NUMBER
    # ============================================================

    @staticmethod
    def _extract_article_number(
        value: str,
    ) -> int | None:

        # "Article 5" -> 5

        parts = value.strip().split()

        if len(parts) < 2:
            return None

        try:
            return int(parts[-1])
        except ValueError:
            return None

    # ============================================================
    # PUBLIC API
    # ============================================================

    def get_article(
        self,
        article_number: int,
    ) -> GDPRArticle:

        if article_number not in self._articles:
            raise KeyError(
                f"Article {article_number} "
                f"not found in GDPR knowledge base."
            )

        return self._articles[
            article_number
        ]

    def get_all_articles(
        self,
    ) -> list[GDPRArticle]:

        return list(
            self._articles.values()
        )

    def get_sub_obligations(
        self,
        article_number: int,
    ) -> list[SubObligation]:

        article = self.get_article(
            article_number
        )

        result = []

        for group in article.requirement_groups:
            result.extend(
                group.obligations
            )

        return result

    def get_groups(
        self,
        article_number: int,
    ) -> list[RequirementGroup]:

        return list(
            self.get_article(
                article_number
            ).requirement_groups
        )

    def article_count(self) -> int:
        return len(self._articles)

    def group_count(self) -> int:

        return sum(
            len(
                article.requirement_groups
            )
            for article in self._articles.values()
        )

    def obligation_count(self) -> int:

        return sum(
            len(group.obligations)
            for article in self._articles.values()
            for group in article.requirement_groups
        )