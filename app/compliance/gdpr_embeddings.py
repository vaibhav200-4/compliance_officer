from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.compliance.gdpr_kb import (
    GDPRKnowledgeBase,
)

from app.compliance.group_retriever import (
    ComplianceGroupRetriever,
)

from app.ingestion.embedder import (
    Embedder,
)

from app.core.logger import (
    get_logger,
)


logger = get_logger()


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_KB_PATH = (
    PROJECT_ROOT
    / "Data"
    / "new_json_gdpr.json"
)


DEFAULT_CACHE_PATH = (
    PROJECT_ROOT
    / "Data"
    / "gdpr"
    / "gdpr_group_embeddings.json"
)


class GDPRGroupEmbeddingCache:

    CACHE_VERSION = 2

    def __init__(
        self,
        knowledge_base: GDPRKnowledgeBase | None = None,
        embedder: Embedder | None = None,
        cache_path: str | Path = DEFAULT_CACHE_PATH,
    ) -> None:

        self.knowledge_base = (
            knowledge_base
            or GDPRKnowledgeBase(
                DEFAULT_KB_PATH
            )
        )

        self.embedder = (
            embedder
            or Embedder()
        )

        self.group_retriever = (
            ComplianceGroupRetriever(
                knowledge_base=self.knowledge_base,
            )
        )

        self.cache_path = Path(
            cache_path
        )

        if not self.cache_path.is_absolute():

            self.cache_path = (
                PROJECT_ROOT
                / self.cache_path
            )

        self._cache: dict[str, Any] = {}

    # ============================================================
    # KEY
    # ============================================================

    @staticmethod
    def _group_key(
        article_number: int,
        group_id: str,
    ) -> str:

        return (
            f"{article_number}:{group_id}"
        )

    # ============================================================
    # BUILD RECORDS
    # ============================================================

    def build_records(
        self,
    ) -> list[dict[str, Any]]:

        records = []

        articles = (
            self.knowledge_base
            .get_all_articles()
        )

        for article in articles:

            if (
                article.checkability
                == "not_applicable"
            ):
                continue

            groups = (
                self.group_retriever
                .get_groups(
                    article.article_number
                )
            )

            for group in groups:

                query = (
                    self.group_retriever
                    .build_group_query(
                        group
                    )
                )

                key = self._group_key(
                    article.article_number,
                    group.group_id,
                )

                records.append(
                    {
                        "key": key,
                        "article_number": (
                            article.article_number
                        ),
                        "article_name": (
                            article.article_name
                        ),
                        "group_id": (
                            group.group_id
                        ),
                        "query": query,
                    }
                )

        return records

    # ============================================================
    # GENERATE
    # ============================================================

    def generate(
        self,
        *,
        force: bool = False,
        batch_size: int = 50,
    ) -> dict[str, Any]:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        records = self.build_records()

        if not records:
            raise ValueError(
                "No GDPR requirement groups found."
            )

        logger.info(
            f"Found {len(records)} GDPR "
            f"requirement groups."
        )

        # --------------------------------------------------------
        # LOAD / RESET CACHE
        # --------------------------------------------------------

        if (
            self.cache_path.exists()
            and not force
        ):

            self.load()

        else:

            self._cache = {
                "version": self.CACHE_VERSION,
                "kb_path": str(
                    self.knowledge_base.json_path
                ),
                "embedding_dimension": 0,
                "group_count": 0,
                "groups": {},
            }

        cached_groups = (
            self._cache.setdefault(
                "groups",
                {},
            )
        )

        pending_records = [
            record
            for record in records
            if record["key"]
            not in cached_groups
        ]

        logger.info(
            f"Already cached: "
            f"{len(cached_groups)}"
        )

        logger.info(
            f"Remaining: "
            f"{len(pending_records)}"
        )

        if not pending_records:

            logger.success(
                "All GDPR group embeddings "
                "already exist."
            )

            return self._cache

        # --------------------------------------------------------
        # BATCH
        # --------------------------------------------------------

        total_pending = len(
            pending_records
        )

        total_batches = (
            total_pending
            + batch_size
            - 1
        ) // batch_size

        for batch_number, start in enumerate(
            range(
                0,
                total_pending,
                batch_size,
            ),
            start=1,
        ):

            end = min(
                start + batch_size,
                total_pending,
            )

            batch_records = (
                pending_records[
                    start:end
                ]
            )

            queries = [
                record["query"]
                for record in batch_records
            ]

            logger.info(
                f"Embedding batch "
                f"{batch_number}/"
                f"{total_batches} | "
                f"{len(queries)} groups"
            )

            vectors = (
                self.embedder
                .embed_documents(
                    queries
                )
            )

            if len(vectors) != len(
                batch_records
            ):

                raise RuntimeError(
                    "Embedding response "
                    "count mismatch. "
                    f"Expected="
                    f"{len(batch_records)}, "
                    f"Received="
                    f"{len(vectors)}"
                )

            # ----------------------------------------------------
            # SAVE VECTORS
            # ----------------------------------------------------

            for record, vector in zip(
                batch_records,
                vectors,
            ):

                cached_groups[
                    record["key"]
                ] = {
                    "article_number": (
                        record[
                            "article_number"
                        ]
                    ),
                    "article_name": (
                        record[
                            "article_name"
                        ]
                    ),
                    "group_id": (
                        record[
                            "group_id"
                        ]
                    ),
                    "query": (
                        record["query"]
                    ),
                    "embedding": vector,
                }

            self._cache[
                "embedding_dimension"
            ] = len(vectors[0])

            self._cache[
                "group_count"
            ] = len(cached_groups)

            # Save after EVERY batch
            self.save()

            logger.success(
                f"Saved batch "
                f"{batch_number}/"
                f"{total_batches} | "
                f"Progress="
                f"{len(cached_groups)}/"
                f"{len(records)}"
            )

        return self._cache

    # ============================================================
    # SAVE
    # ============================================================

    def save(self) -> None:

        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.cache_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self._cache,
                f,
                indent=2,
                ensure_ascii=False,
            )

    # ============================================================
    # LOAD
    # ============================================================

    def load(
        self,
    ) -> dict[str, Any]:

        if not self.cache_path.exists():

            raise FileNotFoundError(
                f"GDPR embedding cache "
                f"not found:\n"
                f"{self.cache_path}"
            )

        with self.cache_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            self._cache = json.load(f)

        if "groups" not in self._cache:

            raise ValueError(
                "Invalid GDPR embedding cache."
            )

        return self._cache

    # ============================================================
    # GET EMBEDDING
    # ============================================================

    def get_embedding(
        self,
        article_number: int,
        group_id: str,
    ) -> list[float]:

        if not self._cache:
            self.load()

        key = self._group_key(
            article_number,
            group_id,
        )

        group = (
            self._cache[
                "groups"
            ].get(key)
        )

        if group is None:

            raise KeyError(
                f"No embedding found for "
                f"Article {article_number}, "
                f"group {group_id}."
            )

        return group["embedding"]

    # ============================================================
    # GET RECORD
    # ============================================================

    def get_group_record(
        self,
        article_number: int,
        group_id: str,
    ) -> dict[str, Any]:

        if not self._cache:
            self.load()

        key = self._group_key(
            article_number,
            group_id,
        )

        group = (
            self._cache[
                "groups"
            ].get(key)
        )

        if group is None:

            raise KeyError(
                f"No group found for "
                f"Article {article_number}, "
                f"group {group_id}."
            )

        return group

    # ============================================================
    # STATS
    # ============================================================

    @property
    def group_count(self) -> int:

        return len(
            self._cache.get(
                "groups",
                {},
            )
        )


# ================================================================
# DIRECT TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GDPR GROUP EMBEDDING TEST")
    print("=" * 70)

    kb = GDPRKnowledgeBase()

    print(
        f"\nArticles     : "
        f"{kb.article_count()}"
    )

    print(
        f"Groups       : "
        f"{kb.group_count()}"
    )

    print(
        f"Sub-obligations : "
        f"{kb.obligation_count()}"
    )

    print("\nArticle 5 groups:")

    for group in kb.get_groups(5):

        print(
            f"  {group.group_id:<10} "
            f"{group.principle}"
        )

    print("\nSUCCESS")