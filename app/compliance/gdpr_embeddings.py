from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.compliance.gdpr_kb import GDPRKnowledgeBase
from app.compliance.group_retriever import ComplianceGroupRetriever
from app.ingestion.embedder import Embedder
from app.core.logger import get_logger

logger = get_logger()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "gdpr"
    / "gdpr_group_embeddings.json"
)


class GDPRGroupEmbeddingCache:
    """
    Generates and stores one embedding for every GDPR parent group.

    Example:

        Article 5
            5.1.a -> embedding
            5.1.b -> embedding
            5.1.c -> embedding
            5.1.d -> embedding
            5.1.e -> embedding
            5.1.f -> embedding
            5.2   -> embedding

    These vectors are used as query vectors against the
    company-policy Pinecone namespace.

    The GDPR embeddings themselves are NOT stored in the
    company-policy namespace.
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        knowledge_base: GDPRKnowledgeBase | None = None,
        embedder: Embedder | None = None,
        cache_path: str | Path = DEFAULT_CACHE_PATH,
    ) -> None:

        self.knowledge_base = (
            knowledge_base
            or GDPRKnowledgeBase()
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

        self.cache_path = Path(cache_path)

        if not self.cache_path.is_absolute():
            self.cache_path = PROJECT_ROOT / self.cache_path

        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Query key
    # ------------------------------------------------------------------

    @staticmethod
    def _group_key(
        article_number: int,
        group_id: str,
    ) -> str:
        """
        Create a unique key for a GDPR group.
        """

        return f"{article_number}:{group_id}"

    # ------------------------------------------------------------------
    # Build embedding records
    # ------------------------------------------------------------------

    def build_records(
        self,
    ) -> list[dict[str, Any]]:
        """
        Build embedding records for every GDPR parent group.
        """

        records: list[dict[str, Any]] = []

        articles = self.knowledge_base.get_all_articles()

        for article in articles:

            groups = self.group_retriever.get_groups(
                article.article_number
            )

            for group in groups:

                query = (
                    self.group_retriever.build_group_query(
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
                        "article_number": article.article_number,
                        "article_name": article.article_name,
                        "group_id": group.group_id,
                        "query": query,
                    }
                )

        return records

    # ------------------------------------------------------------------
    # Generate embeddings
    # ------------------------------------------------------------------

    def generate(
    self,
    *,
    force: bool = False,
    batch_size: int = 50,
    ) -> dict[str, Any]:
        """
        Generate GDPR group embeddings incrementally.

        Successful batches are saved immediately so that a quota
        failure or network failure does not destroy previous work.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if batch_size > 100:
            raise ValueError(
                "batch_size cannot exceed Google's maximum of 100."
            )

        # ---------------------------------------------------------
        # Build all GDPR group records
        # ---------------------------------------------------------

        records = self.build_records()

        if not records:
            raise ValueError(
                "No GDPR group records were found."
            )

        total_groups = len(records)

        logger.info(
            f"Found {total_groups} GDPR groups."
        )

        # ---------------------------------------------------------
        # Load existing cache if available
        # ---------------------------------------------------------

        if self.cache_path.exists() and not force:

            logger.info(
                f"Existing GDPR embedding cache found: "
                f"{self.cache_path}"
            )

            self.load()

        else:

            self._cache = {
                "version": self.CACHE_VERSION,
                "embedding_model": (
                    "configured-google-embedding-model"
                ),
                "embedding_dimension": 0,
                "group_count": 0,
                "groups": {},
            }

        cached_groups = self._cache.setdefault(
            "groups",
            {},
        )

        logger.info(
            f"Already cached: "
            f"{len(cached_groups)}/{total_groups} groups."
        )

        # ---------------------------------------------------------
        # Determine groups still requiring embeddings
        # ---------------------------------------------------------

        pending_records = [
            record
            for record in records
            if record["key"] not in cached_groups
        ]

        if not pending_records:

            logger.success(
                "All GDPR group embeddings are already cached."
            )

            self._cache["group_count"] = len(
                cached_groups
            )

            self.save()

            return self._cache

        logger.info(
            f"Remaining groups to embed: "
            f"{len(pending_records)}"
        )

        # ---------------------------------------------------------
        # Process pending groups in batches
        # ---------------------------------------------------------

        total_pending = len(pending_records)

        total_batches = (
            total_pending + batch_size - 1
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

            batch_records = pending_records[
                start:end
            ]

            batch_queries = [
                record["query"]
                for record in batch_records
            ]

            logger.info(
                f"Embedding batch "
                f"{batch_number}/{total_batches} | "
                f"{len(batch_records)} groups"
            )

            try:

                vectors = self.embedder.embed_documents(
                    batch_queries
                )

            except Exception as exc:

                logger.error(
                    f"Embedding batch "
                    f"{batch_number}/{total_batches} failed: "
                    f"{exc}"
                )

                logger.warning(
                    "Previously completed batches have already "
                    "been saved and can be resumed later."
                )

                raise

            # -----------------------------------------------------
            # Validate response
            # -----------------------------------------------------

            if len(vectors) != len(batch_records):

                raise RuntimeError(
                    f"Embedding response count mismatch. "
                    f"Expected={len(batch_records)}, "
                    f"Received={len(vectors)}."
                )

            # -----------------------------------------------------
            # Add batch to cache
            # -----------------------------------------------------

            for record, vector in zip(
                batch_records,
                vectors,
            ):

                cached_groups[record["key"]] = {
                    "article_number": record[
                        "article_number"
                    ],
                    "article_name": record[
                        "article_name"
                    ],
                    "group_id": record[
                        "group_id"
                    ],
                    "query": record["query"],
                    "embedding": vector,
                }

            # -----------------------------------------------------
            # Update metadata
            # -----------------------------------------------------

            self._cache[
                "embedding_dimension"
            ] = len(vectors[0])

            self._cache[
                "group_count"
            ] = len(cached_groups)

            # -----------------------------------------------------
            # SAVE IMMEDIATELY
            # -----------------------------------------------------

            self.save()

            logger.success(
                f"Saved batch {batch_number}/{total_batches}. "
                f"Progress: "
                f"{len(cached_groups)}/{total_groups}"
            )

        # ---------------------------------------------------------
        # Finished
        # ---------------------------------------------------------

        self._cache[
            "group_count"
        ] = len(cached_groups)

        self.save()

        logger.success(
            f"GDPR embedding generation completed. "
            f"Cached {len(cached_groups)} groups."
        )

        return self._cache
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Save the embedding cache to disk.
        """

        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.cache_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self._cache,
                file,
                indent=2,
            )

        logger.success(
            f"GDPR embedding cache saved to "
            f"{self.cache_path}"
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """
        Load an existing embedding cache.
        """

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"GDPR embedding cache not found: "
                f"{self.cache_path}"
            )

        with self.cache_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self._cache = json.load(file)

        if "groups" not in self._cache:
            raise ValueError(
                "Invalid GDPR embedding cache: "
                "'groups' field is missing."
            )

        logger.success(
            f"Loaded "
            f"{len(self._cache['groups'])} "
            f"GDPR group embeddings."
        )

        return self._cache

    # ------------------------------------------------------------------
    # Get one embedding
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        article_number: int,
        group_id: str,
    ) -> list[float]:
        """
        Return the cached embedding for one GDPR group.

        Example:

            cache.get_embedding(5, "5.1.f")
        """

        if not self._cache:
            self.load()

        key = self._group_key(
            article_number,
            group_id,
        )

        group = self._cache["groups"].get(
            key
        )

        if group is None:
            raise KeyError(
                f"No cached embedding found for "
                f"Article {article_number}, "
                f"group {group_id}."
            )

        return group["embedding"]

    # ------------------------------------------------------------------
    # Get group record
    # ------------------------------------------------------------------

    def get_group_record(
        self,
        article_number: int,
        group_id: str,
    ) -> dict[str, Any]:
        """
        Return complete cached information for a group.
        """

        if not self._cache:
            self.load()

        key = self._group_key(
            article_number,
            group_id,
        )

        group = self._cache["groups"].get(
            key
        )

        if group is None:
            raise KeyError(
                f"No cached GDPR group found for "
                f"Article {article_number}, "
                f"group {group_id}."
            )

        return group

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def group_count(self) -> int:
        """
        Return number of cached GDPR groups.
        """

        if not self._cache:
            return 0

        return len(
            self._cache.get(
                "groups",
                {},
            )
        )



if __name__ == "__main__":
    print("=" * 70)
    print("GDPR GROUP EMBEDDING GENERATOR")
    print("=" * 70)

    print("\n[1] Initializing embedding cache...")

    cache = GDPRGroupEmbeddingCache()

    print("SUCCESS")

    print("\n[2] Generating GDPR group embeddings...")
    print("-" * 70)

    data = cache.generate(force=False)

    print("\n[3] Generation completed")
    print("-" * 70)

    print(f"Groups generated     : {data['group_count']}")
    print(f"Embedding dimension  : {data['embedding_dimension']}")
    print(f"Cache file           : {cache.cache_path}")

    print("\n[4] Testing Article 5")
    print("-" * 70)

    article_5_groups = [
        "5.1.a",
        "5.1.b",
        "5.1.c",
        "5.1.d",
        "5.1.e",
        "5.1.f",
        "5.2",
    ]

    for group_id in article_5_groups:

        vector = cache.get_embedding(
            article_number=5,
            group_id=group_id,
        )

        print(
            f"{group_id:<10} "
            f"dimension={len(vector)}"
        )

    print("\n" + "=" * 70)
    print("GDPR EMBEDDING GENERATION SUCCESSFUL")
    print("=" * 70)

