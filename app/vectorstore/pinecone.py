"""Pinecone storage for company-policy document chunks."""
# app/vectorstore/pinecone.py
from __future__ import annotations

from langchain_core.documents import Document
from pinecone import Pinecone

from app.core.config import settings
from app.core.logger import get_logger
from app.ingestion.embedder import Embedder

logger = get_logger()

COMPANY_POLICY_NAMESPACE = "company-policy"

class PineconeManager:
    """Store documents in Pinecone using the shared application Embedder."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")
        if not settings.PINECONE_INDEX_NAME:
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set.")

        self.embedder = embedder or Embedder()
        self.client = Pinecone(api_key=settings.PINECONE_API_KEY)

        if settings.PINECONE_INDEX_NAME not in self.client.list_indexes().names():
            raise ValueError(
                f"Pinecone index does not exist: {settings.PINECONE_INDEX_NAME}"
            )

        self.index_description = self.client.describe_index(settings.PINECONE_INDEX_NAME)
        self.index_dimension = self.index_description.dimension
        self.index = self.client.Index(settings.PINECONE_INDEX_NAME)
        logger.success(f"Connected to Pinecone index '{settings.PINECONE_INDEX_NAME}'.")

    def add_documents(
        self,
        documents: list[Document],
        *,
        namespace: str = COMPANY_POLICY_NAMESPACE,
    ) -> list[str]:
        """Embed and upsert non-empty documents into an explicit namespace."""
        if not documents:
            return []
        if namespace != COMPANY_POLICY_NAMESPACE:
            raise ValueError("Company-policy ingestion must use namespace 'company-policy'.")

        usable_documents = [doc for doc in documents if doc.page_content.strip()]
        if not usable_documents:
            raise ValueError("No non-empty chunks are available for embedding.")

        texts = [document.page_content for document in usable_documents]
        vectors = self.embedder.embed_documents(texts)
        if len(vectors) != len(usable_documents):
            raise RuntimeError("Embedding response count does not match chunk count.")
        if any(len(vector) != self.index_dimension for vector in vectors):
            raise ValueError(
                f"Embedding dimension does not match Pinecone index dimension "
                f"({self.index_dimension})."
            )

        vector_ids: list[str] = []
        records = []
        for document, values in zip(usable_documents, vectors):
            chunk_id = document.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError("Chunk is missing required metadata: chunk_id")
            vector_ids.append(chunk_id)
            records.append(
                {
                    "id": chunk_id,
                    "values": values,
                    "metadata": {
                        **document.metadata,
                        "text": document.page_content,
                    },
                }
            )

        self.index.upsert(vectors=records, namespace=namespace)
        logger.success(f"Upserted {len(records)} vectors to namespace '{namespace}'.")
        return vector_ids

    def describe_index_stats(self):
        """Return Pinecone index statistics for connection verification."""
        return self.index.describe_index_stats()

    def similarity_search(
        self,
        query: str,
        *,
        namespace: str = COMPANY_POLICY_NAMESPACE,
        top_k: int = 1,
    ):
        """Search only the company-policy namespace."""
        if namespace != COMPANY_POLICY_NAMESPACE:
            raise ValueError("Company-policy search must use namespace 'company-policy'.")
        vector = self.embedder.embed_query(query)
        return self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )

    def similarity_search_by_vector(
        self,
        vector: list[float],
        *,
        namespace: str = COMPANY_POLICY_NAMESPACE,
        top_k: int = 5,
    ):
        """Search only the company-policy namespace using a supplied vector."""
        if namespace != COMPANY_POLICY_NAMESPACE:
            raise ValueError("Company-policy search must use namespace 'company-policy'.")
        if not vector:
            raise ValueError("Search vector must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        logger.info(
            f"Performing vector-based similarity search in namespace '{namespace}'."
        )
        return self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
