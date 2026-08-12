#app/ingestion/embedder.py
from google import genai

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class Embedder:
    """
    Handles embedding generation using Google's embedding model.
    """

    def __init__(self):
        logger.info("Initializing embedding model...")

        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)

        logger.success("Embedding model initialized.")

    def get_embedding_model(self):
        """
        Returns the embedding model instance.
        """
        return self

    def embed_query(self, query: str):
        """
        Generate embedding for a user query.
        """
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=query,
            config={"output_dimensionality": settings.EMBEDDING_DIMENSION},
        )
        return response.embeddings[0].values

    def embed_documents(self, texts: list[str]):
        """
        Generate embeddings for multiple text chunks.
        """
        if not texts:
            return []

        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=texts,
            config={"output_dimensionality": settings.EMBEDDING_DIMENSION},
        )
        return [embedding.values for embedding in response.embeddings]
