#app/ingestion/splitter.py
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class DocumentSplitter:
    """
    Splits LangChain documents into smaller chunks.
    """

    def __init__(self) -> None:
        if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        self.text_splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
                length_function=len,
            )
        )

    def split(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Split LangChain documents into chunks.

        Args:
            documents: Documents to split.

        Returns:
            Chunked documents.
        """

        if not documents:
            logger.warning("No documents to split.")
            return []

        logger.info(
            f"Splitting {len(documents)} document(s)..."
        )

        chunks = self.text_splitter.split_documents(documents)

        chunk_indexes: dict[str, int] = {}
        for chunk in chunks:
            document_id = chunk.metadata.get("document_id")
            if not document_id:
                raise ValueError("Every document must have document_id before splitting.")

            chunk_index = chunk_indexes.get(document_id, 0)
            chunk_indexes[document_id] = chunk_index + 1

            chunk.metadata.update(
                {
                    "chunk_id": f"{document_id}:{chunk_index}",
                    "chunk_index": chunk_index,
                }
            )

        logger.success(
            f"Created {len(chunks)} chunks from {len(documents)} document(s)."
        )

        return chunks
