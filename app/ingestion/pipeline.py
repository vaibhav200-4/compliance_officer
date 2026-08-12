#app/ingestion/pipeline.py
from pathlib import Path
from uuid import uuid4

from app.core.logger import get_logger
from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import DocumentSplitter
from app.vectorstore.pinecone import COMPANY_POLICY_NAMESPACE, PineconeManager

logger = get_logger()


class IngestionPipeline:

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()
        self.vector_store = PineconeManager()

    def ingest(
        self,
        file_paths: str | Path | list[str | Path],
    ):

        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        all_documents = []
        processed_files = []
        document_ids = []

        logger.info(
            f"Starting ingestion of {len(file_paths)} file(s)."
        )

        for file_path in file_paths:

            file_path = Path(file_path)

            logger.info(
                f"Processing '{file_path.name}'"
            )

            # Unique ID for this uploaded document
            document_id = str(uuid4())

            documents = self.loader.load(file_path)

            # Attach metadata BEFORE splitting
            for document in documents:

                document.metadata.update(
                    {
                        "document_id": document_id,
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                    }
                )

            all_documents.extend(documents)
            processed_files.append(file_path.name)
            document_ids.append(document_id)

            logger.info(
                f"Assigned document_id={document_id} "
                f"to '{file_path.name}'"
            )

        chunks = self.splitter.split(all_documents)

        if not chunks:
            raise ValueError("No non-empty chunks were created; nothing was indexed.")

        ids = self.vector_store.add_documents(
            chunks,
            namespace=COMPANY_POLICY_NAMESPACE,
        )

        logger.success(
            f"Ingestion completed. "
            f"Files={len(processed_files)} "
            f"Documents={len(all_documents)} "
            f"Chunks={len(chunks)}"
        )

        return {
            "files": processed_files,
            "documents": len(all_documents),
            "chunks": len(chunks),
            "vector_ids": ids,
            "document_ids": document_ids,
            "namespace": COMPANY_POLICY_NAMESPACE,
        }
