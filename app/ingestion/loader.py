#app/ingestion/loader.py
from pathlib import Path

from langchain_core.documents import Document

from app.core.logger import get_logger
from app.ingestion.extractors.mistral_extractor import MistralExtractor
from app.ingestion.extractors.text_extractor import TextExtractor

logger = get_logger()


class DocumentLoader:
    """
    Routes documents to the appropriate extractor.

    Supported Formats
    -----------------
    OCR:
        PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP

    Office:
        DOCX, PPTX

    Tabular:
        CSV, XLSX

    Text:
        TXT, MD
    """

    def __init__(self):
        mistral = MistralExtractor()
        text = TextExtractor()

        self._extractors = {
            # OCR
            ".pdf": mistral,
            ".png": mistral,
            ".jpg": mistral,
            ".jpeg": mistral,
            ".tiff": mistral,
            ".bmp": mistral,
            ".webp": mistral,

            # Office
            ".docx": None,
            ".pptx": None,

            # Tabular
            ".csv": None,
            ".xlsx": None,

            # Text
            ".txt": text,
            ".md": text,
        }

    def load(self, file_path: str | Path) -> list[Document]:
        """
        Load a document and return LangChain Documents.
        """

        file_path = Path(file_path)

        self._validate(file_path)

        suffix = file_path.suffix.lower()

        extractor = self._get_extractor(suffix)

        if extractor is None:
            supported = ", ".join(sorted(self._extractors.keys()))
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported formats: {supported}"
            )

        logger.info(
            f"Loading '{file_path.name}' using "
            f"{extractor.__class__.__name__}"
        )

        documents = extractor.extract(file_path)

        logger.success(
            f"Loaded {len(documents)} document(s) from "
            f"'{file_path.name}'."
        )

        return documents

    def _get_extractor(self, suffix: str):
        """Load optional extractors only when their file type is requested."""
        if suffix not in self._extractors:
            return None

        extractor = self._extractors[suffix]
        if extractor is not None:
            return extractor

        if suffix in {".docx", ".pptx"}:
            from app.ingestion.extractors.office_extractor import OfficeExtractor

            extractor = OfficeExtractor()
        elif suffix in {".csv", ".xlsx"}:
            from app.ingestion.extractors.tabular_extractor import TabularExtractor

            extractor = TabularExtractor()
        else:
            return None

        self._extractors[suffix] = extractor
        return extractor

    @staticmethod
    def _validate(file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"{file_path} is not a valid file."
            )
