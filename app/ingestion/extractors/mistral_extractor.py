import json
import time
from datetime import datetime
from pathlib import Path
from mistralai.client import Mistral
from langchain_core.documents import Document

from app.ingestion.cleaner import TextCleaner
from app.core.logger import get_logger

logger = get_logger()


class MistralExtractor:
    """
    Extract text from PDFs using Mistral OCR.

    Returns:
        [
            {
                "page": 0,
                "text": "..."
            },
            ...
        ]
    """

    MODEL = "mistral-ocr-latest"
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 2

    def __init__(
        self,
        api_key: str | None = None,
        save_json: bool = True,
        json_output_dir: str | Path = "data/ocr_output",
    ):
        from app.core.config import settings

        api_key = api_key or settings.MISTRAL_API_KEY

        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable is not set."
            )

        self.client = Mistral(api_key=api_key)
        self.cleaner = TextCleaner()

        self.save_json = save_json
        self.json_output_dir = Path(json_output_dir)

        if self.save_json:
            self.json_output_dir.mkdir(parents=True, exist_ok=True)

        logger.success("Mistral OCR initialized.")

    def _upload_pdf(self, pdf_path: Path):
        """
        Upload PDF to Mistral.
        """

        return self.client.files.upload(
            file={
                "file_name": pdf_path.name,
                "content": pdf_path.read_bytes(),
            },
            purpose="ocr",
        )

    def _get_signed_url(self, file_id: str) -> str:
        """
        Generate temporary signed URL.
        """

        signed = self.client.files.get_signed_url(
            file_id=file_id,
            expiry=1,
        )

        return signed.url

    def _call_ocr(self, document_url: str):

        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                return self.client.ocr.process(
                    model=self.MODEL,
                    document={
                        "type": "document_url",
                        "document_url": document_url,
                    },
                    include_image_base64=False,
                )

            except Exception as exc:

                last_error = exc

                wait = attempt * self.RETRY_BACKOFF_SECONDS

                logger.warning(
                    f"OCR attempt {attempt}/{self.MAX_RETRIES} failed: {exc}"
                )

                time.sleep(wait)

        logger.exception("OCR failed after all retries.")

        raise last_error

    def extract(self, pdf_path: str | Path) -> list[Document]:

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"MistralExtractor only accepts PDF files: {pdf_path}")

        logger.info(f"Uploading {pdf_path.name}...")

        uploaded = self._upload_pdf(pdf_path)

        logger.info("Generating signed URL...")

        document_url = self._get_signed_url(uploaded.id)

        logger.info("Running OCR...")

        response = self._call_ocr(document_url)

        response_dict = response.model_dump()

        if self.save_json:
            self._persist_json(pdf_path, response_dict)

        documents = []

        for page in response_dict.get("pages", []):

            text = self.cleaner.clean(
                (page.get("markdown") or "").strip()
            )

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page.get("index", 0) + 1,
                    },
                )
            )

        if not documents:
            raise ValueError(f"Mistral OCR returned no usable text for: {pdf_path.name}")

        return documents


    def _persist_json(
        self,
        pdf_path: Path,
        response_dict: dict,
    ) -> Path:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_file = (
            self.json_output_dir /
            f"{pdf_path.stem}_{timestamp}.json"
        )

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                response_dict,
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Saved OCR JSON -> {out_file}")

        return out_file
