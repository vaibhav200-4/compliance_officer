from pathlib import Path

from langchain_core.documents import Document


class TextExtractor:

    def extract(self, file_path: Path):

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "page": 1,
                },
            )
        ]