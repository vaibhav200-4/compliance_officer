#app/ingestion/cleaner.py
import re


class TextCleaner:
    """Clean OCR output while preserving Markdown formatting."""

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing spaces
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        # Collapse excessive blank lines (keep markdown readable)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove multiple spaces (not newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)

        return text.strip()