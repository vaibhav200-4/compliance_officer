from pathlib import Path

import pandas as pd

from langchain_core.documents import Document


class TabularExtractor:

    def extract(self, file_path: Path):

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(file_path)

        else:
            df = pd.read_excel(file_path)

        text = df.to_markdown(index=False)

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "page": 1,
                },
            )
        ]