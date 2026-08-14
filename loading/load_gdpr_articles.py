import argparse
import json
from pathlib import Path


def flatten_gdpr_articles(json_path, output_path):
    input_path = Path(json_path)
    output_path = Path(output_path)

    articles = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    chunks = []

    for article in articles:
        article_number = article["article_number"]
        article_name = article["article_name"]
        chapter_number = article["chapter_number"]

        recital_references = article.get(
            "recital_references", []
        )

        compliance_checkability = article.get(
            "compliance_checkability",
            "unknown"
        )

        for section in article["sections"]:
            section_name = section["section_name"]
            text = section["text"].strip()

            chunk_id = (
                f"gdpr_article_{article_number}_"
                f"{section_name.replace(' ', '_')}"
                .replace("(", "_")
                .replace(")", "_")
            )

            chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "source": "GDPR",
                    "document_type": "regulation",
                    "article_number": article_number,
                    "article_name": article_name,
                    "chapter_number": chapter_number,
                    "section_name": section_name,
                    "recital_references": recital_references,
                    "compliance_checkability": compliance_checkability
                }
            })

    output_path.write_text(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"Created {len(chunks)} chunks")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten GDPR articles into section-level chunks"
    )

    parser.add_argument(
        "--json",
        required=True,
        help="Path to gdpr_articles.json"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Path to output gdpr_chunks.json"
    )

    args = parser.parse_args()

    flatten_gdpr_articles(
        args.json,
        args.out
    )


if __name__ == "__main__":
    main()