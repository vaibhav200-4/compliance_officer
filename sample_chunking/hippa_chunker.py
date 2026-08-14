import json
from pathlib import Path


INPUT_FILE = Path("data/hipaa_security_requirements.json")
OUTPUT_FILE = Path("sample_chunking/hipaa_security_chunks.json")





def build_embedding_text(requirement, source):
    evidence = "; ".join(requirement.get("evidence_expected", []))
    questions = " ".join(requirement.get("assessment_questions", []))

    return (
        f"Framework: {requirement.get('framework', source.get('framework', 'HIPAA'))}\n"
        f"Jurisdiction: {source.get('jurisdiction', 'United States')}\n"
        f"Control Domain: {requirement.get('control_domain', '')}\n"
        f"Regulation: {requirement.get('regulation', '')}\n"
        f"Requirement: {requirement.get('requirement', '')}\n"
        f"Evidence Expected: {evidence}\n"
        f"Assessment Questions: {questions}"
    )


def create_chunks(data):
    chunks = []

    requirements = data.get("requirements", [])

    if not isinstance(requirements, list):
        raise ValueError("'requirements' must be a list")

    for requirement in requirements:

        if "requirement_id" not in requirement:
            raise ValueError(
                "Every requirement must contain 'requirement_id'"
            )

        chunk = {
            "chunk_id": requirement["requirement_id"],
            "framework": data.get("framework", "HIPAA"),
            "jurisdiction": data.get(
                "jurisdiction",
                "United States"
            ),
            "source_document": data.get(
                "source_document",
                ""
            ),
            "source_version": data.get(
                "source_version",
                ""
            ),
            "source_type": data.get(
                "source_type",
                ""
            ),
            "scope": data.get(
                "scope",
                ""
            ),
            "control_domain": requirement.get(
                "control_domain",
                ""
            ),
            "regulation": requirement.get(
                "regulation",
                ""
            ),
            "requirement": requirement.get(
                "requirement",
                ""
            ),
            "evidence_expected": requirement.get(
                "evidence_expected",
                []
            ),
            "assessment_questions": requirement.get(
                "assessment_questions",
                []
            ),
            "compliance_status": requirement.get(
                "compliance_status",
                [
                    "COMPLIANT",
                    "PARTIAL",
                    "NON_COMPLIANT",
                    "NOT_ASSESSED"
                ]
            ),
            "text_for_embedding": build_embedding_text(
                requirement,
                data
            )
        }

        chunks.append(chunk)

    return chunks


def main():

    print("=" * 60)
    print("HIPAA SECURITY SEMANTIC CHUNKER")
    print("=" * 60)

    print(f"Input file : {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    # Check input
    if not INPUT_FILE.exists():
        print("\nERROR: Input file not found.")
        print("\nExpected file:")
        print(INPUT_FILE)
        print("\nMake sure your folder structure is:")
        print("GDPR_KB/")
        print("├── hippa_chunker.py")
        print("└── data/")
        print("    └── hipaa_security_requirements.json")
        return

    # Load JSON
    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # Create chunks
    chunks = create_chunks(data)

    # Make sure data directory exists
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save chunks
    output = {
        "framework": data.get(
            "framework",
            "HIPAA"
        ),
        "source_document": data.get(
            "source_document",
            ""
        ),
        "source_version": data.get(
            "source_version",
            ""
        ),
        "chunking_strategy":
            "one semantic chunk per regulatory/security requirement",
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()