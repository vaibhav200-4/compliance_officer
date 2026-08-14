"""
RBI KYC semantic chunker.

Reads:
    rbi_kyc_requirements_v2.json

Creates:
    rbi_kyc_chunks.json

Each regulatory requirement becomes one semantic chunk.
This preserves the requirement, evidence expectations, assessment questions,
and metadata needed later for embeddings + Supabase pgvector.
"""

import json
from pathlib import Path


INPUT_FILE = Path("data/rbi_kyc_requirements_v2.json")
OUTPUT_FILE = Path("sample_chunking/rbi_kyc_chunks.json")


def build_embedding_text(requirement: dict) -> str:
    """Build the text that will later be embedded."""
    evidence = "; ".join(requirement.get("evidence_expected", []))
    questions = " ".join(requirement.get("assessment_questions", []))

    return (
        f"Framework: {requirement.get('framework', 'RBI KYC')}\n"
        f"Jurisdiction: India\n"
        f"Control Domain: {requirement.get('control_domain', '')}\n"
        f"Regulation: {requirement.get('regulation', '')}\n"
        f"Requirement: {requirement.get('requirement', '')}\n"
        f"Evidence Expected: {evidence}\n"
        f"Assessment Questions: {questions}"
    )


def create_chunk(requirement: dict, source_metadata: dict) -> dict:
    """Convert one requirement into one semantic RAG chunk."""
    requirement_id = requirement["requirement_id"]

    return {
        "chunk_id": requirement_id,
        "framework": source_metadata.get("framework", "RBI KYC"),
        "jurisdiction": source_metadata.get("jurisdiction", "India"),
        "authority": source_metadata.get("authority", "Reserve Bank of India"),
        "source_document": source_metadata.get("source_document", ""),
        "source_version": source_metadata.get("source_version", ""),
        "source_type": source_metadata.get("source_type", ""),
        "regulation": requirement.get("regulation", ""),
        "control_domain": requirement.get("control_domain", ""),
        "requirement": requirement.get("requirement", ""),
        "evidence_expected": requirement.get("evidence_expected", []),
        "assessment_questions": requirement.get("assessment_questions", []),
        "compliance_status": requirement.get(
            "compliance_status",
            ["COMPLIANT", "PARTIAL", "NON_COMPLIANT", "NOT_ASSESSED"],
        ),
        "text_for_embedding": build_embedding_text(requirement),
    }


def chunk_requirements(data: dict) -> list[dict]:
    """Create one semantic chunk per regulatory requirement."""
    source_metadata = {
        "framework": data.get("framework", "RBI KYC"),
        "jurisdiction": data.get("jurisdiction", "India"),
        "authority": data.get("authority", "Reserve Bank of India"),
        "source_document": data.get("source_document", ""),
        "source_version": data.get("source_version", ""),
        "source_type": data.get("source_type", ""),
    }

    requirements = data.get("requirements", [])

    if not isinstance(requirements, list):
        raise ValueError("'requirements' must be a list")

    chunks = []

    for requirement in requirements:
        if not isinstance(requirement, dict):
            raise ValueError("Every requirement must be a JSON object")

        if "requirement_id" not in requirement:
            raise ValueError("Every requirement must contain 'requirement_id'")

        chunks.append(create_chunk(requirement, source_metadata))

    return chunks


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run this script from the project root."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = chunk_requirements(data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "framework": data.get("framework", "RBI KYC"),
        "source_document": data.get("source_document", ""),
        "source_version": data.get("source_version", ""),
        "chunking_strategy": "one semantic chunk per regulatory requirement",
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("RBI KYC SEMANTIC CHUNKER")
    print("=" * 60)
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Chunks: {len(chunks)}")
    print()
    print("Chunking completed successfully.")


if __name__ == "__main__":
    main()